#include "rnbt/driver.hpp"
#include "rnbt/workload.hpp"

#include <cmath>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <ranges>
#include <reclaimnet.grpc.pb.h>
#include <thread>

namespace rnbt {

namespace {

[[nodiscard]] std::int64_t wall_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] std::string var(const Vars &vars, std::string_view key) {
  const auto found = vars.find(key);
  return found == vars.end() ? std::string{} : found->second;
}

[[nodiscard]] std::string state_path(const Vars &vars, std::string_view instance,
                                     std::uint32_t rank) {
  return std::format("{}/{}/r{}", var(vars, "state_root"), instance, rank);
}

[[nodiscard]] std::string fixed(double value) { return std::format("{:.3f}", value); }

[[nodiscard]] Vars workload_shell_vars(const Remote &remote, const HostConfig &host) {
  const auto node = remote.node_vars(host);
  Vars vars;
  for (const char *key : {"load_binary", "profile_path", "workload_state", "workload_log"})
    vars.emplace(key, shell_quote(var(node, key)));
  return vars;
}

[[nodiscard]] std::string period_name(std::uint32_t period) {
  return std::format("period-{:04}", period);
}

}

Driver::Driver(const LoadedReplay &replay, const Scenario &scenario, const Plan &plan,
               std::filesystem::path directory, bool dry_run)
    : replay_{replay}, scenario_{scenario}, plan_{plan}, directory_{std::move(directory)},
      dry_run_{dry_run}, remote_{replay.testbed},
      wall_origin_ns_{wall_now_ns() +
                      static_cast<std::int64_t>(replay.config.timing.start_delay_s * 1e9)} {
  if (!dry_run_)
    channels_ =
        std::make_unique<Channels>(replay.testbed.credentials, replay.testbed.rpc_timeout_s);
}

std::int64_t Driver::wall_of(Instant trace_at) const noexcept {
  return wall_origin_ns_ + (ns_of(trace_at) - ns_of(plan_.origin));
}

std::vector<const HostConfig *> Driver::hosts() const {
  std::vector<const HostConfig *> out;
  for (const auto &node : plan_.nodes)
    if (const auto *host = replay_.testbed.node(node))
      out.push_back(host);
  return out;
}

bool Driver::sleep_until(std::int64_t wall_ns, std::stop_token stop) const {
  std::mutex mutex;
  std::condition_variable_any cv;
  std::unique_lock lock{mutex};
  const auto deadline = std::chrono::system_clock::time_point{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{wall_ns})};
  return !cv.wait_until(lock, stop, deadline, [] { return false; }) && !stop.stop_requested();
}

void Driver::run_commands(std::span<const Expected<Command>> commands, std::string_view step,
                          ActionRecord &record) {
  std::vector<Command> ready;
  for (const auto &command : commands) {
    if (!command) {
      record.ok = false;
      record.errors.push_back(std::format("{}: {}", step, command.error().message));
      continue;
    }
    ready.push_back(*command);
  }
  Json trail = Json::array();
  if (dry_run_) {
    for (const auto &command : ready)
      trail.push_back({{"target", command.label}, {"line", command.line}});
    record.details[std::string{step}] = std::move(trail);
    return;
  }
  const auto results = run_parallel(ready, replay_.testbed.parallelism, remote_.timeout());
  for (auto &&[command, result] : std::views::zip(ready, results)) {
    trail.push_back(
        {{"target", command.label},
         {"exit", result.exit_code},
         {"seconds", result.seconds},
         {"timed_out", result.timed_out},
         {"output", result.output.size() > 2000 ? result.output.substr(0, 2000) : result.output}});
    if (!result.ok()) {
      record.ok = false;
      record.errors.push_back(std::format("{} on {} exited {}{}", step, command.label,
                                          result.exit_code, result.timed_out ? " (timeout)" : ""));
    } else if (step == "workload_status") {
      const auto state = Json::parse(result.output, nullptr, false);
      if (state.is_discarded() || !state.is_object() ||
          (state.value("status", std::string{}) != "running" &&
           state.value("status", std::string{}) != "completed")) {
        record.ok = false;
        record.errors.push_back(
            std::format("workload actor on {} is not running or completed", command.label));
      }
    }
  }
  record.details[std::string{step}] = std::move(trail);
}

Status Driver::validate_workloads(std::span<const HostConfig *const> nodes) const {
  if (!replay_.config.workload_profiles)
    return {};
  for (const auto *host : nodes) {
    const auto *spec = scenario_.find_node(host->trace_node);
    if (!spec || spec->gpu_count == 0)
      return failure(std::format("{} has no positive trace GPU count", host->trace_node));
    const auto path =
        std::filesystem::path{*replay_.config.workload_profiles} / (host->trace_node + ".json");
    const auto profile = job::WorkloadProfile::load(path);
    if (!profile)
      return std::unexpected{profile.error()};
    if (profile->trace_node() != host->trace_node ||
        profile->window_start_ns() != ns_of(plan_.from) ||
        profile->window_end_ns() != ns_of(plan_.to))
      return failure(std::format("{} must match node {} and selected replay window [{}, {}]",
                                 path.string(), host->trace_node, ns_of(plan_.from),
                                 ns_of(plan_.to)));
  }
  return {};
}

Status Driver::start_workloads(std::span<const HostConfig *const> nodes, ActionRecord &record) {
  if (!replay_.config.workload_profiles)
    return {};
  if (auto valid = validate_workloads(nodes); !valid) {
    record.ok = false;
    record.errors.push_back(valid.error().message);
    return valid;
  }
  std::vector<Expected<Command>> prepare, upload, launch;
  for (const auto *host : nodes) {
    const auto vars = remote_.node_vars(*host);
    const auto profile_path = var(vars, "profile_path");
    const auto workload_state = var(vars, "workload_state");
    const auto workload_log = var(vars, "workload_log");
    prepare.push_back(remote_.on_node(
        *host, "mkdir -p {profile_parent} {state_parent} {log_parent}",
        Vars{{"profile_parent",
              shell_quote(std::filesystem::path{profile_path}.parent_path().string())},
             {"state_parent",
              shell_quote(std::filesystem::path{workload_state}.parent_path().string())},
             {"log_parent",
              shell_quote(std::filesystem::path{workload_log}.parent_path().string())}}));
    const auto local =
        std::filesystem::path{*replay_.config.workload_profiles} / (host->trace_node + ".json");
    upload.push_back(remote_.upload(*host, local, profile_path));
    auto launch_vars = workload_shell_vars(remote_, *host);
    launch_vars.emplace("gpu_count",
                        std::to_string(scenario_.find_node(host->trace_node)->gpu_count));
    launch_vars.emplace("workload_origin_ns", std::to_string(wall_of(plan_.from)));
    launch.push_back(remote_.on_node(*host, replay_.testbed.commands.workload_start, launch_vars));
  }
  run_commands(prepare, "workload_prepare", record);
  if (!record.ok)
    return failure("cannot prepare physical workload directories");
  run_commands(upload, "workload_upload", record);
  if (!record.ok)
    return failure("cannot upload physical workload profiles");
  run_commands(launch, "workload_start", record);
  if (!record.ok)
    return failure("physical GPU workload actor startup failed");
  for (const auto *host : nodes)
    running_workloads_.insert(host->trace_node);
  next_workload_health_check_ = {};
  return {};
}

Status Driver::check_workloads(std::span<const HostConfig *const> nodes, ActionRecord &record,
                               bool force) {
  if (!replay_.config.workload_profiles)
    return {};
  if (!force && std::chrono::steady_clock::now() < next_workload_health_check_)
    return {};
  std::vector<Expected<Command>> checks;
  for (const auto *host : nodes)
    if (running_workloads_.contains(host->trace_node) && !offline_.contains(host->trace_node))
      checks.push_back(remote_.on_node(*host, "{env} {load_binary} status --state {workload_state}",
                                       workload_shell_vars(remote_, *host)));
  if (!checks.empty()) {
    run_commands(checks, "workload_status", record);
    next_workload_health_check_ = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  }
  return record.ok ? Status{} : Status{failure("physical GPU workload health check failed")};
}

Status Driver::stop_workloads(std::span<const HostConfig *const> nodes, ActionRecord &record,
                              bool check_health) {
  if (!replay_.config.workload_profiles)
    return {};
  if (check_health)
    (void)check_workloads(nodes, record, true);
  std::vector<Expected<Command>> stop;
  for (const auto *host : nodes)
    stop.push_back(remote_.on_node(*host, replay_.testbed.commands.workload_stop,
                                   workload_shell_vars(remote_, *host)));
  run_commands(stop, "workload_stop", record);
  if (!record.ok)
    return failure("physical GPU workload stop or health check failed");
  for (const auto *host : nodes)
    running_workloads_.erase(host->trace_node);
  return {};
}

Status Driver::wait_healthy(std::span<const HostConfig *const> nodes, std::stop_token stop) {
  if (dry_run_ || nodes.empty())
    return {};
  const auto until = std::chrono::steady_clock::now() +
                     std::chrono::duration<double>{replay_.config.timing.health_timeout_s};
  std::set<std::string> pending;
  for (const auto *host : nodes)
    pending.insert(host->trace_node);
  while (!pending.empty() && !stop.stop_requested()) {
    for (auto it = pending.begin(); it != pending.end();) {
      const auto *host = replay_.testbed.node(*it);
      const auto healthy = channels_->healthy(host->node_daemon_address);
      it = healthy && *healthy ? pending.erase(it) : std::next(it);
    }
    if (pending.empty() || std::chrono::steady_clock::now() > until)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds{500});
  }
  if (!pending.empty()) {
    std::string names;
    for (const auto &name : pending)
      names += (names.empty() ? "" : ", ") + name;
    return failure(std::format("node daemons not healthy: {}", names));
  }
  return {};
}

Status Driver::testbed_reset(std::uint32_t period, Instant warm_until, ActionRecord &record,
                             std::stop_token stop) {
  const auto &commands = replay_.testbed.commands;
  const auto nodes = hosts();
  if (auto valid = validate_workloads(nodes); !valid) {
    record.ok = false;
    record.errors.push_back(valid.error().message);
    return valid;
  }
  const auto on_nodes = [&](std::string_view pattern) {
    std::vector<Expected<Command>> out;
    for (const auto *host : nodes)
      out.push_back(remote_.on_node(*host, pattern));
    return out;
  };
  if (auto stopped = stop_workloads(nodes, record); !stopped)
    return stopped;
  run_commands(on_nodes(commands.job_kill_all), "kill_jobs", record);
  if (!record.ok)
    return failure("reset could not stop existing jobs");
  active_instances_.clear();
  run_commands(on_nodes(commands.node_daemon_stop), "node_daemon_stop", record);
  if (!record.ok)
    return failure("reset could not stop node daemons");
  const auto coordinator_period =
      (replay_.config.workload_profiles ? replay_.config.run_id + "/" : "") + period_name(period);
  const Vars coordinator_vars{{"period", coordinator_period}};
  const std::array coordinator_steps{
      std::pair{std::string_view{"coordinator_stop"}, std::string_view{commands.coordinator_stop}},
      std::pair{std::string_view{"coordinator_wipe"}, std::string_view{commands.coordinator_wipe}},
      std::pair{std::string_view{"coordinator_configure"},
                std::string_view{commands.coordinator_configure}},
      std::pair{std::string_view{"coordinator_start"},
                std::string_view{commands.coordinator_start}}};
  for (const auto &[step, pattern] : coordinator_steps) {
    const std::array command{remote_.on_coordinator(pattern, coordinator_vars)};
    run_commands(command, step, record);
    if (!record.ok)
      return failure(std::format("reset failed at {}", step));
  }
  run_commands(on_nodes(commands.node_wipe), "node_wipe", record);
  if (!record.ok)
    return failure("reset could not clear node state");
  run_commands(on_nodes("{env} {job_binary} probe --vram-bytes 1048576 --gpus 1"),
               "physical_gpu_preflight", record);
  if (!record.ok)
    return failure("physical GPU preflight failed");
  if (!dry_run_)
    std::this_thread::sleep_for(std::chrono::seconds{2});
  run_commands(on_nodes(commands.node_daemon_start), "node_daemon_start", record);
  if (!record.ok)
    return failure("reset could not start node daemons");
  offline_.clear();
  if (auto healthy = wait_healthy(nodes, stop); !healthy) {
    record.ok = false;
    record.errors.push_back(healthy.error().message);
    return failure("reset node daemon health check failed");
  }
  if (auto started = start_workloads(nodes, record); !started)
    return started;
  if (!dry_run_)
    std::this_thread::sleep_for(
        std::chrono::duration<double>{replay_.config.timing.registration_grace_s});
  std::uint64_t warmed = 0, warm_failed = 0;
  if (replay_.config.warm_hazard && !dry_run_)
    for (const auto &event : scenario_.reclaims) {
      if (event.at >= warm_until)
        break;
      if (!replay_.testbed.node(event.node))
        continue;
      auto reported = channels_->report_reclaim(replay_.testbed.coordinator.address, event.node,
                                                event.kind, event.notice, wall_of(event.at));
      reported ? ++warmed : ++warm_failed;
      if (!reported) {
        record.ok = false;
        record.errors.push_back(reported.error().message);
        break;
      }
    }
  record.details["hazard_events_warmed"] = warmed;
  record.details["hazard_events_failed"] = warm_failed;
  record.details["coordinator_trace"] =
      std::format("{}/trace/{}", replay_.testbed.paths.coordinator_data, coordinator_period);
  return record.ok ? Status{} : failure("reset incomplete");
}

Status Driver::testbed_stop(ActionRecord &record) {
  const auto &commands = replay_.testbed.commands;
  std::vector<Expected<Command>> kill, stop_node_daemons;
  const auto nodes = hosts();
  (void)stop_workloads(nodes, record);
  std::vector<Expected<Command>> clear_links;
  for (const auto *host : nodes) {
    kill.push_back(remote_.on_node(*host, commands.job_kill_all));
    stop_node_daemons.push_back(remote_.on_node(*host, commands.node_daemon_stop));
    if (replay_.config.replay_link_capacity)
      clear_links.push_back(remote_.on_node(*host, commands.link_clear));
  }
  run_commands(kill, "kill_jobs", record);
  run_commands(stop_node_daemons, "node_daemon_stop", record);
  if (!clear_links.empty())
    run_commands(clear_links, "link_clear", record);
  const std::array coordinator{remote_.on_coordinator(commands.coordinator_stop)};
  run_commands(coordinator, "coordinator_stop", record);
  return record.ok ? Status{} : failure("stop incomplete");
}

void Driver::on(const ResetTestbed &reset, const Action &action, ActionRecord &record,
                std::stop_token stop) {
  const auto &period = plan_.periods.at(reset.period);
  (void)action;
  (void)testbed_reset(reset.period, period.start, record, stop);
}

void Driver::on(const Depart &depart, const Action &, ActionRecord &record, std::stop_token) {
  const auto *host = replay_.testbed.node(depart.node);
  const auto &commands = replay_.testbed.commands;
  const std::array<const HostConfig *, 1> one{host};
  if (auto stopped = stop_workloads(one, record); !stopped)
    return;
  const std::array kill{remote_.on_node(*host, commands.job_kill_all)};
  run_commands(kill, "kill_jobs", record);
  if (!record.ok)
    return;
  const std::array stop_node_daemon{remote_.on_node(*host, commands.node_daemon_stop)};
  run_commands(stop_node_daemon, "node_daemon_stop", record);
  if (!record.ok)
    return;
  const std::array wipe{remote_.on_node(*host, commands.node_wipe)};
  run_commands(wipe, "node_wipe", record);
  if (record.ok)
    offline_.insert(depart.node);
}

void Driver::on(const Return &back, const Action &, ActionRecord &record, std::stop_token stop) {
  const auto *host = replay_.testbed.node(back.node);
  const std::array probe{
      remote_.on_node(*host, "{env} {job_binary} probe --vram-bytes 1048576 --gpus 1")};
  run_commands(probe, "physical_gpu_preflight", record);
  if (!record.ok)
    return;
  const std::array start{remote_.on_node(*host, replay_.testbed.commands.node_daemon_start)};
  run_commands(start, "node_daemon_start", record);
  if (!record.ok)
    return;
  const std::array<const HostConfig *, 1> one{host};
  if (auto healthy = wait_healthy(one, stop); !healthy) {
    record.ok = false;
    record.errors.push_back(healthy.error().message);
  }
  if (record.ok)
    offline_.erase(back.node);
  if (record.ok)
    (void)start_workloads(one, record);
}

void Driver::on(const StartInstance &start, const Action &, ActionRecord &record, std::stop_token) {
  const auto index = scenario_.job_index(start.job);
  if (!index) {
    record.ok = false;
    record.errors.push_back(std::format("unknown job {}", start.job));
    return;
  }
  const auto &job = scenario_.jobs.at(*index);
  for (const auto &node : start.nodes)
    if (offline_.contains(node)) {
      record.ok = false;
      record.errors.push_back(std::format("{} is offline at the scripted start", node));
      return;
    }
  const auto vram = job.per_rank_vram();
  const auto park = start.park_after_resume
                        ? std::format(" --exit-after-resume-s {}",
                                      fixed(replay_.config.timing.episode_post_resume_s))
                        : std::string{};
  std::vector<Expected<Command>> launches;
  for (std::uint32_t rank = 0; rank < start.nodes.size(); ++rank) {
    const auto *host = replay_.testbed.node(start.nodes[rank]);
    const auto launch_pattern = replay_.testbed.commands.job_launch +
                                (replay_.config.workload_profiles ? " --external-load true" : "");
    launches.push_back(remote_.on_node(
        *host, launch_pattern,
        Vars{{"state_path", state_path(remote_.node_vars(*host), start.instance, rank)},
             {"instance", start.instance},
             {"rank", std::to_string(rank)},
             {"target_s", fixed(start.target_s)},
             {"progress_s", fixed(start.progress_s)},
             {"checkpoint_bytes", std::to_string(job.per_rank_checkpoint_bytes)},
             {"vram_bytes", std::to_string(vram)},
             {"gpus", std::to_string(job.per_node_gpus)},
             {"park", park}}));
  }
  run_commands(launches, "launch", record);
  if (!record.ok)
    return;
  if (dry_run_) {
    active_instances_.insert(start.instance);
    return;
  }
  for (std::uint32_t rank = static_cast<std::uint32_t>(start.nodes.size()); rank-- > 0;) {
    const auto *host = replay_.testbed.node(start.nodes[rank]);
    const auto vars = remote_.node_vars(*host);
    const auto log_dir = var(vars, "log_dir");
    const auto job_binary = var(vars, "job_binary");
    reclaimnet::proto::RuntimeRegistration registration;
    auto *runtime = registration.mutable_runtime();
    runtime->set_job_id(start.instance);
    runtime->set_state_path(state_path(vars, start.instance, rank));
    runtime->set_backend("hook");
    runtime->set_state_directory(true);
    runtime->set_memory_limit_bytes(vram);
    const auto add = [](auto *field, std::initializer_list<std::string> values) {
      for (const auto &value : values)
        field->Add(std::string{value});
    };
    add(runtime->mutable_checkpoint_command(),
        {job_binary, "checkpoint", "--state", "{state_path}", "--out", "{checkpoint_path}"});
    add(runtime->mutable_restore_command(), {job_binary, "restore", "--state", "{state_path}",
                                             "--in", "{checkpoint_path}", "--log-dir", log_dir});
    add(runtime->mutable_start_command(),
        {job_binary, "start", "--state", "{state_path}", "--instance", start.instance, "--rank",
         std::to_string(rank), "--log-dir", log_dir, "--target-s", fixed(start.target_s),
         "--checkpoint-bytes", std::to_string(job.per_rank_checkpoint_bytes), "--vram-bytes",
         std::to_string(vram), "--gpus", std::to_string(job.per_node_gpus)});
    if (start.park_after_resume)
      add(runtime->mutable_start_command(),
          {"--exit-after-resume-s", fixed(replay_.config.timing.episode_post_resume_s)});
    if (replay_.config.workload_profiles)
      add(runtime->mutable_start_command(), {"--external-load", "true"});
    add(runtime->mutable_pause_command(), {job_binary, "pause", "--state", "{state_path}"});
    add(runtime->mutable_resume_command(), {job_binary, "resume", "--state", "{state_path}"});
    add(runtime->mutable_stop_command(), {job_binary, "stop", "--state", "{state_path}"});
    auto *spec = registration.mutable_job();
    spec->set_job_id(start.instance);
    spec->set_node_id(start.nodes.front());
    spec->set_vram_bytes(job.logical_vram_bytes);
    spec->set_checkpoint_size(job.per_rank_checkpoint_bytes * job.gang_size);
    spec->set_lambda_e(job.lambda_i);
    registration.set_loss_budget_s(job.loss_budget_s);
    registration.set_cuda_capability_required(job.min_cuda);
    registration.set_gpu_count_required(job.per_node_gpus);
    registration.set_remaining_runtime_s(std::max(1.0, start.target_s - start.progress_s));
    for (const auto &node : start.nodes)
      registration.add_rank_nodes(node);
    registration.set_rank_count(job.gang_size);
    registration.set_rank(rank);
    if (auto registered = channels_->register_runtime(host->node_daemon_address, registration);
        !registered) {
      record.ok = false;
      record.errors.push_back(registered.error().message);
      return;
    }
  }
  active_instances_.insert(start.instance);
}

void Driver::on(const InjectReclaim &reclaim, const Action &, ActionRecord &record,
                std::stop_token) {
  const auto *host = replay_.testbed.node(reclaim.node);
  record.details["timestamp_ns"] = wall_now_ns();
  if (offline_.contains(reclaim.node)) {
    record.ok = false;
    record.errors.push_back(std::format("{} is already offline", reclaim.node));
    return;
  }
  if (dry_run_)
    return;
  if (auto sent =
          channels_->reclaim(host->node_daemon_address, reclaim.node, reclaim.kind, reclaim.notice,
                             record.details["timestamp_ns"].get<std::int64_t>());
      !sent) {
    record.ok = false;
    record.errors.push_back(sent.error().message);
  }
}

void Driver::on(const StopInstance &stop_instance, const Action &, ActionRecord &record,
                std::stop_token) {
  if (!active_instances_.contains(stop_instance.instance)) {
    record.details["stop_instance_skipped"] = "instance was already stopped";
    return;
  }
  std::vector<Expected<Command>> commands;
  for (const auto *host : hosts())
    if (!offline_.contains(host->trace_node))
      commands.push_back(remote_.on_node(*host, replay_.testbed.commands.job_stop_instance,
                                         Vars{{"instance", stop_instance.instance}}));
  run_commands(commands, "stop_instance", record);
  if (record.ok)
    active_instances_.erase(stop_instance.instance);
}

void Driver::on(const SetLinkCapacity &capacity, const Action &, ActionRecord &record,
                std::stop_token) {
  if (!replay_.config.replay_link_capacity)
    return;
  const auto *host = replay_.testbed.node(capacity.node);
  if (!host || !std::isfinite(capacity.bps) || capacity.bps < 0 ||
      capacity.bps >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    record.ok = false;
    record.errors.push_back(
        "link replay requires a mapped node and a finite nonnegative uint64 capacity");
    return;
  }
  const auto bps = static_cast<std::uint64_t>(std::round(capacity.bps));
  const auto &pattern =
      bps == 0 ? replay_.testbed.commands.link_block : replay_.testbed.commands.link_set_rate;
  const std::array command{
      remote_.on_node(*host, pattern, Vars{{"link_bps", std::to_string(bps)}})};
  run_commands(command, "link_capacity", record);
  record.details["capacity_target_bps"] = bps;
}

void Driver::on(const EndReplay &, const Action &, ActionRecord &record, std::stop_token) {
  const auto nodes = hosts();
  (void)stop_workloads(nodes, record, true);
  std::vector<Expected<Command>> kill, clear_links;
  for (const auto *host : nodes) {
    kill.push_back(remote_.on_node(*host, replay_.testbed.commands.job_kill_all));
    if (replay_.config.replay_link_capacity)
      clear_links.push_back(remote_.on_node(*host, replay_.testbed.commands.link_clear));
  }
  run_commands(kill, "kill_jobs_at_window_end", record);
  if (!clear_links.empty())
    run_commands(clear_links, "link_clear", record);
  if (record.ok)
    active_instances_.clear();
}

void Driver::execute(const Action &action, ActionRecord &record, std::stop_token stop) {
  const auto *stop_instance = std::get_if<StopInstance>(&action.body);
  const bool duplicate_stop = stop_instance && !active_instances_.contains(stop_instance->instance);
  if (!duplicate_stop && !std::holds_alternative<ResetTestbed>(action.body) &&
      !std::holds_alternative<EndReplay>(action.body))
    if (auto healthy = check_workloads(hosts(), record); !healthy)
      return;
  std::visit([&](const auto &body) { on(body, action, record, stop); }, action.body);
}

Expected<RunReport> Driver::run(std::stop_token stop) {
  std::error_code ec;
  std::filesystem::create_directories(directory_, ec);
  if (ec)
    return failure(std::format("cannot create {}: {}", directory_.string(), ec.message()));
  log_ = JsonlSink{directory_ / "driver.jsonl"};
  wall_origin_ns_ =
      wall_now_ns() + static_cast<std::int64_t>(replay_.config.timing.start_delay_s * 1e9);
  if (auto valid = validate_workloads(hosts()); !valid)
    return std::unexpected{valid.error()};
  if (auto written = write_json_file(directory_ / "plan.json", plan_.to_json(scenario_.calendar));
      !written)
    return std::unexpected{written.error()};
  Json manifest{
      {"run_id", replay_.config.run_id},
      {"dry_run", dry_run_},
      {"status", "running"},
      {"gpu_backend", "cuda"},
      {"telemetry_source", "nvml"},
      {"workload_profiles", encode(replay_.config.workload_profiles)},
      {"profile_wall_origin_ns",
       replay_.config.workload_profiles ? Json(wall_of(plan_.from)) : Json(nullptr)},
      {"target_sources",
       {{"gpu_utilization", replay_.config.workload_profiles ? "workload profile / trace telemetry"
                                                             : "job control loop"},
        {"link_capacity", replay_.config.replay_link_capacity ? "trace bandwidth_samples.jsonl"
                                                              : "physical network"}}},
      {"measured_source",
       {{"gpu", "real NVIDIA NVML"}, {"payload", "real CUDA device allocations"}}},
      {"wall_origin_ns", wall_origin_ns_},
      {"trace_origin_ns", ns_of(plan_.origin)},
      {"trace", replay_.config.trace},
      {"mode", std::string{name_of(plan_.mode)}},
      {"window",
       {{"from", scenario_.calendar.format(plan_.from)},
        {"to", scenario_.calendar.format(plan_.to)}}},
      {"config", replay_.raw_config},
      {"testbed", replay_.raw_testbed},
      {"plan_stats", encode(plan_.stats)},
      {"started_wall_ns", wall_now_ns()}};
  if (auto written = write_json_file(directory_ / "run_manifest.json", manifest); !written)
    return std::unexpected{written.error()};
  RunReport report{.directory = directory_, .wall_origin_ns = wall_origin_ns_};
  for (auto &&[index, action] : std::views::enumerate(plan_.actions)) {
    if (stop.stop_requested())
      break;
    const auto planned = wall_origin_ns_ + action.offset.count();
    ActionRecord record{.index = static_cast<std::uint64_t>(index),
                        .type = std::string{name_of(action.body)},
                        .action = encode_action(action),
                        .planned_wall_ns = planned};
    bool interrupted_wait = false;
    try {
      if (!dry_run_) {
        if (!replay_.config.workload_profiles || running_workloads_.empty()) {
          interrupted_wait = !sleep_until(planned, stop);
        } else {
          while (wall_now_ns() < planned && !stop.stop_requested()) {
            const auto checkpoint = std::min(planned, wall_now_ns() + std::int64_t{5'000'000'000});
            if (!sleep_until(checkpoint, stop)) {
              interrupted_wait = true;
              break;
            }
            const auto checked_at = wall_now_ns();
            if (auto healthy = check_workloads(hosts(), record); !healthy) {
              record.type = "workload_health";
              record.started_wall_ns = checked_at;
              record.details["waiting_for_action"] = std::string{name_of(action.body)};
              break;
            }
          }
          interrupted_wait = interrupted_wait || stop.stop_requested();
        }
      }
      if (!interrupted_wait && record.ok) {
        record.started_wall_ns = wall_now_ns();
        record.lateness_s =
            dry_run_ ? 0.0 : static_cast<double>(record.started_wall_ns - planned) / 1e9;
        execute(action, record, stop);
      }
    } catch (const std::exception &error) {
      record.ok = false;
      record.errors.push_back(error.what());
    }
    if (interrupted_wait && record.ok)
      break;
    if (record.started_wall_ns == 0)
      record.started_wall_ns = wall_now_ns();
    record.finished_wall_ns = wall_now_ns();
    report.max_lateness_s = std::max(report.max_lateness_s, record.lateness_s);
    ++report.actions;
    report.failed += record.ok ? 0 : 1;
    log_.write(record);
    log_.flush();
    if (!record.ok) {
      manifest["failed_action"] = {
          {"index", record.index}, {"type", record.type}, {"errors", record.errors}};
      break;
    }
  }
  if (!dry_run_ && !replay_.config.workload_profiles && report.failed == 0 &&
      !stop.stop_requested())
    sleep_until(wall_now_ns() + static_cast<std::int64_t>(replay_.config.timing.drain_s * 1e9),
                stop);
  if (!dry_run_ && (report.failed > 0 || stop.stop_requested())) {
    manifest["status"] = report.failed > 0 ? "failed" : "interrupted";
    manifest["actions"] = report.actions;
    manifest["failed_actions"] = report.failed;
    manifest["interrupted"] = stop.stop_requested();
    (void)write_json_file(directory_ / "run_manifest.json", manifest);
    ActionRecord cleanup{
        .index = report.actions, .type = "abort_cleanup", .started_wall_ns = wall_now_ns()};
    try {
      (void)testbed_stop(cleanup);
    } catch (const std::exception &error) {
      cleanup.ok = false;
      cleanup.errors.push_back(error.what());
    }
    cleanup.finished_wall_ns = wall_now_ns();
    log_.write(cleanup);
    log_.flush();
    manifest["cleanup"] = encode(cleanup);
  }
  manifest["finished_wall_ns"] = wall_now_ns();
  manifest["actions"] = report.actions;
  manifest["failed_actions"] = report.failed;
  manifest["max_lateness_s"] = report.max_lateness_s;
  manifest["interrupted"] = stop.stop_requested();
  manifest["status"] = report.failed > 0       ? "failed"
                       : stop.stop_requested() ? "interrupted"
                                               : "completed";
  if (auto written = write_json_file(directory_ / "run_manifest.json", manifest); !written)
    return std::unexpected{written.error()};
  if (stop.stop_requested() && report.failed == 0)
    return failure(std::format("replay interrupted; cleanup and run manifest saved in {}",
                               directory_.string()));
  return report;
}

Status collect(const LoadedReplay &replay, const std::filesystem::path &run_directory) {
  const Remote remote{replay.testbed};
  std::vector<Command> commands;
  const auto target = run_directory / "collected";
  std::error_code ec;
  std::filesystem::create_directories(target / "jobs", ec);
  if (ec)
    return failure(
        std::format("cannot create collection directory {}: {}", target.string(), ec.message()));
  std::filesystem::create_directories(target / "coordinator", ec);
  if (ec)
    return failure(
        std::format("cannot create collection directory {}: {}", target.string(), ec.message()));
  for (const auto &host : replay.testbed.nodes) {
    const auto local = (target / "jobs" / host.trace_node).string();
    std::filesystem::create_directories(local, ec);
    if (ec)
      return failure(std::format("cannot create collection directory {}: {}", local, ec.message()));
    auto command = remote.fetch(host.ssh, var(remote.node_vars(host), "log_dir"), local);
    if (!command)
      return std::unexpected{command.error()};
    commands.push_back(*command);
    if (replay.config.workload_profiles) {
      const auto workload_local = target / "workloads" / host.trace_node;
      std::filesystem::create_directories(workload_local, ec);
      if (ec)
        return failure(std::format("cannot create workload collection directory {}: {}",
                                   workload_local.string(), ec.message()));
      auto workload = remote.fetch(host.ssh, var(remote.node_vars(host), "workload_root") + "/.",
                                   workload_local.string());
      if (!workload)
        return std::unexpected{workload.error()};
      commands.push_back(*workload);
    }
  }
  const auto coordinator_trace =
      replay.testbed.paths.coordinator_data + "/trace" +
      (replay.config.workload_profiles ? "/" + replay.config.run_id : "");
  auto coordinator = remote.fetch(replay.testbed.coordinator.ssh, coordinator_trace,
                                  (target / "coordinator").string());
  if (!coordinator)
    return std::unexpected{coordinator.error()};
  commands.push_back(*coordinator);
  const auto results = run_parallel(commands, replay.testbed.parallelism, remote.timeout());
  Json summary = Json::array();
  std::size_t failed = 0;
  for (auto &&[command, result] : std::views::zip(commands, results)) {
    summary.push_back({{"line", command.line},
                       {"exit", result.exit_code},
                       {"timed_out", result.timed_out},
                       {"output", result.output}});
    failed += result.ok() ? 0 : 1;
  }
  if (auto written = write_json_file(target / "collect.json", summary); !written)
    return written;
  if (failed > 0)
    return failure(std::format("{} collection commands failed; see {}", failed,
                               (target / "collect.json").string()));
  return {};
}

Status render_deployment(const LoadedReplay &replay, const Scenario &scenario,
                         const std::filesystem::path &out) {
  std::error_code ec;
  std::filesystem::create_directories(out / "node_daemons", ec);
  std::filesystem::create_directories(out / "systemd", ec);
  const auto &testbed = replay.testbed;
  const Remote remote{testbed};
  const auto &paths = testbed.paths;
  const auto &tls = testbed.credentials;
  const auto port_of = [](std::string_view address) {
    const auto colon = address.rfind(':');
    return colon == std::string_view::npos ? std::string{"50052"}
                                           : std::string{address.substr(colon + 1)};
  };
  {
    std::ofstream file{out / "coordinator.yaml"};
    file << std::format(
        "listen_address: \"0.0.0.0:{}\"\nepoch_seconds: 30.0\nB_ckpt_bps: 7000000000\nmigration_share: 0.70\n"
        "p3_bmin_bps: 50000000\np1_beta: 0.20\np1_loss_budget_s: 18000.0\nalpha: 1.0\nbeta: 2.0\ngamma: 0.5\n"
        "sqlite_path: \"{}/reclaimnet.db\"\nprometheus_port: 9090\ntls_cert_path: \"{}\"\ntls_key_path: \"{}\"\n"
        "tls_ca_path: \"{}\"\nmtls_enabled: {}\nheartbeat_timeout_s: 30.0\ntopology_file: \"{}/etc/topology.json\"\n"
        "hazard_window_s: 3600\ncapacity_window_s: 300\ncapacity_confidence_factor: 0.8\np2_k_min: 2\n"
        "p2_load_limit_pct: 80\np3_min_useful_transfer_s: 1\nrestore_time_s: 5\ntrace_dir: {}/trace/initial\n"
        "run_id: {}\njwt_enabled: false\n",
        port_of(testbed.coordinator.address), paths.coordinator_data, tls.cert, tls.key, tls.ca,
        tls.tls ? "true" : "false", paths.reclaimnet, paths.coordinator_data, replay.config.run_id);
  }
  for (const auto &host : testbed.nodes) {
    const auto *spec = scenario.find_node(host.trace_node);
    if (!spec)
      return failure(std::format("{} is not a trace node", host.trace_node));
    std::ofstream file{out / "node_daemons" / std::format("{}.yaml", host.trace_node)};
    file << std::format(
        "node_id: \"{}\"\nbuilding: \"{}\"\nswitch_id: \"{}\"\ncoordinator_address: \"{}\"\n"
        "listen_address: \"0.0.0.0:{}\"\nadvertise_address: \"{}\"\ntotal_vram_bytes: {}\n"
        "link_capacity_bps: {}\niface: \"{}\"\ncheckpoint_dir: \"{}\"\ntls_cert_path: \"{}\"\n"
        "tls_key_path: \"{}\"\ntls_ca_path: \"{}\"\nmtls_enabled: {}\ntelemetry_interval_s: 1.0\n"
        "heartbeat_interval_s: 5.0\nmigration_port: {}\ncontrol_poll_s: 0.1\nbpf_object_dir: \"{}/data_plane\"\n"
        "ebpf_enabled: false\ngpu_isolation_enabled: false\nseccomp_profile: \"{}/docker/seccomp.json\"\n",
        spec->id, spec->building, spec->switch_id, testbed.coordinator.address,
        port_of(host.node_daemon_address), host.node_daemon_address, spec->total_vram_bytes,
        spec->link_capacity_bps, host.network_interface.empty() ? "eth0" : host.network_interface,
        paths.checkpoint_dir, tls.cert, tls.key, tls.ca, tls.tls ? "true" : "false",
        host.migration_port, paths.reclaimnet, paths.reclaimnet);
  }
  auto topology = read_json_file(scenario.topology_file);
  if (!topology)
    return std::unexpected{topology.error()};
  std::set<std::string> mapped;
  for (const auto &host : testbed.nodes)
    mapped.insert(host.trace_node);
  const auto keep = [&](const std::string &id) {
    return mapped.contains(id) || !scenario.find_node(id);
  };
  Json links = Json::array();
  for (const auto &link : (*topology)["links"])
    if (keep(link["src"].get<std::string>()) && keep(link["dst"].get<std::string>()))
      links.push_back(link);
  (*topology)["links"] = std::move(links);
  if (auto written = write_json_file(out / "topology.json", *topology); !written)
    return written;
  {
    std::ofstream unit{out / "systemd" / "rnbt-node-daemon.service"};
    unit << std::format(
        "[Unit]\nDescription=ReclaimNet node daemon (backtest)\nAfter=network-online.target\n\n[Service]\n"
        "Environment=RNBT_NODE=%i\nEnvironment=RNBT_LOG_DIR={}\n"
        "UnsetEnvironment=RNBT_FAKE_GPU_DIR RNBT_FAKE_GPUS RNBT_FAKE_VRAM_BYTES\n"
        "ExecStart={}/bin/node_agent {}/etc/node-daemon.yaml\n"
        "KillMode=mixed\nTimeoutStopSec=15\n\n[Install]\nWantedBy=multi-user.target\n",
        paths.log_dir, paths.reclaimnet, paths.reclaimnet);
    std::ofstream coordinator{out / "systemd" / "rnbt-coordinator.service"};
    coordinator << std::format(
        "[Unit]\nDescription=ReclaimNet coordinator (backtest)\nAfter=network-online.target\n\n[Service]\n"
        "ExecStart={}/bin/coordinator {}/etc/coordinator.yaml\nTimeoutStopSec=15\n\n[Install]\n"
        "WantedBy=multi-user.target\n",
        paths.reclaimnet, paths.reclaimnet);
  }
  std::ofstream script{out / "install.sh"};
  script
      << "#!/usr/bin/env bash\nset -euo pipefail\nBIN=${BIN:?set BIN to the directory holding coordinator, node_agent, rnbt-job and rnbt-load}\n";
  script << std::format("ssh {0} sudo mkdir -p {1}/bin {1}/etc {2}\n", testbed.coordinator.ssh,
                        paths.reclaimnet, paths.coordinator_data);
  script << std::format(
      "scp \"$BIN/coordinator\" {0}:/tmp/ && ssh {0} sudo mv /tmp/coordinator {1}/bin/\n",
      testbed.coordinator.ssh, paths.reclaimnet);
  script << std::format(
      "scp coordinator.yaml topology.json systemd/rnbt-coordinator.service {0}:/tmp/ && ssh {0} "
      "'sudo mv /tmp/coordinator.yaml /tmp/topology.json {1}/etc/ && sudo mv "
      "/tmp/rnbt-coordinator.service /etc/systemd/system/ && sudo systemctl daemon-reload'\n",
      testbed.coordinator.ssh, paths.reclaimnet);
  for (const auto &host : testbed.nodes) {
    const auto vars = remote.node_vars(host);
    const auto job_binary = var(vars, "job_binary");
    const auto load_binary = var(vars, "load_binary");
    const auto workload_root = var(vars, "workload_root");
    const auto workload_logs = workload_root + "/logs";
    const auto base_path = [&](const std::string &path) {
      const auto suffix = "/runs/" + replay.config.run_id;
      return replay.config.workload_profiles && path.ends_with(suffix)
                 ? path.substr(0, path.size() - suffix.size())
                 : path;
    };
    const auto base_logs = base_path(var(vars, "log_dir"));
    const auto base_workloads = base_path(workload_root);
    const std::set<std::string> writable{
        var(vars, "state_root"), var(vars, "log_dir"), var(vars, "checkpoint_dir"),
        workload_root,           workload_logs,        base_logs,
        base_logs + "/runs",     base_workloads,       base_workloads + "/runs"};
    std::string owned_paths;
    for (const auto &path : writable)
      owned_paths += " " + shell_quote(path);
    const auto mkdir = std::format(
        "sudo mkdir -p {} {} {} {}{}", shell_quote(paths.reclaimnet + "/bin"),
        shell_quote(paths.reclaimnet + "/etc"),
        shell_quote(std::filesystem::path{job_binary}.parent_path().string()),
        shell_quote(std::filesystem::path{load_binary}.parent_path().string()), owned_paths);
    const auto install = std::format(
        "sudo mv /tmp/node_agent {0}/bin/ && sudo mv /tmp/rnbt-job {1} && sudo mv /tmp/rnbt-load {2} && "
        "sudo mv {3} {0}/etc/node-daemon.yaml && "
        "sudo chown \"$(id -u):$(id -g)\" {4} && "
        "sed {5} /tmp/rnbt-node-daemon.service | sudo tee "
        "/etc/systemd/system/rnbt-node-daemon.service >/dev/null && sudo systemctl daemon-reload",
        shell_quote(paths.reclaimnet), shell_quote(job_binary), shell_quote(load_binary),
        shell_quote("/tmp/" + host.trace_node + ".yaml"), owned_paths,
        shell_quote("s/%i/" + host.trace_node + "/"));
    script << std::format("ssh {} {}\n", shell_quote(host.ssh), shell_quote(mkdir));
    script << std::format("scp \"$BIN/node_agent\" \"$BIN/rnbt-job\" \"$BIN/rnbt-load\" {} "
                          "systemd/rnbt-node-daemon.service {}\n",
                          shell_quote("node_daemons/" + host.trace_node + ".yaml"),
                          shell_quote(host.ssh + ":/tmp/"));
    script << std::format("ssh {} {}\n", shell_quote(host.ssh), shell_quote(install));
  }
  std::filesystem::permissions(out / "install.sh", std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::add, ec);
  return {};
}

}
