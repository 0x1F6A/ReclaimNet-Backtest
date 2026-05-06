#include "rnbt/evaluate.hpp"

#include "rnbt/jsonl.hpp"
#include "rnbt/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <set>

namespace rnbt {

namespace fs = std::filesystem;

namespace {

struct Executed {
  std::string event_id;
  std::string node;
  ReclaimKind kind{};
  double notice_s{};
  std::int64_t trace_ns{};
  std::int64_t injected_ns{};
  bool ok{};
};

[[nodiscard]] bool is_start(std::string_view event) {
  return event == "start" || event == "resume-start";
}

[[nodiscard]] bool is_end(std::string_view event) {
  return event == "pause" || event == "stop" || event == "kill" || event == "park" ||
         event == "complete";
}

struct Timeline {
  std::vector<job::LogLine> lines;

  [[nodiscard]] std::optional<std::size_t> active_at(std::string_view node, std::int64_t t) const {
    std::optional<std::size_t> open;
    for (std::size_t i = 0; i < lines.size() && lines[i].t_ns <= t; ++i) {
      const auto &line = lines[i];
      if (is_start(line.event) || line.event == "continued")
        open = line.node == node ? std::optional<std::size_t>{i} : std::nullopt;
      else if (is_end(line.event))
        open.reset();
    }
    return open;
  }

  [[nodiscard]] std::optional<std::size_t> next_end(std::size_t from, std::int64_t t) const {
    for (std::size_t i = from; i < lines.size(); ++i)
      if (lines[i].t_ns >= t && is_end(lines[i].event))
        return i;
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::size_t> next_start(std::int64_t after) const {
    for (std::size_t i = 0; i < lines.size(); ++i)
      if (lines[i].t_ns > after && lines[i].event == "resume-start")
        return i;
    return std::nullopt;
  }
};

[[nodiscard]] std::string job_of(std::string_view instance) {
  const auto at = instance.find('@');
  return std::string{at == std::string_view::npos ? instance : instance.substr(0, at)};
}

[[nodiscard]] double workload_number(const Json &object, std::string_view field, double minimum,
                                     double maximum) {
  const auto &value = object.at(field);
  if (!value.is_number())
    throw DecodeError{std::format("{} must be numeric", field)};
  const auto number = value.get<double>();
  if (!std::isfinite(number) || number < minimum || number > maximum)
    throw DecodeError{
        std::format("{} must be finite and between {} and {}", field, minimum, maximum)};
  return number;
}

[[nodiscard]] Expected<Json> measure_workloads(const fs::path &run, const Json &manifest) {
  const auto profiles = manifest.find("workload_profiles");
  if (profiles == manifest.end() || profiles->is_null())
    return Json::object();
  try {
    if (!profiles->is_string() || profiles->get_ref<const std::string &>().empty())
      throw DecodeError{"workload_profiles must be a nonempty profile directory"};
    const auto &nodes = manifest.at("testbed").at("nodes");
    if (!nodes.is_array() || nodes.empty())
      throw DecodeError{"profiled workload replay requires manifest.testbed.nodes"};
    std::set<std::string> seen;
    std::vector<double> target_values, actual_values, absolute_errors;
    Json per_node = Json::array();
    for (const auto &host : nodes) {
      const auto node = host.at("trace_node").get<std::string>();
      if (node.empty() || node == "." || node == ".." ||
          node.find_first_of("/\\") != std::string::npos || !seen.insert(node).second)
        throw DecodeError{
            "manifest.testbed.nodes contains an unsafe, empty, or duplicate trace_node"};
      const auto directory = run / "collected" / "workloads" / node;
      const auto state = read_json_file(directory / "state.json");
      if (!state)
        return failure(std::format("workload {}: {}", node, state.error().message));
      const auto status = state->at("status").get<std::string>();
      if (status != "completed" && status != "stopped")
        return failure(std::format(
            "workload {} has status '{}'; collect a completed or stopped physical actor", node,
            status));
      if (!state->at("error").get<std::string>().empty())
        return failure(
            std::format("workload {} failed: {}", node, state->at("error").get<std::string>()));
      if (state->at("node").get<std::string>() != node)
        return failure(std::format("workload {} state belongs to a different node", node));
      const auto &written = state->at("samples_written");
      if (!written.is_number_integer() ||
          (written.is_number_integer() && !written.is_number_unsigned() &&
           written.get<std::int64_t>() < 0))
        throw DecodeError{
            std::format("workload {} samples_written must be a nonnegative integer", node)};
      const auto declared_samples = written.get<std::uint64_t>();
      if (status == "completed" && declared_samples == 0)
        return failure(std::format(
            "workload {} completed without real GPU measurements; the actor may have missed the replay window",
            node));
      const auto sample_path = directory / "logs" / "samples.jsonl";
      std::error_code ec;
      const bool has_log = fs::exists(sample_path, ec);
      if (ec || (!has_log && declared_samples != 0))
        return failure(std::format(
            "workload {} is missing readable collected sample logs despite samples_written={}",
            node, declared_samples));
      const auto initial_count = target_values.size();
      double last_offset = -1;
      if (has_log)
        for (const auto &sample : json_lines(sample_path)) {
          if (!sample.is_object())
            throw DecodeError{std::format("{} must contain JSON objects", sample_path.string())};
          const auto event = sample.at("event").get<std::string>();
          if (event == "failed" ||
              (sample.contains("error") && !sample.at("error").get<std::string>().empty()))
            throw DecodeError{
                std::format("{} contains a failed workload event", sample_path.string())};
          if (event != "sample")
            continue;
          if (sample.at("node").get<std::string>() != node)
            throw DecodeError{
                std::format("{} contains samples from a different node", sample_path.string())};
          const auto offset =
              workload_number(sample, "trace_offset_s", 0, std::numeric_limits<double>::max());
          if (offset < last_offset)
            throw DecodeError{
                std::format("{} trace_offset_s goes backwards", sample_path.string())};
          last_offset = offset;
          const auto &target = sample.at("target");
          const auto target_util = workload_number(target, "gpu_util_pct", 0, 100);
          (void)workload_number(target, "link_tx_bps", 0, std::numeric_limits<double>::max());
          (void)workload_number(target, "link_rx_bps", 0, std::numeric_limits<double>::max());
          const auto &devices = sample.at("actual_gpu_samples");
          if (!devices.is_array() || devices.empty())
            throw DecodeError{std::format("{} has no real NVML GPU samples", sample_path.string())};
          double actual_util = 0;
          std::set<std::string> device_ids;
          for (const auto &device : devices) {
            const auto uuid = device.at("uuid").get<std::string>();
            if (uuid.empty() || !device_ids.insert(uuid).second)
              throw DecodeError{
                  std::format("{} requires distinct physical GPU UUIDs", sample_path.string())};
            actual_util += workload_number(device, "utilization_gpu_percent", 0, 100) /
                           static_cast<double>(devices.size());
            if (device.contains("utilization_memory_percent"))
              (void)workload_number(device, "utilization_memory_percent", 0, 100);
          }
          const auto reported_util = workload_number(sample, "actual_gpu_util_pct", 0, 100);
          const auto reported_error = workload_number(sample, "gpu_util_error_pct", -100, 100);
          if (std::abs(reported_util - actual_util) > 1e-6 ||
              std::abs(reported_error - (actual_util - target_util)) > 1e-6)
            throw DecodeError{
                std::format("{} measured utilization or error disagrees with the NVML samples",
                            sample_path.string())};
          target_values.push_back(target_util);
          actual_values.push_back(actual_util);
          absolute_errors.push_back(std::abs(actual_util - target_util));
        }
      const auto observed_samples = target_values.size() - initial_count;
      if (observed_samples < declared_samples)
        return failure(std::format("workload {} has {} collected samples but state declares {}",
                                   node, observed_samples, declared_samples));
      per_node.push_back({{"trace_node", node},
                          {"status", status},
                          {"samples", observed_samples},
                          {"latest_actor_samples_written", declared_samples}});
    }
    const auto mean = [](const std::vector<double> &values) -> Json {
      if (values.empty())
        return nullptr;
      double result = 0;
      for (const auto value : values)
        result += value / static_cast<double>(values.size());
      return result;
    };
    const auto p95 = stats::quantile(absolute_errors, 0.95);
    return Json{{"samples", target_values.size()},
                {"nodes", std::move(per_node)},
                {"target_source", "trace_telemetry"},
                {"measurement_source", "nvml"},
                {"mean_target_gpu_util_pct", mean(target_values)},
                {"mean_actual_gpu_util_pct", mean(actual_values)},
                {"mean_absolute_gpu_util_error_pct", mean(absolute_errors)},
                {"p95_absolute_gpu_util_error_pct", p95 ? Json(*p95) : Json(nullptr)}};
  } catch (const std::exception &error) {
    return failure(std::format("workload replay measurements: {}", error.what()));
  }
}

}

Expected<RunMeasurement> measure_run(const fs::path &run, const MetricsSpec &spec) {
  RunMeasurement out{.directory = run};
  auto manifest = read_json_file(run / "run_manifest.json");
  if (!manifest)
    return std::unexpected{manifest.error()};
  bool physical_gpu_run = false;
  try {
    physical_gpu_run = manifest->value("gpu_backend", std::string{}) == "cuda";
    const auto status = manifest->value("status", std::string{"completed"});
    if (status != "completed" || manifest->value("failed_actions", std::uint64_t{0}) != 0 ||
        manifest->value("interrupted", false) || manifest->value("dry_run", false))
      return failure(
          std::format("{} is an incomplete, failed, or dry replay; evaluation requires a "
                      "completed physical GPU run",
                      run.string()));
  } catch (const Json::exception &error) {
    return failure(std::format("{}: invalid run manifest: {}", run.string(), error.what()));
  }
  out.manifest = *manifest;
  auto workloads = measure_workloads(run, *manifest);
  if (!workloads)
    return std::unexpected{workloads.error()};
  out.workload_summary = std::move(*workloads);
  const auto trace_dir = (*manifest)["trace"].get<std::string>();
  auto scenario = Scenario::load(ScenarioSource{.directory = trace_dir, .load_episodes = false});
  if (!scenario)
    return std::unexpected{scenario.error()};
  std::vector<Executed> executed;
  bool started_jobs = false;
  try {
    for (auto &record : json_lines(run / "driver.jsonl")) {
      if (record["type"] == "start" && record.value("ok", false))
        started_jobs = true;
      if (record["type"] != "reclaim")
        continue;
      const auto &action = record["action"];
      executed.push_back(
          Executed{.event_id = action["event_id"].get<std::string>(),
                   .node = action["node"].get<std::string>(),
                   .kind = decode<ReclaimKind>(action["kind"]),
                   .notice_s = action["notice_s"].get<double>(),
                   .trace_ns = action["trace_ns"].get<std::int64_t>(),
                   .injected_ns = record["details"].value(
                       "timestamp_ns", record["started_wall_ns"].get<std::int64_t>()),
                   .ok = record["ok"].get<bool>()});
    }
  } catch (const std::exception &error) {
    return failure(std::format("driver.jsonl: {}", error.what()));
  }
  std::map<std::string, Timeline> by_rank;
  std::int64_t last_seen = (*manifest).value("finished_wall_ns", std::int64_t{0});
  const auto jobs_root = run / "collected" / "jobs";
  std::error_code ec;
  if (fs::exists(jobs_root, ec))
    for (const auto &entry : fs::recursive_directory_iterator{jobs_root, ec})
      if (entry.is_regular_file() && entry.path().extension() == ".jsonl")
        try {
          for (auto &&line : records<job::LogLine>(entry.path())) {
            if (physical_gpu_run && (line.status == "failed" || line.event.starts_with("error:")))
              return failure(std::format("{}: GPU job {} rank {} failed ({}); this run cannot be "
                                         "evaluated as successful",
                                         entry.path().string(), line.instance, line.rank,
                                         line.event));
            if (physical_gpu_run && is_start(line.event) &&
                (!line.gpu_samples.is_array() || line.gpu_samples.empty()))
              return failure(std::format("{}: GPU job {} rank {} {} has no real GPU samples",
                                         entry.path().string(), line.instance, line.rank,
                                         line.event));
            last_seen = std::max(last_seen, line.t_ns);
            by_rank[std::format("{}#{}", line.instance, line.rank)].lines.push_back(
                std::move(line));
          }
        } catch (const std::exception &error) {
          return failure(std::format("{}: {}", entry.path().string(), error.what()));
        }
  if (physical_gpu_run && started_jobs && by_rank.empty())
    return failure(
        std::format("{} has no collected GPU job logs despite starting jobs; collect the "
                    "physical run before evaluating it",
                    run.string()));
  std::map<std::string, std::vector<std::string>> instances;
  for (auto &[key, timeline] : by_rank) {
    std::ranges::stable_sort(timeline.lines, std::less<>{}, &job::LogLine::t_ns);
    std::erase_if(timeline.lines,
                  [](const job::LogLine &line) { return line.event.starts_with("gpu:"); });
    instances[key.substr(0, key.rfind('#'))].push_back(key);
  }
  const double tolerance = spec.handoff_tolerance_s;
  for (const auto &event : executed) {
    const auto day = scenario->calendar.day_index(instant_of(event.trace_ns));
    out.event_day.insert_or_assign(event.event_id, day);
    std::uint32_t affected = 0;
    for (const auto &[instance, ranks] : instances) {
      std::optional<std::string> hit;
      for (const auto &key : ranks)
        if (by_rank.at(key).active_at(event.node, event.injected_ns))
          hit = key;
      if (!hit)
        continue;
      ++affected;
      const auto &rank_line = by_rank.at(*hit);
      const auto job_id = job_of(instance);
      const auto index = scenario->job_index(job_id);
      MeasuredOutcome outcome{.event_id = event.event_id,
                              .node = event.node,
                              .kind = event.kind,
                              .notice_s = event.notice_s,
                              .trace_ns = event.trace_ns,
                              .injected_ns = event.injected_ns,
                              .day = day,
                              .instance = instance,
                              .job = job_id,
                              .job_class = index ? scenario->jobs[*index].job_class : std::string{},
                              .gpus = index ? scenario->jobs[*index].gpus() : 1,
                              .injected_ok = event.ok};
      const auto start_index = rank_line.active_at(event.node, event.injected_ns).value_or(0);
      const auto end = rank_line.next_end(start_index, event.injected_ns);
      if (!end) {
        outcome.outcome = Outcome::unresolved;
        outcome.downtime_s = 0.0;
        out.outcomes.push_back(std::move(outcome));
        continue;
      }
      const auto &stop = rank_line.lines[*end];
      outcome.stop_ns = stop.t_ns;
      outcome.stop_event = stop.event;
      outcome.progress_before_s = stop.progress_s;
      if (stop.event == "complete") {
        outcome.outcome = Outcome::not_running;
        out.outcomes.push_back(std::move(outcome));
        continue;
      }
      const auto resumed = rank_line.next_start(stop.t_ns);
      if (!resumed) {
        outcome.outcome = Outcome::unresolved;
        outcome.downtime_s = std::max(0.0, static_cast<double>(last_seen - stop.t_ns) / 1e9);
        outcome.downtime_gpu_h = outcome.downtime_s * outcome.gpus / 3600.0;
        out.outcomes.push_back(std::move(outcome));
        continue;
      }
      const auto &resume = rank_line.lines[*resumed];
      outcome.resolved = true;
      outcome.resume_ns = resume.t_ns;
      outcome.restored_progress_s = resume.progress_s;
      outcome.rollback_s = std::max(0.0, outcome.progress_before_s - resume.progress_s);
      outcome.downtime_s = std::max(0.0, static_cast<double>(resume.t_ns - stop.t_ns) / 1e9);
      for (const auto &key : ranks)
        if (const auto start = by_rank.at(key).next_start(stop.t_ns))
          outcome.destinations.push_back(by_rank.at(key).lines[*start].node);
      if (const auto *source = scenario->find_node(event.node);
          source && !outcome.destinations.empty())
        outcome.same_building =
            std::ranges::all_of(outcome.destinations, [&](const std::string &node) {
              const auto *destination = scenario->find_node(node);
              return destination && destination->building == source->building;
            });
      if (event.kind == ReclaimKind::scheduled)
        outcome.outcome =
            outcome.rollback_s <= tolerance ? Outcome::handoff : Outcome::degraded_recovery;
      else
        outcome.outcome = Outcome::checkpoint_recovery;
      if (outcome.restored_progress_s && *outcome.restored_progress_s <= tolerance &&
          outcome.progress_before_s > tolerance)
        outcome.outcome = Outcome::restart_from_input;
      outcome.primary_success = event.kind == ReclaimKind::scheduled
                                    ? outcome.outcome == Outcome::handoff
                                    : outcome.outcome == Outcome::checkpoint_recovery;
      outcome.recompute_gpu_h = outcome.rollback_s * outcome.gpus / 3600.0;
      outcome.downtime_gpu_h = outcome.downtime_s * outcome.gpus / 3600.0;
      out.outcomes.push_back(std::move(outcome));
    }
    out.affected_per_event.insert_or_assign(event.event_id, affected);
  }
  return out;
}

Json summarize(const RunMeasurement &run, const MetricsSpec &spec) {
  Json windows = Json::array();
  for (const auto &window : spec.windows) {
    const auto in = [&](std::int64_t day) {
      return static_cast<double>(day) >= window.from_day &&
             static_cast<double>(day) < window.to_day;
    };
    auto rows =
        run.outcomes | std::views::filter([&](const MeasuredOutcome &o) { return in(o.day); });
    std::size_t events = 0, idle = 0, scheduled = 0, emergency = 0;
    for (const auto &[event, affected] : run.affected_per_event)
      if (const auto day = run.event_day.find(event);
          day != run.event_day.end() && in(day->second)) {
        ++events;
        idle += affected == 0;
      }
    double recompute = 0, downtime = 0, recompute_s = 0, downtime_s = 0, recompute_e = 0,
           downtime_e = 0;
    std::size_t pairs = 0, resolved = 0, primary = 0, primary_s = 0, primary_e = 0, handoff = 0,
                same = 0, unresolved = 0;
    std::vector<double> downtimes, downtimes_s, downtimes_e;
    for (const auto &o : rows) {
      if (o.outcome == Outcome::not_running)
        continue;
      ++pairs;
      const bool is_scheduled = o.kind == ReclaimKind::scheduled;
      (is_scheduled ? scheduled : emergency) += 1;
      recompute += o.recompute_gpu_h;
      downtime += o.downtime_gpu_h;
      (is_scheduled ? recompute_s : recompute_e) += o.recompute_gpu_h;
      (is_scheduled ? downtime_s : downtime_e) += o.downtime_gpu_h;
      primary += o.primary_success;
      (is_scheduled ? primary_s : primary_e) += o.primary_success;
      handoff += o.outcome == Outcome::handoff;
      unresolved += !o.resolved;
      if (o.resolved) {
        ++resolved;
        downtimes.push_back(o.downtime_s);
        (is_scheduled ? downtimes_s : downtimes_e).push_back(o.downtime_s);
        same += o.same_building.value_or(false);
      }
    }
    const auto pct = [](std::size_t a, std::size_t b) {
      return b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0;
    };
    const auto opt = [](std::optional<double> v) { return v ? Json(*v) : Json(nullptr); };
    windows.push_back({{"window", window.name},
                       {"events", events},
                       {"events_without_running_job", idle},
                       {"affected_pairs", pairs},
                       {"scheduled_pairs", scheduled},
                       {"emergency_pairs", emergency},
                       {"work_loss_gpu_h", recompute + downtime},
                       {"recompute_gpu_h", recompute},
                       {"downtime_gpu_h", downtime},
                       {"scheduled_loss_gpu_h", recompute_s + downtime_s},
                       {"emergency_loss_gpu_h", recompute_e + downtime_e},
                       {"median_downtime_s", opt(stats::median(downtimes))},
                       {"p99_downtime_s", opt(stats::quantile(downtimes, 0.99))},
                       {"scheduled_median_downtime_s", opt(stats::median(downtimes_s))},
                       {"emergency_median_downtime_s", opt(stats::median(downtimes_e))},
                       {"primary_success_pct", pct(primary, pairs)},
                       {"scheduled_zero_loss_handoff_pct", pct(primary_s, scheduled)},
                       {"emergency_checkpoint_recovery_pct", pct(primary_e, emergency)},
                       {"resolved_pct", pct(resolved, pairs)},
                       {"unresolved_pairs", unresolved},
                       {"same_building_pct", pct(same, resolved)},
                       {"handoffs", handoff}});
  }
  Json summary{{"run", run.directory.string()}, {"windows", std::move(windows)}};
  if (!run.workload_summary.empty())
    summary["workload_replay"] = run.workload_summary;
  return summary;
}

Json compare(const RunMeasurement &candidate, const RunMeasurement &baseline,
             const MetricsSpec &spec) {
  const auto per_event = [](const RunMeasurement &run) {
    std::map<std::string, double, std::less<>> loss;
    for (const auto &[event, affected] : run.affected_per_event)
      loss[event] = 0.0;
    for (const auto &o : run.outcomes)
      loss[o.event_id] += o.recompute_gpu_h + o.downtime_gpu_h;
    return loss;
  };
  const auto a = per_event(candidate), b = per_event(baseline);
  std::vector<double> differences;
  std::map<std::int64_t, std::vector<std::pair<double, double>>> blocks;
  for (const auto &[event, loss] : a)
    if (const auto other = b.find(event); other != b.end()) {
      differences.push_back(loss - other->second);
      const auto day = candidate.event_day.contains(event) ? candidate.event_day.at(event) : 0;
      blocks[day].emplace_back(loss, other->second);
    }
  const auto test = stats::wilcoxon_signed_rank(differences);
  std::vector<std::vector<std::pair<double, double>>> grouped;
  for (auto &[day, rows] : blocks)
    grouped.push_back(std::move(rows));
  const auto ratio = stats::bootstrap_ratio(grouped, spec.bootstrap_iterations, spec.seed);
  return {{"paired_events", differences.size()},
          {"candidate", candidate.directory.string()},
          {"baseline", baseline.directory.string()},
          {"wilcoxon",
           {{"n_nonzero", test.n},
            {"w_plus", test.w_plus},
            {"z", test.z},
            {"p_two_sided", test.p_two_sided}}},
          {"loss_ratio_candidate_over_baseline",
           {{"estimate", ratio.estimate}, {"ci95_low", ratio.low}, {"ci95_high", ratio.high}}}};
}

std::string render_table(const Json &summary) {
  std::string out;
  for (const auto &window : summary["windows"]) {
    const auto number = [&](std::string_view key) {
      const auto &value = window[std::string{key}];
      return value.is_null() ? std::string{"-"} : std::format("{:.2f}", value.get<double>());
    };
    out += std::format("[{}] events={} affected={} idle={}\n", window["window"].get<std::string>(),
                       window["events"].get<std::size_t>(),
                       window["affected_pairs"].get<std::size_t>(),
                       window["events_without_running_job"].get<std::size_t>());
    out += std::format(
        "  work loss {} GPU-h (recompute {} + downtime {}); scheduled {} / emergency {}\n",
        number("work_loss_gpu_h"), number("recompute_gpu_h"), number("downtime_gpu_h"),
        number("scheduled_loss_gpu_h"), number("emergency_loss_gpu_h"));
    out += std::format(
        "  downtime median {} s, p99 {} s; success {}% (handoff {}%, emergency {}%); unresolved {}\n",
        number("median_downtime_s"), number("p99_downtime_s"), number("primary_success_pct"),
        number("scheduled_zero_loss_handoff_pct"), number("emergency_checkpoint_recovery_pct"),
        window["unresolved_pairs"].get<std::size_t>());
  }
  return out;
}

}
