#pragma once

#include "rnbt/reflect.hpp"

#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace rnbt {

struct WindowConfig {
  std::optional<std::string> from{};
  std::optional<std::string> to{};
};

template <> struct describe<WindowConfig> {
  using fields = std::tuple<Field<"from", &WindowConfig::from>, Field<"to", &WindowConfig::to>>;
};

struct EventFilter {
  std::vector<std::string> ids{};
  std::vector<ReclaimKind> kinds{};

  [[nodiscard]] bool admits(std::string_view id, ReclaimKind kind) const noexcept {
    return (ids.empty() || std::ranges::contains(ids, id)) &&
           (kinds.empty() || std::ranges::contains(kinds, kind));
  }
};

template <> struct describe<EventFilter> {
  using fields = std::tuple<Field<"ids", &EventFilter::ids, Presence::optional>,
                            Field<"kinds", &EventFilter::kinds, Presence::optional>>;
};

struct TimingConfig {
  double start_delay_s = 30.0;
  double node_return_s = 3600.0;
  double resolution_cap_s = 7200.0;
  double episode_post_resume_s = 60.0;
  double drain_s = 600.0;
  double jobs_warmup_s = 0.0;
  bool compress_idle = false;
  double idle_threshold_s = 900.0;
  double reset_budget_s = 180.0;
  double health_timeout_s = 120.0;
  double registration_grace_s = 10.0;
};

template <> struct describe<TimingConfig> {
  using T = TimingConfig;
  using fields =
      std::tuple<Field<"start_delay_s", &T::start_delay_s, Presence::optional>,
                 Field<"node_return_s", &T::node_return_s, Presence::optional>,
                 Field<"resolution_cap_s", &T::resolution_cap_s, Presence::optional>,
                 Field<"episode_post_resume_s", &T::episode_post_resume_s, Presence::optional>,
                 Field<"drain_s", &T::drain_s, Presence::optional>,
                 Field<"jobs_warmup_s", &T::jobs_warmup_s, Presence::optional>,
                 Field<"compress_idle", &T::compress_idle, Presence::optional>,
                 Field<"idle_threshold_s", &T::idle_threshold_s, Presence::optional>,
                 Field<"reset_budget_s", &T::reset_budget_s, Presence::optional>,
                 Field<"health_timeout_s", &T::health_timeout_s, Presence::optional>,
                 Field<"registration_grace_s", &T::registration_grace_s, Presence::optional>>;
};

struct ReplayConfig {
  std::string run_id = "replay";
  std::string trace{};
  std::optional<std::string> workload_profiles{};
  bool replay_link_capacity = false;
  std::optional<std::string> migration_priors{};
  std::string testbed{};
  ReplayMode mode = ReplayMode::episodes;
  WindowConfig window{};
  EventFilter events{};
  TimingConfig timing{};
  bool reset_between_periods = true;
  bool warm_hazard = true;
  std::string output = "runs/replay";
};

template <> struct describe<ReplayConfig> {
  using R = ReplayConfig;
  using fields =
      std::tuple<Field<"run_id", &R::run_id, Presence::optional>, Field<"trace", &R::trace>,
                 Field<"workload_profiles", &R::workload_profiles>,
                 Field<"replay_link_capacity", &R::replay_link_capacity, Presence::optional>,
                 Field<"migration_priors", &R::migration_priors>, Field<"testbed", &R::testbed>,
                 Field<"mode", &R::mode, Presence::optional>,
                 Field<"window", &R::window, Presence::optional>,
                 Field<"events", &R::events, Presence::optional>,
                 Field<"timing", &R::timing, Presence::optional>,
                 Field<"reset_between_periods", &R::reset_between_periods, Presence::optional>,
                 Field<"warm_hazard", &R::warm_hazard, Presence::optional>,
                 Field<"output", &R::output, Presence::optional>>;
};

struct HostConfig {
  std::string trace_node{};
  std::string host{};
  std::string ssh{};
  std::string node_daemon_address{};
  std::string network_interface{};
  std::uint16_t migration_port = 7600;
  std::map<std::string, std::string> vars{};
};

template <> struct describe<HostConfig> {
  using H = HostConfig;
  using fields = std::tuple<
      Field<"trace_node", &H::trace_node>, Field<"host", &H::host>,
      Field<"ssh", &H::ssh, Presence::optional>,
      Field<"node_daemon_address", &H::node_daemon_address, Presence::required, "agent_address">,
      Field<"network_interface", &H::network_interface, Presence::optional>,
      Field<"migration_port", &H::migration_port, Presence::optional>,
      Field<"vars", &H::vars, Presence::optional>>;
};

struct CoordinatorHost {
  std::string host{};
  std::string ssh{};
  std::string address{};
  std::map<std::string, std::string> vars{};
};

template <> struct describe<CoordinatorHost> {
  using C = CoordinatorHost;
  using fields =
      std::tuple<Field<"host", &C::host>, Field<"ssh", &C::ssh, Presence::optional>,
                 Field<"address", &C::address>, Field<"vars", &C::vars, Presence::optional>>;
};

struct TestbedPaths {
  std::string reclaimnet = "/opt/reclaimnet";
  std::string job_binary = "/opt/rnbt/bin/rnbt-job";
  std::string load_binary = "/opt/rnbt/bin/rnbt-load";
  std::string workload_root = "/var/lib/rnbt/workload";
  std::string state_root = "/var/lib/rnbt/state";
  std::string log_dir = "/var/lib/rnbt/logs";
  std::string checkpoint_dir = "/var/lib/rnbt/checkpoints";
  std::string coordinator_data = "/var/lib/rnbt/coordinator";
};

template <> struct describe<TestbedPaths> {
  using P = TestbedPaths;
  using fields = std::tuple<Field<"reclaimnet", &P::reclaimnet, Presence::optional>,
                            Field<"job_binary", &P::job_binary, Presence::optional>,
                            Field<"load_binary", &P::load_binary, Presence::optional>,
                            Field<"workload_root", &P::workload_root, Presence::optional>,
                            Field<"state_root", &P::state_root, Presence::optional>,
                            Field<"log_dir", &P::log_dir, Presence::optional>,
                            Field<"checkpoint_dir", &P::checkpoint_dir, Presence::optional>,
                            Field<"coordinator_data", &P::coordinator_data, Presence::optional>>;
};

struct TestbedCommands {
  std::string remote = "ssh -o BatchMode=yes -o ConnectTimeout=10 {ssh} {quoted}";
  std::string fetch = "scp -q -r {ssh}:{remote_path} {local_path}";
  std::string node_daemon_start = "sudo systemctl start rnbt-node-daemon";
  std::string node_daemon_stop = "sudo systemctl stop rnbt-node-daemon";
  std::string workload_start =
      "{env} {load_binary} start --profile {profile_path} --state {workload_state} "
      "--log {workload_log} --gpus {gpu_count} --start-wall-ns {workload_origin_ns}";
  std::string workload_stop = "{env} {load_binary} stop --state {workload_state}";
  std::string link_set_rate =
      "sudo -n tc qdisc replace dev {network_interface} root handle 7ffe: tbf "
      "rate {link_bps}bit burst 65536 latency 100ms";
  std::string link_block =
      "sudo -n tc qdisc replace dev {network_interface} root handle 7ffe: netem loss 100%";
  std::string link_clear = "q=$(sudo -n tc qdisc show dev {network_interface}) || exit $?; "
                           "if printf '%s\\n' \"$q\" | grep -q ' 7ffe: root'; then "
                           "sudo -n tc qdisc del dev {network_interface} root handle 7ffe:; fi";
  std::string coordinator_start = "sudo systemctl start rnbt-coordinator";
  std::string coordinator_stop = "sudo systemctl stop rnbt-coordinator";
  std::string coordinator_configure =
      "sudo sed -i 's|^trace_dir:.*|trace_dir: {coordinator_data}/trace/{period}|' "
      "{reclaimnet}/etc/coordinator.yaml";
  std::string node_wipe = "sudo rm -rf {checkpoint_dir}/* {state_root}/*";
  std::string coordinator_wipe = "sudo rm -f {coordinator_data}/reclaimnet.db*";
  std::string job_launch =
      "{env} {job_binary} start --state {state_path} --instance {instance} --rank {rank} "
      "--node {node} --log-dir {log_dir} --target-s {target_s} --progress-s {progress_s} "
      "--checkpoint-bytes {checkpoint_bytes} --vram-bytes {vram_bytes} --gpus {gpus}{park}";
  std::string job_kill_all =
      "{env} {job_binary} kill-all --root {state_root} --root {checkpoint_dir}";
  std::string job_stop_instance =
      "{env} {job_binary} stop-instance --instance {instance} --root {state_root} --root "
      "{checkpoint_dir}";
};

template <> struct describe<TestbedCommands> {
  using C = TestbedCommands;
  using fields = std::tuple<
      Field<"remote", &C::remote, Presence::optional>,
      Field<"fetch", &C::fetch, Presence::optional>,
      Field<"node_daemon_start", &C::node_daemon_start, Presence::optional, "agent_start">,
      Field<"node_daemon_stop", &C::node_daemon_stop, Presence::optional, "agent_stop">,
      Field<"workload_start", &C::workload_start, Presence::optional>,
      Field<"workload_stop", &C::workload_stop, Presence::optional>,
      Field<"link_set_rate", &C::link_set_rate, Presence::optional>,
      Field<"link_block", &C::link_block, Presence::optional>,
      Field<"link_clear", &C::link_clear, Presence::optional>,
      Field<"coordinator_start", &C::coordinator_start, Presence::optional>,
      Field<"coordinator_stop", &C::coordinator_stop, Presence::optional>,
      Field<"coordinator_configure", &C::coordinator_configure, Presence::optional>,
      Field<"node_wipe", &C::node_wipe, Presence::optional>,
      Field<"coordinator_wipe", &C::coordinator_wipe, Presence::optional>,
      Field<"job_launch", &C::job_launch, Presence::optional>,
      Field<"job_kill_all", &C::job_kill_all, Presence::optional>,
      Field<"job_stop_instance", &C::job_stop_instance, Presence::optional>>;
};

struct CredentialConfig {
  bool tls = false;
  std::string ca{};
  std::string cert{};
  std::string key{};
  std::string jwt_private_key{};
  std::string target_name_override{};
};

template <> struct describe<CredentialConfig> {
  using C = CredentialConfig;
  using fields =
      std::tuple<Field<"tls", &C::tls, Presence::optional>, Field<"ca", &C::ca, Presence::optional>,
                 Field<"cert", &C::cert, Presence::optional>,
                 Field<"key", &C::key, Presence::optional>,
                 Field<"jwt_private_key", &C::jwt_private_key, Presence::optional>,
                 Field<"target_name_override", &C::target_name_override, Presence::optional>>;
};

struct Testbed {
  CoordinatorHost coordinator{};
  std::vector<HostConfig> nodes{};
  TestbedPaths paths{};
  TestbedCommands commands{};
  CredentialConfig credentials{};
  bool local = false;
  std::size_t parallelism = 16;
  double rpc_timeout_s = 15.0;
  double command_timeout_s = 120.0;

  [[nodiscard]] const HostConfig *node(std::string_view trace_node) const noexcept {
    const auto found = std::ranges::find(nodes, trace_node, &HostConfig::trace_node);
    return found == nodes.end() ? nullptr : std::addressof(*found);
  }
};

template <> struct describe<Testbed> {
  using T = Testbed;
  using fields = std::tuple<Field<"coordinator", &T::coordinator>, Field<"nodes", &T::nodes>,
                            Field<"paths", &T::paths, Presence::optional>,
                            Field<"commands", &T::commands, Presence::optional>,
                            Field<"credentials", &T::credentials, Presence::optional>,
                            Field<"local", &T::local, Presence::optional>,
                            Field<"parallelism", &T::parallelism, Presence::optional>,
                            Field<"rpc_timeout_s", &T::rpc_timeout_s, Presence::optional>,
                            Field<"command_timeout_s", &T::command_timeout_s, Presence::optional>>;
};

struct LoadedReplay {
  ReplayConfig config{};
  Testbed testbed{};
  Json raw_config{};
  Json raw_testbed{};
  std::filesystem::path config_directory{};
};

[[nodiscard]] Expected<LoadedReplay> load_replay(const std::filesystem::path &config,
                                                 std::span<const std::string> overrides);

}
