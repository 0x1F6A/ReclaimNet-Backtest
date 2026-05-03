#pragma once

#include "rnbt/jsonl.hpp"
#include "rnbt/plan.hpp"
#include "rnbt/rpc.hpp"
#include "rnbt/shell.hpp"

#include <filesystem>
#include <set>
#include <stop_token>

namespace rnbt {

struct ActionRecord {
  std::uint64_t index{};
  std::string type{};
  Json action{};
  std::int64_t planned_wall_ns{};
  std::int64_t started_wall_ns{};
  std::int64_t finished_wall_ns{};
  double lateness_s{};
  bool ok = true;
  std::vector<std::string> errors{};
  Json details = Json::object();
};

template <> struct describe<ActionRecord> {
  using A = ActionRecord;
  using fields = std::tuple<
      Field<"index", &A::index>, Field<"type", &A::type>, Field<"action", &A::action>,
      Field<"planned_wall_ns", &A::planned_wall_ns>, Field<"started_wall_ns", &A::started_wall_ns>,
      Field<"finished_wall_ns", &A::finished_wall_ns>, Field<"lateness_s", &A::lateness_s>,
      Field<"ok", &A::ok>, Field<"errors", &A::errors>, Field<"details", &A::details>>;
};

struct RunReport {
  std::filesystem::path directory{};
  std::uint64_t actions{};
  std::uint64_t failed{};
  double max_lateness_s{};
  std::int64_t wall_origin_ns{};
};

class Driver {
public:
  Driver(const LoadedReplay &replay, const Scenario &scenario, const Plan &plan,
         std::filesystem::path directory, bool dry_run);

  [[nodiscard]] Expected<RunReport> run(std::stop_token stop);

  Status testbed_reset(std::uint32_t period, Instant warm_until, ActionRecord &record,
                       std::stop_token stop);
  Status testbed_stop(ActionRecord &record);

private:
  [[nodiscard]] std::int64_t wall_of(Instant trace_at) const noexcept;
  [[nodiscard]] std::vector<const HostConfig *> hosts() const;
  void execute(const Action &action, ActionRecord &record, std::stop_token stop);
  void on(const ResetTestbed &reset, const Action &action, ActionRecord &record,
          std::stop_token stop);
  void on(const Depart &depart, const Action &action, ActionRecord &record, std::stop_token stop);
  void on(const Return &back, const Action &action, ActionRecord &record, std::stop_token stop);
  void on(const StartInstance &start, const Action &action, ActionRecord &record,
          std::stop_token stop);
  void on(const InjectReclaim &reclaim, const Action &action, ActionRecord &record,
          std::stop_token stop);
  void on(const StopInstance &stop_instance, const Action &action, ActionRecord &record,
          std::stop_token stop);
  void on(const SetLinkCapacity &capacity, const Action &action, ActionRecord &record,
          std::stop_token stop);
  void on(const EndReplay &end, const Action &action, ActionRecord &record, std::stop_token stop);
  void run_commands(std::span<const Expected<Command>> commands, std::string_view step,
                    ActionRecord &record);
  Status wait_healthy(std::span<const HostConfig *const> nodes, std::stop_token stop);
  Status validate_workloads(std::span<const HostConfig *const> nodes) const;
  Status start_workloads(std::span<const HostConfig *const> nodes, ActionRecord &record);
  Status check_workloads(std::span<const HostConfig *const> nodes, ActionRecord &record,
                         bool force = false);
  Status stop_workloads(std::span<const HostConfig *const> nodes, ActionRecord &record,
                        bool check_health = false);
  bool sleep_until(std::int64_t wall_ns, std::stop_token stop) const;

  const LoadedReplay &replay_;
  const Scenario &scenario_;
  const Plan &plan_;
  std::filesystem::path directory_{};
  bool dry_run_{};
  Remote remote_;
  std::unique_ptr<Channels> channels_{};
  JsonlSink log_{};
  std::int64_t wall_origin_ns_{};
  std::set<std::string> offline_{};
  std::set<std::string> active_instances_{};
  std::set<std::string> running_workloads_{};
  std::chrono::steady_clock::time_point next_workload_health_check_{};
};

[[nodiscard]] Status collect(const LoadedReplay &replay,
                             const std::filesystem::path &run_directory);
[[nodiscard]] Status render_deployment(const LoadedReplay &replay, const Scenario &scenario,
                                       const std::filesystem::path &out);

}
