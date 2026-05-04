#pragma once

#include "rnbt/job.hpp"
#include "rnbt/scenario.hpp"

#include <filesystem>
#include <map>

namespace rnbt {

struct MeasuredOutcome {
  std::string event_id{};
  std::string node{};
  ReclaimKind kind{};
  double notice_s{};
  std::int64_t trace_ns{};
  std::int64_t injected_ns{};
  std::int64_t day{};
  std::string instance{};
  std::string job{};
  std::string job_class{};
  std::uint32_t gpus{};
  std::optional<std::int64_t> stop_ns{};
  std::optional<std::int64_t> resume_ns{};
  double progress_before_s{};
  std::optional<double> restored_progress_s{};
  double rollback_s{};
  double downtime_s{};
  double recompute_gpu_h{};
  double downtime_gpu_h{};
  Outcome outcome{Outcome::unresolved};
  bool primary_success{};
  bool resolved{};
  std::string stop_event{};
  std::vector<std::string> destinations{};
  std::optional<bool> same_building{};
  bool injected_ok{};
};

template <> struct describe<MeasuredOutcome> {
  using M = MeasuredOutcome;
  using fields = std::tuple<
      Field<"event_id", &M::event_id>, Field<"node_id", &M::node>, Field<"reclaim_type", &M::kind>,
      Field<"notice_s", &M::notice_s>, Field<"trace_ns", &M::trace_ns>,
      Field<"injected_ns", &M::injected_ns>, Field<"day", &M::day>,
      Field<"instance_id", &M::instance>, Field<"job_id", &M::job>,
      Field<"job_class", &M::job_class>, Field<"gpus", &M::gpus>, Field<"stop_ns", &M::stop_ns>,
      Field<"resume_ns", &M::resume_ns>,
      Field<"logical_progress_seconds_before_reclaim", &M::progress_before_s>,
      Field<"restored_logical_progress_seconds", &M::restored_progress_s>,
      Field<"rollback_seconds", &M::rollback_s>, Field<"downtime_seconds", &M::downtime_s>,
      Field<"recomputation_loss_gpu_hours", &M::recompute_gpu_h>,
      Field<"downtime_loss_gpu_hours", &M::downtime_gpu_h>, Field<"outcome", &M::outcome>,
      Field<"primary_success", &M::primary_success>, Field<"recovery_resolved", &M::resolved>,
      Field<"stop_event", &M::stop_event>, Field<"actual_resume_node_ids", &M::destinations>,
      Field<"same_building", &M::same_building>, Field<"injected_ok", &M::injected_ok>>;
};

struct WindowSpec {
  std::string name{};
  double from_day{};
  double to_day{1e9};
};

template <> struct describe<WindowSpec> {
  using fields = std::tuple<Field<"name", &WindowSpec::name>,
                            Field<"from_day", &WindowSpec::from_day, Presence::optional>,
                            Field<"to_day", &WindowSpec::to_day, Presence::optional>>;
};

struct MetricsSpec {
  std::vector<WindowSpec> windows{{"replayed", -1e9, 1e9}};
  double handoff_tolerance_s = 3.0;
  std::size_t bootstrap_iterations = 1000;
  std::uint64_t seed = 1;
};

template <> struct describe<MetricsSpec> {
  using M = MetricsSpec;
  using fields =
      std::tuple<Field<"windows", &M::windows, Presence::optional>,
                 Field<"handoff_tolerance_s", &M::handoff_tolerance_s, Presence::optional>,
                 Field<"bootstrap_iterations", &M::bootstrap_iterations, Presence::optional>,
                 Field<"seed", &M::seed, Presence::optional>>;
};

struct RunMeasurement {
  std::filesystem::path directory{};
  std::vector<MeasuredOutcome> outcomes{};
  std::map<std::string, std::uint32_t, std::less<>> affected_per_event{};
  std::map<std::string, std::int64_t, std::less<>> event_day{};
  Json manifest{};
  Json workload_summary = Json::object();
};

[[nodiscard]] Expected<RunMeasurement> measure_run(const std::filesystem::path &run,
                                                   const MetricsSpec &spec);
[[nodiscard]] Json summarize(const RunMeasurement &run, const MetricsSpec &spec);
[[nodiscard]] Json compare(const RunMeasurement &candidate, const RunMeasurement &baseline,
                           const MetricsSpec &spec);
[[nodiscard]] std::string render_table(const Json &summary);

}
