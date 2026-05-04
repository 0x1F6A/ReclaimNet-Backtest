#pragma once

#include "rnbt/calendar.hpp"
#include "rnbt/jsonl.hpp"

#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace rnbt {

struct NodeSpec {
  std::string id{};
  std::string address{};
  std::string building{};
  std::string switch_id{};
  std::uint32_t gpu_count{1};
  std::uint64_t total_vram_bytes{};
  std::uint64_t link_capacity_bps{};
  double cuda_capability{};
  Instant registered_at{};
};

template <> struct describe<NodeSpec> {
  using fields =
      std::tuple<Field<"node_id", &NodeSpec::id>, Field<"address", &NodeSpec::address>,
                 Field<"building", &NodeSpec::building>, Field<"switch_id", &NodeSpec::switch_id>,
                 Field<"gpu_count", &NodeSpec::gpu_count>,
                 Field<"total_vram_bytes", &NodeSpec::total_vram_bytes>,
                 Field<"link_capacity_bps", &NodeSpec::link_capacity_bps>,
                 Field<"cuda_capability", &NodeSpec::cuda_capability>,
                 Field<"registered_at_ns", &NodeSpec::registered_at>>;
};

struct BusyInterval {
  std::string segment_id{};
  Instant start{};
  Instant end{};
  std::vector<std::string> nodes{};
};

template <> struct describe<BusyInterval> {
  using fields =
      std::tuple<Field<"segment_id", &BusyInterval::segment_id, Presence::optional>,
                 Field<"start_ns", &BusyInterval::start>, Field<"end_ns", &BusyInterval::end>,
                 Field<"node_ids", &BusyInterval::nodes>>;
};

struct JobSpec {
  std::string id{};
  Instant submit{};
  Nanos requested_runtime{};
  std::uint32_t gang_size{1};
  std::uint32_t per_node_gpus{1};
  std::uint64_t logical_vram_bytes{};
  std::uint64_t per_rank_checkpoint_bytes{};
  std::uint64_t logical_checkpoint_bytes{};
  double min_cuda{};
  double loss_budget_s{};
  double lambda_i{};
  std::vector<std::string> initial_nodes{};
  std::string job_class{};
  std::string workload_tag{};
  std::vector<BusyInterval> busy_intervals{};

  [[nodiscard]] constexpr std::uint32_t gpus() const noexcept { return gang_size * per_node_gpus; }
  [[nodiscard]] constexpr std::uint64_t per_rank_vram() const noexcept {
    return gang_size ? logical_vram_bytes / gang_size : logical_vram_bytes;
  }
};

template <> struct describe<JobSpec> {
  using fields = std::tuple<
      Field<"job_id", &JobSpec::id>, Field<"submit_time_ns", &JobSpec::submit>,
      Field<"requested_runtime_s", &JobSpec::requested_runtime>,
      Field<"gang_size", &JobSpec::gang_size>, Field<"per_node_gpu_count", &JobSpec::per_node_gpus>,
      Field<"logical_vram_bytes", &JobSpec::logical_vram_bytes>,
      Field<"per_rank_checkpoint_size_bytes", &JobSpec::per_rank_checkpoint_bytes>,
      Field<"logical_checkpoint_size_bytes", &JobSpec::logical_checkpoint_bytes>,
      Field<"min_cuda_capability", &JobSpec::min_cuda>,
      Field<"loss_budget_s", &JobSpec::loss_budget_s>, Field<"lambda_i", &JobSpec::lambda_i>,
      Field<"initial_node_ids", &JobSpec::initial_nodes>, Field<"job_class", &JobSpec::job_class>,
      Field<"workload_tag", &JobSpec::workload_tag, Presence::optional>,
      Field<"busy_intervals", &JobSpec::busy_intervals, Presence::optional>>;
};

struct ReclaimSpec {
  std::string event_id{};
  std::string node{};
  ReclaimKind kind{};
  Nanos notice{};
  Instant at{};

  [[nodiscard]] constexpr Instant deadline() const noexcept { return saturating_add(at, notice); }
};

template <> struct describe<ReclaimSpec> {
  using fields =
      std::tuple<Field<"event_id", &ReclaimSpec::event_id, Presence::optional>,
                 Field<"node_id", &ReclaimSpec::node>, Field<"type", &ReclaimSpec::kind>,
                 Field<"tau_seconds", &ReclaimSpec::notice>, Field<"timestamp", &ReclaimSpec::at>>;
};

struct EpisodeSpec {
  std::string event_id{};
  std::string job{};
  std::vector<std::string> nodes{};
  Instant start{};
  Instant end{};
  std::string reason{};
  std::string segment_id{};
};

template <> struct describe<EpisodeSpec> {
  using fields =
      std::tuple<Field<"event_id", &EpisodeSpec::event_id>, Field<"job_id", &EpisodeSpec::job>,
                 Field<"node_ids", &EpisodeSpec::nodes>, Field<"start_ns", &EpisodeSpec::start>,
                 Field<"end_ns", &EpisodeSpec::end>, Field<"reason", &EpisodeSpec::reason>,
                 Field<"segment_id", &EpisodeSpec::segment_id, Presence::optional>>;
};

struct CapacitySample {
  std::string node{};
  double bps{};
  Instant at{};
};

template <> struct describe<CapacitySample> {
  using fields =
      std::tuple<Field<"node_id", &CapacitySample::node>, Field<"bps", &CapacitySample::bps>,
                 Field<"timestamp", &CapacitySample::at>>;
};

struct TelemetrySample {
  std::string node{};
  double gpu_util_pct{};
  double vram_used_pct{};
  std::uint64_t vram_free_bytes{};
  double link_tx_bps{};
  double link_rx_bps{};
  double cpu_load_1m{};
  Instant at{};
};

template <> struct describe<TelemetrySample> {
  using fields = std::tuple<Field<"node_id", &TelemetrySample::node>,
                            Field<"gpu_util_pct", &TelemetrySample::gpu_util_pct>,
                            Field<"vram_used_pct", &TelemetrySample::vram_used_pct>,
                            Field<"vram_free_bytes", &TelemetrySample::vram_free_bytes>,
                            Field<"link_tx_bps", &TelemetrySample::link_tx_bps, Presence::optional>,
                            Field<"link_rx_bps", &TelemetrySample::link_rx_bps, Presence::optional>,
                            Field<"cpu_load_1m", &TelemetrySample::cpu_load_1m, Presence::optional>,
                            Field<"timestamp", &TelemetrySample::at>>;
};

struct HazardSample {
  std::string node{};
  double lambda{};
  Instant at{};
};

template <> struct describe<HazardSample> {
  using fields =
      std::tuple<Field<"node_id", &HazardSample::node>, Field<"lambda_e", &HazardSample::lambda>,
                 Field<"timestamp", &HazardSample::at>>;
};

struct MigrationPrior {
  std::string node{};
  double mean_migration_s{};
};

template <> struct describe<MigrationPrior> {
  using fields = std::tuple<Field<"node_id", &MigrationPrior::node>,
                            Field<"mean_migration_s", &MigrationPrior::mean_migration_s>>;
};

struct ScenarioSource {
  std::filesystem::path directory{};
  std::optional<std::filesystem::path> migration_priors{};
  bool load_hazard_samples = false;
  bool load_episodes = true;
};

struct InputDigest {
  std::string file{};
  std::uintmax_t bytes{};
};

template <> struct describe<InputDigest> {
  using fields = std::tuple<Field<"file", &InputDigest::file>, Field<"bytes", &InputDigest::bytes>>;
};

class Scenario {
public:
  [[nodiscard]] static Expected<Scenario> load(const ScenarioSource &source);

  std::filesystem::path directory{};
  std::filesystem::path topology_file{};
  std::vector<NodeSpec> nodes{};
  std::vector<JobSpec> jobs{};
  std::vector<ReclaimSpec> reclaims{};
  std::vector<EpisodeSpec> episodes{};
  std::vector<EpisodeSpec> lifecycle_segments{};
  std::vector<CapacitySample> node_capacity{};
  std::vector<CapacitySample> core_capacity{};
  std::vector<TelemetrySample> telemetry{};
  std::vector<HazardSample> hazard{};
  std::map<std::string, double, std::less<>> migration_priors_s{};
  std::vector<InputDigest> inputs{};
  Calendar calendar{};
  Instant window_start{};
  Instant window_end{};

  [[nodiscard]] const NodeSpec *find_node(std::string_view id) const noexcept;
  [[nodiscard]] std::optional<std::size_t> job_index(std::string_view id) const noexcept;
  [[nodiscard]] const ReclaimSpec *find_reclaim(std::string_view event_id) const noexcept;
  [[nodiscard]] const EpisodeSpec *episode_of(std::string_view event_id) const noexcept;

private:
  std::map<std::string, std::size_t, std::less<>> node_index_{};
  std::map<std::string, std::size_t, std::less<>> job_index_{};
  std::map<std::string, std::size_t, std::less<>> reclaim_index_{};
  std::map<std::string, std::size_t, std::less<>> episode_index_{};
  void index();
};

}
