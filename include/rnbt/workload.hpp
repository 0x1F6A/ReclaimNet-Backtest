#pragma once

#include "rnbt/core.hpp"

#include <filesystem>
#include <vector>

namespace rnbt::job {

struct WorkloadTarget {
  double gpu_util_pct{};
  double link_tx_bps{};
  double link_rx_bps{};
};

class WorkloadProfile {
public:
  [[nodiscard]] static Expected<WorkloadProfile> load(const std::filesystem::path &path);

  [[nodiscard]] Expected<WorkloadTarget> at(double offset_s) const;
  [[nodiscard]] const std::string &trace_node() const noexcept { return trace_node_; }
  [[nodiscard]] std::int64_t window_start_ns() const noexcept { return window_start_ns_; }
  [[nodiscard]] std::int64_t window_end_ns() const noexcept { return window_end_ns_; }
  [[nodiscard]] double duration_s() const noexcept {
    return static_cast<double>(window_end_ns_ - window_start_ns_) / 1e9;
  }

private:
  struct Sample {
    double offset_s{};
    WorkloadTarget target;
  };
  std::string trace_node_;
  std::int64_t window_start_ns_{};
  std::int64_t window_end_ns_{};
  std::vector<Sample> samples_;
};

}
