#pragma once

#include "rnbt/reflect.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace rnbt::job {

struct JobState {
  std::string instance{};
  std::uint32_t rank{};
  std::string node{};
  double target_s{};
  double progress_s{};
  std::uint64_t checkpoint_bytes{};
  std::uint64_t vram_bytes{};
  std::uint32_t gpus{1};
  std::optional<double> exit_after_resume_s{};
  std::string log_dir{};
  std::string status = "new";
  bool restored{};
  std::int64_t pid{};
  std::int64_t updated_ns{};
  double tick_s = 1.0;
  double log_every_s = 10.0;
  std::string error{};
  Json gpu_samples = Json::array();
  bool external_load = false;
};

struct CheckpointHeader {
  std::string magic = "rnbt-job-checkpoint/2-cuda";
  std::string gpu_backend = "cuda";
  bool external_load = false;
  std::string instance{};
  std::uint32_t rank{};
  double progress_s{};
  double target_s{};
  std::uint64_t checkpoint_bytes{};
  std::uint64_t vram_bytes{};
  std::uint32_t gpus{1};
  std::int64_t captured_ns{};
  std::string node{};
};

struct LogLine {
  std::int64_t t_ns{};
  std::string instance{};
  std::uint32_t rank{};
  std::string node{};
  std::string event{};
  double progress_s{};
  std::int64_t pid{};
  std::string status{};
  Json gpu_samples = Json::array();
};

[[nodiscard]] int run(std::span<char *const> arguments);

[[nodiscard]] Expected<JobState> load_state(const std::filesystem::path &directory);
[[nodiscard]] Expected<CheckpointHeader> read_checkpoint(const std::filesystem::path &file);
inline constexpr std::size_t kHeaderBytes = 4096;

}

namespace rnbt {

template <> struct describe<job::JobState> {
  using S = job::JobState;
  using fields = std::tuple<
      Field<"instance", &S::instance>, Field<"rank", &S::rank>, Field<"node", &S::node>,
      Field<"target_s", &S::target_s>, Field<"progress_s", &S::progress_s>,
      Field<"checkpoint_bytes", &S::checkpoint_bytes>, Field<"vram_bytes", &S::vram_bytes>,
      Field<"gpus", &S::gpus>, Field<"exit_after_resume_s", &S::exit_after_resume_s>,
      Field<"log_dir", &S::log_dir>, Field<"status", &S::status>, Field<"restored", &S::restored>,
      Field<"pid", &S::pid>, Field<"updated_ns", &S::updated_ns>,
      Field<"tick_s", &S::tick_s, Presence::optional>,
      Field<"log_every_s", &S::log_every_s, Presence::optional>,
      Field<"error", &S::error, Presence::optional>,
      Field<"gpu_samples", &S::gpu_samples, Presence::optional>,
      Field<"external_load", &S::external_load, Presence::optional>>;
};

template <> struct describe<job::CheckpointHeader> {
  using H = job::CheckpointHeader;
  using fields = std::tuple<Field<"magic", &H::magic>, Field<"gpu_backend", &H::gpu_backend>,
                            Field<"external_load", &H::external_load, Presence::optional>,
                            Field<"instance", &H::instance>, Field<"rank", &H::rank>,
                            Field<"progress_s", &H::progress_s>, Field<"target_s", &H::target_s>,
                            Field<"checkpoint_bytes", &H::checkpoint_bytes>,
                            Field<"vram_bytes", &H::vram_bytes>, Field<"gpus", &H::gpus>,
                            Field<"captured_ns", &H::captured_ns>, Field<"node", &H::node>>;
};

template <> struct describe<job::LogLine> {
  using L = job::LogLine;
  using fields = std::tuple<Field<"t_ns", &L::t_ns>, Field<"instance", &L::instance>,
                            Field<"rank", &L::rank>, Field<"node", &L::node>,
                            Field<"event", &L::event>, Field<"progress_s", &L::progress_s>,
                            Field<"pid", &L::pid>, Field<"status", &L::status>,
                            Field<"gpu_samples", &L::gpu_samples, Presence::optional>>;
};

}
