#include "rnbt/job.hpp"
#include "rnbt/gpu.hpp"
#include "rnbt/jsonl.hpp"
#include "rnbt/shell.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <thread>

using namespace rnbt;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

struct Workspace {
  fs::path root = fs::temp_directory_path() / std::format("rnbt-job-{}", ::getpid());
  Workspace() { fs::create_directories(root); }
  ~Workspace() {
    std::ignore = run_command(
        std::format("{} kill-all --root {}", RNBT_JOB_BINARY, shell_quote(root.string())), 10s);
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

int run_job(std::string_view arguments) {
  return run_command(std::format("{} {}", RNBT_JOB_BINARY, arguments), 20s).exit_code;
}

std::optional<std::string> gpu_unavailable() {
  job::Gpu gpu;
  if (auto initialized = gpu.initialize(1048576, 1); !initialized)
    return initialized.error().message;
  return std::nullopt;
}

std::string file_contents(const fs::path &file, std::streamoff offset = 0) {
  std::ifstream in{file, std::ios::binary};
  in.seekg(offset);
  return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

void expect_failed_start(std::string_view resource_arguments, bool hide_devices = false,
                         bool expect_failed_state = false, std::string_view expected_error = {}) {
  Workspace w;
  const auto state = w.root / "state";
  const auto logs = w.root / "logs";
  const auto command = std::format(
      "{}{} start --state {} --instance invalid --rank 0 --node n1 --log-dir {} --target-s 600 "
      "--checkpoint-bytes 8192 --vram-bytes 1048576 --gpus 1 {}",
      hide_devices ? "CUDA_VISIBLE_DEVICES=-1 " : "", shell_quote(RNBT_JOB_BINARY),
      shell_quote(state.string()), shell_quote(logs.string()), resource_arguments);
  const auto result = run_command(command, 20s);
  EXPECT_NE(result.exit_code, 0) << command;
  EXPECT_FALSE(result.timed_out);
  EXPECT_FALSE(result.output.empty()) << "A failed start must explain the failure";
  if (!expected_error.empty())
    EXPECT_NE(result.output.find(expected_error), std::string::npos) << result.output;
  const auto failed = job::load_state(state);
  if (expect_failed_state)
    ASSERT_TRUE(failed) << result.output;
  if (failed) {
    EXPECT_NE(failed->status, "running");
    if (expect_failed_state) {
      EXPECT_EQ(failed->status, "failed");
      EXPECT_FALSE(failed->error.empty());
    }
    EXPECT_DOUBLE_EQ(failed->progress_s, 0.0);
  }
}

std::vector<job::LogLine> log_of(const fs::path &directory, std::string_view instance) {
  std::vector<job::LogLine> lines;
  for (auto &&line : records<job::LogLine>(directory / std::format("{}.r0.jsonl", instance)))
    lines.push_back(line);
  return lines;
}

}

TEST(Job, ProgressPauseCheckpointRestoreAndPark) {
  if (auto unavailable = gpu_unavailable())
    GTEST_SKIP() << "A real CUDA GPU and NVML are required: " << *unavailable;
  Workspace w;
  const auto state = w.root / "state";
  const auto logs = w.root / "logs";
  ASSERT_EQ(run_job(std::format(
                "start --state {} --instance j@e --rank 0 --node n1 --log-dir {} --target-s 600 "
                "--checkpoint-bytes 1048576 --vram-bytes 1048576 --gpus 1 "
                "--tick-s 0.05 --log-every-s 0.1",
                state.string(), logs.string())),
            0);
  std::this_thread::sleep_for(600ms);
  auto running = job::load_state(state);
  ASSERT_TRUE(running);
  EXPECT_EQ(running->status, "running");
  EXPECT_GT(running->progress_s, 0.3);
  ASSERT_EQ(run_job(std::format("pause --state {}", state.string())), 0);
  EXPECT_EQ(fs::file_size(state / "payload.bin"), 1048576u - job::kHeaderBytes);
  const auto paused = job::load_state(state)->progress_s;
  std::this_thread::sleep_for(400ms);
  EXPECT_DOUBLE_EQ(job::load_state(state)->progress_s, paused);

  const auto checkpoint = w.root / "ckpt.bin";
  ASSERT_EQ(
      run_job(std::format("checkpoint --state {} --out {}", state.string(), checkpoint.string())),
      0);
  EXPECT_EQ(fs::file_size(checkpoint), 1048576u);
  const auto header = job::read_checkpoint(checkpoint);
  ASSERT_TRUE(header);
  EXPECT_DOUBLE_EQ(header->progress_s, paused);
  EXPECT_EQ(header->gpu_backend, "cuda");
  const auto gpu_payload = file_contents(checkpoint, job::kHeaderBytes);
  EXPECT_EQ(gpu_payload.size(), 1048576u - job::kHeaderBytes);
  EXPECT_TRUE(std::ranges::any_of(gpu_payload, [](char byte) { return byte != '\0'; }))
      << "The checkpoint must contain GPU-produced data rather than a sparse zero payload";

  ASSERT_EQ(run_job(std::format("resume --state {}", state.string())), 0);
  std::this_thread::sleep_for(300ms);
  EXPECT_GT(job::load_state(state)->progress_s, paused);
  ASSERT_EQ(run_job(std::format("stop --state {}", state.string())), 0);
  EXPECT_EQ(job::load_state(state)->status, "stopped");
  EXPECT_NE(run_job(std::format("checkpoint --state {} --out {}", state.string(),
                                (w.root / "x").string())),
            0);

  const auto moved = w.root / "moved";
  ASSERT_EQ(run_job(std::format("restore --state {} --in {} --log-dir {} --node n2", moved.string(),
                                checkpoint.string(), logs.string())),
            0);
  EXPECT_EQ(job::load_state(moved)->status, "restored");
  EXPECT_EQ(file_contents(moved / "payload.bin"), gpu_payload);
  ASSERT_EQ(run_job(std::format(
                "start --state {} --instance j@e --rank 0 --node n2 --log-dir {} --target-s 600 "
                "--checkpoint-bytes 1048576 --vram-bytes 1048576 --gpus 1 "
                "--tick-s 0.05 --exit-after-resume-s 0.3",
                moved.string(), logs.string())),
            0);
  std::this_thread::sleep_for(800ms);
  const auto parked = job::load_state(moved);
  ASSERT_TRUE(parked);
  EXPECT_EQ(parked->status, "parked");
  EXPECT_GE(parked->progress_s, paused);
  const auto lines = log_of(logs, "j@e");
  const auto has = [&](std::string_view event, std::string_view node) {
    return std::ranges::any_of(
        lines, [&](const job::LogLine &l) { return l.event == event && l.node == node; });
  };
  EXPECT_TRUE(has("start", "n1"));
  EXPECT_TRUE(has("pause", "n1"));
  EXPECT_TRUE(has("restore", "n2"));
  EXPECT_TRUE(has("resume-start", "n2"));
  EXPECT_TRUE(has("park", "n2"));
}

TEST(Job, KillAllMarksRunningJobs) {
  if (auto unavailable = gpu_unavailable())
    GTEST_SKIP() << "A real CUDA GPU and NVML are required: " << *unavailable;
  Workspace w;
  const auto state = w.root / "a" / "r0";
  ASSERT_EQ(run_job(std::format(
                "start --state {} --instance k --rank 0 --node n1 --log-dir {} --target-s 600 "
                "--checkpoint-bytes 8192 --vram-bytes 1048576 --gpus 1 --tick-s 0.05",
                state.string(), (w.root / "logs").string())),
            0);
  ASSERT_EQ(run_job(std::format("kill-all --root {}", w.root.string())), 0);
  const auto killed = job::load_state(state);
  ASSERT_TRUE(killed);
  EXPECT_EQ(killed->status, "killed");
  EXPECT_NE(run_job(std::format("checkpoint --state {} --out {}", state.string(),
                                (w.root / "y").string())),
            0);
}

TEST(Job, MissingCudaDeviceFailsBeforeReportingRunning) { expect_failed_start("", true, true); }

TEST(Job, ProbeFailsWithoutCudaDevice) {
  const auto result = run_command(
      std::format("CUDA_VISIBLE_DEVICES=-1 {} probe", shell_quote(RNBT_JOB_BINARY)), 20s);
  EXPECT_NE(result.exit_code, 0);
  EXPECT_FALSE(result.timed_out);
  EXPECT_FALSE(result.output.empty()) << "A failed GPU probe must explain the failure";
}

TEST(Job, InsufficientDeviceCountFailsBeforeReportingRunning) {
  if (auto unavailable = gpu_unavailable())
    GTEST_SKIP() << "A real CUDA GPU and NVML are required: " << *unavailable;
  expect_failed_start(std::format("--gpus {} --vram-bytes {}",
                                  std::numeric_limits<std::uint32_t>::max(),
                                  std::numeric_limits<std::uint64_t>::max()),
                      false, true);
}

TEST(Job, InsufficientVramFailsBeforeReportingRunning) {
  if (auto unavailable = gpu_unavailable())
    GTEST_SKIP() << "A real CUDA GPU and NVML are required: " << *unavailable;
  expect_failed_start(std::format("--vram-bytes {}", std::numeric_limits<std::uint64_t>::max()),
                      false, true);
}

TEST(Job, InvalidGpuResourcesAreRejected) {
  for (const std::string_view arguments :
       {"--gpus 0", "--vram-bytes 0", "--checkpoint-bytes 4096", "--checkpoint-bytes 0",
        "--gpus -1", "--gpus abc", "--gpus 4294967296", "--vram-bytes -1", "--vram-bytes abc",
        "--vram-bytes 18446744073709551616", "--checkpoint-bytes -1", "--checkpoint-bytes abc"}) {
    SCOPED_TRACE(arguments);
    expect_failed_start(arguments, false, true, arguments.substr(0, arguments.find(' ')));
  }
}

TEST(Job, RejectsLegacyAndTruncatedCheckpointsWithoutHardware) {
  Workspace w;
  const job::CheckpointHeader header{
      .instance = "invalid", .target_s = 600, .checkpoint_bytes = 8192, .vram_bytes = 1048576};
  const auto write_fixture = [&](const fs::path &file, const Json &metadata, bool with_payload) {
    std::string head = metadata.dump();
    head.resize(job::kHeaderBytes, ' ');
    std::ofstream out{file, std::ios::binary};
    out.write(head.data(), static_cast<std::streamsize>(head.size()));
    if (with_payload) {
      const std::string body(header.checkpoint_bytes - job::kHeaderBytes, '\0');
      out.write(body.data(), static_cast<std::streamsize>(body.size()));
    }
  };

  auto legacy = encode(header);
  legacy["magic"] = "rnbt-job-checkpoint/1";
  legacy.erase("gpu_backend");
  const auto old_file = w.root / "legacy.bin";
  write_fixture(old_file, legacy, true);
  const auto old_checkpoint = job::read_checkpoint(old_file);
  ASSERT_FALSE(old_checkpoint);
  EXPECT_FALSE(old_checkpoint.error().message.empty());

  const auto truncated_file = w.root / "truncated.bin";
  write_fixture(truncated_file, encode(header), false);
  const auto truncated_checkpoint = job::read_checkpoint(truncated_file);
  ASSERT_FALSE(truncated_checkpoint);
  EXPECT_FALSE(truncated_checkpoint.error().message.empty());

  auto non_cuda = encode(header);
  non_cuda["gpu_backend"] = "cpu";
  const auto wrong_backend_file = w.root / "wrong-backend.bin";
  write_fixture(wrong_backend_file, non_cuda, true);
  const auto wrong_backend = job::read_checkpoint(wrong_backend_file);
  ASSERT_FALSE(wrong_backend);
  EXPECT_FALSE(wrong_backend.error().message.empty());
}
