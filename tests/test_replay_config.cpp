#include "rnbt/replay_config.hpp"
#include "rnbt/evaluate.hpp"
#include "rnbt/jsonl.hpp"

#include <chrono>
#include <gtest/gtest.h>

using namespace rnbt;

namespace {

struct ReplayFiles {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      std::format("rnbt-replay-config-{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  Json testbed{{"local", true},
               {"coordinator", {{"host", "127.0.0.1"}, {"address", "127.0.0.1:50051"}}},
               {"nodes", Json::array({{{"trace_node", "node-001"},
                                       {"host", "127.0.0.1"},
                                       {"node_daemon_address", "127.0.0.1:50052"}}})}};

  ReplayFiles() { std::filesystem::create_directories(root); }
  ~ReplayFiles() {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }

  Expected<LoadedReplay> load() const {
    if (auto written = write_json_file(root / "testbed.json", testbed); !written)
      return std::unexpected{written.error()};
    if (auto written = write_json_file(root / "replay.json",
                                       Json{{"trace", "trace"}, {"testbed", "testbed.json"}});
        !written)
      return std::unexpected{written.error()};
    return load_replay(root / "replay.json", {});
  }
};

}

TEST(ReplayConfig, ReadsNewAndLegacyNodeDaemonFields) {
  for (const bool legacy : {false, true}) {
    const Json input{
        {"local", true},
        {"coordinator", {{"host", "127.0.0.1"}, {"address", "127.0.0.1:50051"}}},
        {"nodes",
         Json::array({{{"trace_node", "node-001"},
                       {"host", "127.0.0.1"},
                       {legacy ? "agent_address" : "node_daemon_address", "127.0.0.1:50052"}}})},
        {"commands",
         {{legacy ? "agent_start" : "node_daemon_start", "start-custom-service"},
          {legacy ? "agent_stop" : "node_daemon_stop", "stop-custom-service"}}}};
    const auto testbed = try_decode<Testbed>(input);
    ASSERT_TRUE(testbed) << testbed.error().message;
    ASSERT_EQ(testbed->nodes.size(), 1u);
    EXPECT_EQ(testbed->nodes.front().node_daemon_address, "127.0.0.1:50052");
    EXPECT_EQ(testbed->commands.node_daemon_start, "start-custom-service");
    EXPECT_EQ(testbed->commands.node_daemon_stop, "stop-custom-service");
    const auto output = encode(*testbed);
    EXPECT_TRUE(output["nodes"][0].contains("node_daemon_address"));
    EXPECT_FALSE(output["nodes"][0].contains("agent_address"));
    EXPECT_TRUE(output["commands"].contains("node_daemon_start"));
    EXPECT_TRUE(output["commands"].contains("node_daemon_stop"));
    EXPECT_FALSE(output["commands"].contains("agent_start"));
    EXPECT_FALSE(output["commands"].contains("agent_stop"));
  }
}

TEST(ReplayConfig, RejectsConflictingNewAndLegacyFields) {
  const Json node{{"trace_node", "node-001"},
                  {"host", "127.0.0.1"},
                  {"node_daemon_address", "127.0.0.1:50052"},
                  {"agent_address", "127.0.0.1:50053"}};
  EXPECT_FALSE(try_decode<HostConfig>(node));
  for (const auto *action : {"start", "stop"}) {
    const Json commands{{std::string{"node_daemon_"} + action, "new-command"},
                        {std::string{"agent_"} + action, "old-command"}};
    EXPECT_FALSE(try_decode<TestbedCommands>(commands));
  }
}

TEST(ReplayConfig, AllowsMatchingNewAndLegacyFields) {
  const Json commands{{"node_daemon_start", "start-custom-service"},
                      {"agent_start", "start-custom-service"}};
  const auto decoded = try_decode<TestbedCommands>(commands);
  ASSERT_TRUE(decoded) << decoded.error().message;
  EXPECT_EQ(decoded->node_daemon_start, "start-custom-service");
}

TEST(ReplayConfig, RejectsGpuEmulationEnvironmentAndCommandsWhenLoading) {
  const std::vector<std::pair<std::string, std::string>> settings{
      {"/nodes/0/vars/env", "RNBT_FAKE_GPU_DIR=/tmp/gpu"},
      {"/nodes/0/vars/env", "RNBT_FAKE_GPUS=4 RNBT_FAKE_VRAM_BYTES=1024"},
      {"/nodes/0/vars/RNBT_FAKE_GPUS", "4"},
      {"/coordinator/vars/env", "export RNBT_FAKE_GPU_DIR=/tmp/gpu"},
      {"/commands/node_daemon_start", "LD_LIBRARY_PATH=/opt/fake_nvml node_agent config.yaml"},
      {"/commands/agent_start",
       "LD_PRELOAD=/opt/fake-nvml/libnvidia-ml.so.1 node_agent config.yaml"},
      {"/commands/job_launch", "RNBT_FAKE_GPU_DIR=/tmp/gpu {job_binary} start"},
      {"/commands/remote", "ssh {ssh} RNBT_FAKE_GPUS=1 {quoted}"}};
  for (const auto &[pointer, value] : settings) {
    SCOPED_TRACE(pointer + "=" + value);
    ReplayFiles files;
    files.testbed[Json::json_pointer{pointer}] = value;
    const auto loaded = files.load();
    ASSERT_FALSE(loaded);
    EXPECT_NE(loaded.error().message.find("physical GPUs"), std::string::npos);
    EXPECT_NE(loaded.error().message.find(pointer), std::string::npos);
  }
}

TEST(ReplayConfig, AllowsPhysicalGpuConfigurationAndUnrelatedFakeLabels) {
  ReplayFiles files;
  files.testbed["nodes"][0]["vars"] = {{"label", "fake-outage"}, {"env", "CUDA_VISIBLE_DEVICES=0"}};
  files.testbed["commands"]["node_daemon_start"] = "start-service --label fake-outage";
  const auto loaded = files.load();
  ASSERT_TRUE(loaded) << loaded.error().message;
  EXPECT_EQ(loaded->testbed.nodes.front().vars.at("env"), "CUDA_VISIBLE_DEVICES=0");
}

TEST(ReplayConfig, KeepsProfileRunsInsideTheirOwnDirectories) {
  ReplayFiles files;
  ASSERT_TRUE(files.load());
  for (const auto *run_id : {"", ".", "..", "../other", "a/b"}) {
    const std::vector<std::string> overrides{"/workload_profiles=profiles",
                                             "/run_id=" + Json(run_id).dump()};
    const auto loaded = load_replay(files.root / "replay.json", overrides);
    ASSERT_FALSE(loaded) << run_id;
    EXPECT_NE(loaded.error().message.find("run_id"), std::string::npos);
  }
  const std::vector<std::string> overrides{"/workload_profiles=profiles", "/run_id=window-1.v2"};
  const auto loaded = load_replay(files.root / "replay.json", overrides);
  ASSERT_TRUE(loaded) << loaded.error().message;
  EXPECT_EQ(loaded->testbed.paths.log_dir, "/var/lib/rnbt/logs/runs/window-1.v2");
  EXPECT_EQ(loaded->testbed.paths.workload_root, "/var/lib/rnbt/workload/runs/window-1.v2");
}

TEST(ReplayConfig, RejectsEmptyProfileDirectoryAndUnfilledTestbed) {
  ReplayFiles files;
  ASSERT_TRUE(files.load());
  const std::vector<std::string> overrides{"/workload_profiles=\"\""};
  const auto empty = load_replay(files.root / "replay.json", overrides);
  ASSERT_FALSE(empty);
  EXPECT_NE(empty.error().message.find("workload_profiles"), std::string::npos);
  files.testbed["_template"] = true;
  const auto placeholder = files.load();
  ASSERT_FALSE(placeholder);
  EXPECT_NE(placeholder.error().message.find("template"), std::string::npos);
}

TEST(Evaluate, RejectsFailedInterruptedAndDryRunsBeforeReadingResults) {
  const std::vector<Json> manifests{{{"status", "failed"}},  {{"status", "interrupted"}},
                                    {{"status", "running"}}, {{"failed_actions", 1}},
                                    {{"interrupted", true}}, {{"dry_run", true}}};
  for (const auto &manifest : manifests) {
    SCOPED_TRACE(manifest.dump());
    ReplayFiles files;
    ASSERT_TRUE(write_json_file(files.root / "run_manifest.json", manifest));
    const auto measured = measure_run(files.root, MetricsSpec{});
    ASSERT_FALSE(measured);
    EXPECT_NE(measured.error().message.find("completed physical GPU run"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(files.root / "metrics.json"));
  }
}

TEST(Evaluate, AcceptsCompletedAndLegacyManifests) {
  for (const bool legacy : {false, true}) {
    ReplayFiles files;
    Json manifest{{"trace", std::string{RNBT_TEST_DATA} + "/mini_trace"}};
    if (!legacy) {
      manifest["status"] = "completed";
      manifest["gpu_backend"] = "cuda";
    }
    ASSERT_TRUE(write_json_file(files.root / "run_manifest.json", manifest));
    std::ofstream{files.root / "driver.jsonl"};
    const auto measured = measure_run(files.root, MetricsSpec{});
    ASSERT_TRUE(measured) << measured.error().message;
    EXPECT_TRUE(measured->outcomes.empty());
  }
}

TEST(Evaluate, RejectsMissingLogsAfterStartingPhysicalGpuJobs) {
  ReplayFiles files;
  const Json manifest{{"trace", std::string{RNBT_TEST_DATA} + "/mini_trace"},
                      {"status", "completed"},
                      {"gpu_backend", "cuda"}};
  ASSERT_TRUE(write_json_file(files.root / "run_manifest.json", manifest));
  {
    JsonlSink driver{files.root / "driver.jsonl"};
    driver.write_json(Json{{"type", "start"}, {"ok", true}});
    driver.flush();
  }
  const auto measured = measure_run(files.root, MetricsSpec{});
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("no collected GPU job logs"), std::string::npos);
}

TEST(Evaluate, RequiresHealthyGpuJobLogsWithSamplesForPhysicalRuns) {
  const Json samples = Json::array({{{"uuid", "GPU-physical-device"}}});
  const std::vector<job::LogLine> lines{
      {.instance = "job-a", .event = "gpu:sample", .status = "failed", .gpu_samples = samples},
      {.instance = "job-a",
       .event = "error:CUDA synchronization failed",
       .status = "running",
       .gpu_samples = samples},
      {.instance = "job-a", .event = "start", .status = "running"},
      {.instance = "job-a",
       .event = "resume-start",
       .status = "running",
       .gpu_samples = Json{{"uuid", "GPU-physical-device"}}},
      {.instance = "job-a", .event = "start", .status = "running", .gpu_samples = samples},
      {.instance = "job-a", .event = "resume-start", .status = "running", .gpu_samples = samples}};
  for (const bool legacy : {false, true}) {
    for (std::size_t index = 0; index < lines.size(); ++index) {
      SCOPED_TRACE(std::format("legacy={}, line={}", legacy, index));
      ReplayFiles files;
      Json manifest{{"trace", std::string{RNBT_TEST_DATA} + "/mini_trace"},
                    {"status", "completed"}};
      if (!legacy)
        manifest["gpu_backend"] = "cuda";
      ASSERT_TRUE(write_json_file(files.root / "run_manifest.json", manifest));
      std::ofstream{files.root / "driver.jsonl"};
      const auto log_dir = files.root / "collected" / "jobs" / "node-001";
      std::filesystem::create_directories(log_dir);
      {
        JsonlSink jobs{log_dir / "job-a.r0.jsonl"};
        jobs.write(lines[index]);
        jobs.flush();
      }
      const auto measured = measure_run(files.root, MetricsSpec{});
      if (legacy || index >= 4)
        EXPECT_TRUE(measured) << measured.error().message;
      else {
        ASSERT_FALSE(measured);
        EXPECT_NE(measured.error().message.find("GPU job job-a rank 0"), std::string::npos);
      }
    }
  }
}
