#include "rnbt/evaluate.hpp"
#include "rnbt/jsonl.hpp"

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

using namespace rnbt;

namespace {

struct WorkloadRunFiles {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      std::format("rnbt-workload-evaluate-{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  Json manifest{{"trace", std::string{RNBT_TEST_DATA} + "/mini_trace"},
                {"status", "completed"},
                {"gpu_backend", "cuda"},
                {"workload_profiles", "/prepared/profiles"},
                {"testbed", {{"nodes", Json::array({{{"trace_node", "node-001"}}})}}}};
  Json state{{"status", "completed"},
             {"error", ""},
             {"node", "node-001"},
             {"pid", 1234},
             {"samples_written", 0}};

  WorkloadRunFiles() {
    std::filesystem::create_directories(root);
    std::ofstream{root / "driver.jsonl"};
  }
  ~WorkloadRunFiles() {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }

  std::filesystem::path node_directory(std::string_view node = "node-001") const {
    return root / "collected" / "workloads" / node;
  }

  void write_state(std::string_view node = "node-001") const {
    const auto directory = node_directory(node);
    std::filesystem::create_directories(directory);
    ASSERT_TRUE(write_json_file(directory / "state.json", state));
  }

  void write_samples(std::initializer_list<Json> rows) const {
    const auto directory = node_directory() / "logs";
    std::filesystem::create_directories(directory);
    JsonlSink sink{directory / "samples.jsonl"};
    for (const auto &row : rows)
      sink.write_json(row);
    sink.flush();
  }

  Expected<RunMeasurement> measure() const {
    if (auto written = write_json_file(root / "run_manifest.json", manifest); !written)
      return std::unexpected{written.error()};
    return measure_run(root, MetricsSpec{});
  }
};

Json sample(double offset = 0, double target = 20, double first_gpu = 25, double second_gpu = 35) {
  const auto actual = (first_gpu + second_gpu) / 2;
  return {
      {"event", "sample"},
      {"node", "node-001"},
      {"pid", 1234},
      {"trace_offset_s", offset},
      {"target", {{"gpu_util_pct", target}, {"link_tx_bps", 123000000}, {"link_rx_bps", 45000000}}},
      {"actual_gpu_samples",
       Json::array({{{"uuid", "GPU-physical-1"}, {"utilization_gpu_percent", first_gpu}},
                    {{"uuid", "GPU-physical-2"}, {"utilization_gpu_percent", second_gpu}}})},
      {"actual_gpu_util_pct", actual},
      {"gpu_util_error_pct", actual - target}};
}

}

TEST(WorkloadEvaluate, AggregatesRealNvmlMeasurementsAndTargetErrors) {
  WorkloadRunFiles files;
  files.state["samples_written"] = 2;
  files.write_state();
  files.write_samples({sample(), sample(0.2, 80, 65, 75)});
  const auto measured = files.measure();
  ASSERT_TRUE(measured) << measured.error().message;
  const auto summary = summarize(*measured, MetricsSpec{});
  const auto &workload = summary.at("workload_replay");
  EXPECT_EQ(workload.at("samples"), 2);
  EXPECT_EQ(workload.at("target_source"), "trace_telemetry");
  EXPECT_EQ(workload.at("measurement_source"), "nvml");
  EXPECT_DOUBLE_EQ(workload.at("mean_target_gpu_util_pct").get<double>(), 50);
  EXPECT_DOUBLE_EQ(workload.at("mean_actual_gpu_util_pct").get<double>(), 50);
  EXPECT_DOUBLE_EQ(workload.at("mean_absolute_gpu_util_error_pct").get<double>(), 10);
  EXPECT_DOUBLE_EQ(workload.at("p95_absolute_gpu_util_error_pct").get<double>(), 10);
  EXPECT_EQ(workload.at("nodes").at(0).at("samples"), 2);
}

TEST(WorkloadEvaluate, RequiresCollectedStateForEveryMappedNode) {
  WorkloadRunFiles files;
  files.state["status"] = "stopped";
  files.write_state();
  files.manifest["testbed"]["nodes"].push_back({{"trace_node", "node-002"}});
  const auto measured = files.measure();
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("workload node-002"), std::string::npos);
}

TEST(WorkloadEvaluate, RejectsFailedRunningAndIncompleteActors) {
  for (const auto status : {"failed", "running", "starting", "unknown"}) {
    SCOPED_TRACE(status);
    WorkloadRunFiles files;
    files.state["status"] = status;
    files.write_state();
    const auto measured = files.measure();
    ASSERT_FALSE(measured);
    EXPECT_NE(measured.error().message.find("status"), std::string::npos);
  }
}

TEST(WorkloadEvaluate, RejectsErrorEvenWhenStateClaimsCompleted) {
  WorkloadRunFiles files;
  files.state["error"] = "CUDA synchronization failed";
  files.write_state();
  const auto measured = files.measure();
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("CUDA synchronization failed"), std::string::npos);
}

TEST(WorkloadEvaluate, RequiresReadableSamplesWhenStateReportsMeasurements) {
  WorkloadRunFiles files;
  files.state["samples_written"] = 1;
  files.write_state();
  const auto measured = files.measure();
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("sample logs"), std::string::npos);
}

TEST(WorkloadEvaluate, RejectsTruncatedCollection) {
  WorkloadRunFiles files;
  files.state["samples_written"] = 2;
  files.write_state();
  files.write_samples({sample()});
  const auto measured = files.measure();
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("state declares 2"), std::string::npos);
}

TEST(WorkloadEvaluate, RetainsMeasurementsAcrossActorRestarts) {
  WorkloadRunFiles files;
  files.state["samples_written"] = 1;
  files.state["status"] = "stopped";
  files.write_state();
  auto restarted = sample(100, 80, 65, 75);
  restarted["pid"] = 5678;
  files.write_samples({sample(), restarted});
  const auto measured = files.measure();
  ASSERT_TRUE(measured) << measured.error().message;
  EXPECT_EQ(measured->workload_summary.at("samples"), 2);
}

TEST(WorkloadEvaluate, ReportsNoMeasurementsAsNullWhenStoppedBeforeFirstSample) {
  WorkloadRunFiles files;
  files.state["status"] = "stopped";
  files.write_state();
  const auto measured = files.measure();
  ASSERT_TRUE(measured) << measured.error().message;
  const auto &workload = measured->workload_summary;
  EXPECT_EQ(workload.at("samples"), 0);
  EXPECT_TRUE(workload.at("mean_target_gpu_util_pct").is_null());
  EXPECT_TRUE(workload.at("mean_actual_gpu_util_pct").is_null());
  EXPECT_TRUE(workload.at("mean_absolute_gpu_util_error_pct").is_null());
  EXPECT_TRUE(workload.at("p95_absolute_gpu_util_error_pct").is_null());
}

TEST(WorkloadEvaluate, RejectsCompletedActorsWithoutRealMeasurements) {
  WorkloadRunFiles files;
  files.write_state();
  const auto measured = files.measure();
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("completed without real GPU measurements"),
            std::string::npos);
}

TEST(WorkloadEvaluate, LegacyAndNullProfilesDoNotRequireWorkloadLogs) {
  for (const bool remove : {false, true}) {
    WorkloadRunFiles files;
    if (remove)
      files.manifest.erase("workload_profiles");
    else
      files.manifest["workload_profiles"] = nullptr;
    const auto measured = files.measure();
    ASSERT_TRUE(measured) << measured.error().message;
    EXPECT_TRUE(measured->workload_summary.empty());
    EXPECT_FALSE(summarize(*measured, MetricsSpec{}).contains("workload_replay"));
  }
}

TEST(WorkloadEvaluate, RejectsMissingNvmlArraysAndInvalidOrForgedMeasurements) {
  const std::vector<std::pair<std::string, Json>> changes{
      {"/actual_gpu_samples", Json::array()},
      {"/actual_gpu_samples", Json::object()},
      {"/actual_gpu_samples/0/uuid", ""},
      {"/actual_gpu_samples/1/uuid", "GPU-physical-1"},
      {"/actual_gpu_samples/0/utilization_gpu_percent", 101},
      {"/actual_gpu_samples/0/utilization_gpu_percent", nullptr},
      {"/actual_gpu_util_pct", 20},
      {"/actual_gpu_util_pct", -1},
      {"/gpu_util_error_pct", 0},
      {"/gpu_util_error_pct", nullptr},
      {"/target/gpu_util_pct", 120},
      {"/target/link_tx_bps", -1},
      {"/target/link_rx_bps", "not numeric"},
      {"/trace_offset_s", -0.1},
      {"/trace_offset_s", nullptr},
      {"/node", "node-002"}};
  for (const auto &[pointer, value] : changes) {
    SCOPED_TRACE(pointer + "=" + value.dump());
    WorkloadRunFiles files;
    files.state["samples_written"] = 1;
    files.write_state();
    auto row = sample();
    row[Json::json_pointer{pointer}] = value;
    files.write_samples({row});
    const auto measured = files.measure();
    ASSERT_FALSE(measured);
    EXPECT_NE(measured.error().message.find("workload"), std::string::npos);
  }
}

TEST(WorkloadEvaluate, RejectsMalformedOrBackwardsSampleLogs) {
  WorkloadRunFiles files;
  files.state["samples_written"] = 2;
  files.write_state();
  files.write_samples({sample(1), sample(0)});
  auto measured = files.measure();
  ASSERT_FALSE(measured);
  EXPECT_NE(measured.error().message.find("backwards"), std::string::npos);
  std::ofstream{files.node_directory() / "logs" / "samples.jsonl"} << "{truncated\n";
  measured = files.measure();
  EXPECT_FALSE(measured);
}

TEST(WorkloadEvaluate, RejectsUnsafeNodePathsAndMissingMappedNodes) {
  for (const auto node : {"../escape", "", ".", "a/b", "a\\b"}) {
    SCOPED_TRACE(node);
    WorkloadRunFiles files;
    files.manifest["testbed"]["nodes"][0]["trace_node"] = node;
    const auto measured = files.measure();
    ASSERT_FALSE(measured);
    EXPECT_NE(measured.error().message.find("unsafe"), std::string::npos);
  }
  WorkloadRunFiles files;
  files.manifest["testbed"]["nodes"] = Json::array();
  EXPECT_FALSE(files.measure());
}
