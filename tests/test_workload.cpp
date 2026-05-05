#include "rnbt/workload.hpp"

#include "rnbt/jsonl.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace rnbt;

namespace {

struct ProfileFile {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      std::format("rnbt-workload-{}.json",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  Json input{
      {"schema_version", 1},
      {"trace_node", "node-001"},
      {"window_start_ns", 100'000'000'000LL},
      {"window_end_ns", 110'000'000'000LL},
      {"samples",
       Json::array(
           {{{"offset_s", 0.0}, {"gpu_util_pct", 10}, {"link_tx_bps", 1000}, {"link_rx_bps", 2000}},
            {{"offset_s", 2.0}, {"gpu_util_pct", 80}, {"link_tx_bps", 3000}, {"link_rx_bps", 4000}},
            {{"offset_s", 7.0}, {"gpu_util_pct", 0}, {"link_tx_bps", 0}, {"link_rx_bps", 0}}})}};

  ~ProfileFile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }

  Expected<job::WorkloadProfile> load() const {
    if (auto status = write_json_file(path, input); !status)
      return std::unexpected{status.error()};
    return job::WorkloadProfile::load(path);
  }
};

}

TEST(WorkloadProfile, HoldsTargetsUntilNextSampleAndIncludesWindowEnd) {
  ProfileFile file;
  const auto profile = file.load();
  ASSERT_TRUE(profile) << profile.error().message;
  EXPECT_EQ(profile->trace_node(), "node-001");
  EXPECT_EQ(profile->window_start_ns(), 100'000'000'000LL);
  EXPECT_EQ(profile->window_end_ns(), 110'000'000'000LL);
  EXPECT_DOUBLE_EQ(profile->duration_s(), 10);
  ASSERT_TRUE(profile->at(0));
  EXPECT_DOUBLE_EQ(profile->at(0)->gpu_util_pct, 10);
  EXPECT_DOUBLE_EQ(profile->at(1.999)->gpu_util_pct, 10);
  EXPECT_DOUBLE_EQ(profile->at(2)->gpu_util_pct, 80);
  EXPECT_DOUBLE_EQ(profile->at(6.999)->link_tx_bps, 3000);
  EXPECT_DOUBLE_EQ(profile->at(6.999)->link_rx_bps, 4000);
  EXPECT_DOUBLE_EQ(profile->at(7)->gpu_util_pct, 0);
  EXPECT_DOUBLE_EQ(profile->at(10)->gpu_util_pct, 0);
}

TEST(WorkloadProfile, RejectsInvalidQueryOffsets) {
  ProfileFile file;
  const auto profile = file.load();
  ASSERT_TRUE(profile) << profile.error().message;
  EXPECT_FALSE(profile->at(-0.01));
  EXPECT_FALSE(profile->at(10.001));
  EXPECT_FALSE(profile->at(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(profile->at(std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(job::WorkloadProfile{}.at(0));
}

TEST(WorkloadProfile, RejectsUnsortedMissingAndOutOfRangeSamples) {
  for (const auto &samples :
       {Json::array(), Json::array({{{"offset_s", 0}, {"gpu_util_pct", 10}, {"link_tx_bps", 0}}}),
        Json::array(
            {{{"offset_s", 1}, {"gpu_util_pct", 10}, {"link_tx_bps", 0}, {"link_rx_bps", 0}}}),
        Json::array(
            {{{"offset_s", 0}, {"gpu_util_pct", -1}, {"link_tx_bps", 0}, {"link_rx_bps", 0}}}),
        Json::array(
            {{{"offset_s", 0}, {"gpu_util_pct", 101}, {"link_tx_bps", 0}, {"link_rx_bps", 0}}}),
        Json::array(
            {{{"offset_s", 0}, {"gpu_util_pct", 50}, {"link_tx_bps", -1}, {"link_rx_bps", 0}}}),
        Json::array(
            {{{"offset_s", 0}, {"gpu_util_pct", 50}, {"link_tx_bps", 0}, {"link_rx_bps", -1}}})}) {
    ProfileFile file;
    file.input["samples"] = samples;
    EXPECT_FALSE(file.load()) << samples.dump();
  }
  for (const auto &offset : {Json(-1), Json(0), Json(11), Json("2"), Json(nullptr)}) {
    ProfileFile file;
    file.input["samples"][1]["offset_s"] = offset;
    EXPECT_FALSE(file.load()) << offset.dump();
  }
  ProfileFile decreasing;
  decreasing.input["samples"][2]["offset_s"] = 1;
  EXPECT_FALSE(decreasing.load());
}

TEST(WorkloadProfile, RejectsMalformedMetadataAndNonfiniteTargets) {
  for (const auto &change :
       {Json{{"schema_version", 2}}, Json{{"schema_version", 1.0}}, Json{{"trace_node", ""}},
        Json{{"window_start_ns", -1}}, Json{{"window_start_ns", 1.5}},
        Json{{"window_end_ns", 100'000'000'000LL}},
        Json{{"window_end_ns", std::numeric_limits<std::uint64_t>::max()}}}) {
    ProfileFile file;
    file.input.update(change);
    EXPECT_FALSE(file.load()) << change.dump();
  }
  for (const char *field : {"gpu_util_pct", "link_tx_bps", "link_rx_bps"}) {
    for (const auto &value :
         {Json(nullptr), Json("50"), Json(std::numeric_limits<double>::infinity())}) {
      ProfileFile file;
      file.input["samples"][0][field] = value;
      EXPECT_FALSE(file.load()) << field;
    }
  }
}
