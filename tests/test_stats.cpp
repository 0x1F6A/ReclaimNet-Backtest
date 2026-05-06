#include "rnbt/stats.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace rnbt::stats;

TEST(Stats, QuantilesInterpolate) {
  EXPECT_DOUBLE_EQ(*quantile({4.0, 1.0, 3.0, 2.0}, 0.5), 2.5);
  EXPECT_DOUBLE_EQ(*quantile({4.0, 1.0, 3.0, 2.0}, 1.0), 4.0);
  EXPECT_FALSE(quantile({}, 0.5));
  EXPECT_DOUBLE_EQ(*median(std::vector{5, 1, 3}), 3.0);
}

TEST(Stats, WilcoxonHandlesTiesAndZeros) {
  const std::vector<double> differences{1, 2, 3, 4, 5, -1, 0};
  const auto test = wilcoxon_signed_rank(differences);
  EXPECT_EQ(test.n, 6u);
  EXPECT_DOUBLE_EQ(test.w_plus, 19.5);
  EXPECT_NEAR(test.z, 8.5 / std::sqrt(22.625), 1e-9);
  EXPECT_NEAR(test.p_two_sided, std::erfc(test.z / std::sqrt(2.0)), 1e-12);
  EXPECT_DOUBLE_EQ(wilcoxon_signed_rank(std::vector<double>{0, 0}).p_two_sided, 1.0);
}

TEST(Stats, BootstrapOfIdenticalRunsIsDegenerate) {
  const std::vector<std::vector<std::pair<double, double>>> blocks{{{1, 1}, {2, 2}}, {{3, 3}}};
  const auto interval = bootstrap_ratio(blocks, 200, 7);
  EXPECT_DOUBLE_EQ(interval.estimate, 1.0);
  EXPECT_DOUBLE_EQ(interval.low, 1.0);
  EXPECT_DOUBLE_EQ(interval.high, 1.0);
}
