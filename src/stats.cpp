#include "rnbt/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace rnbt::stats {

std::optional<double> quantile(std::vector<double> values, double q) {
  if (values.empty())
    return std::nullopt;
  std::ranges::sort(values);
  const double position = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = std::min(lower + 1, values.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

Wilcoxon wilcoxon_signed_rank(std::span<const double> differences) {
  std::vector<std::pair<double, int>> nonzero;
  for (const double d : differences)
    if (d != 0.0 && std::isfinite(d))
      nonzero.emplace_back(std::abs(d), d > 0 ? 1 : -1);
  Wilcoxon out{.n = nonzero.size()};
  if (nonzero.empty())
    return out;
  std::ranges::sort(nonzero, std::less<>{}, &std::pair<double, int>::first);
  std::vector<double> ranks(nonzero.size());
  double tie_correction = 0.0;
  for (std::size_t i = 0; i < nonzero.size();) {
    std::size_t j = i;
    while (j + 1 < nonzero.size() && nonzero[j + 1].first == nonzero[i].first)
      ++j;
    const double rank = (static_cast<double>(i + j) + 2.0) / 2.0;
    for (std::size_t k = i; k <= j; ++k)
      ranks[k] = rank;
    const double t = static_cast<double>(j - i + 1);
    tie_correction += t * t * t - t;
    i = j + 1;
  }
  for (std::size_t i = 0; i < nonzero.size(); ++i)
    if (nonzero[i].second > 0)
      out.w_plus += ranks[i];
  const double n = static_cast<double>(nonzero.size());
  const double mean = n * (n + 1.0) / 4.0;
  const double variance = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tie_correction / 48.0;
  if (variance <= 0.0)
    return out;
  const double continuity = out.w_plus > mean ? -0.5 : out.w_plus < mean ? 0.5 : 0.0;
  out.z = (out.w_plus - mean + continuity) / std::sqrt(variance);
  out.p_two_sided = std::erfc(std::abs(out.z) / std::sqrt(2.0));
  return out;
}

Interval bootstrap_ratio(std::span<const std::vector<std::pair<double, double>>> blocks,
                         std::size_t iterations, std::uint64_t seed) {
  const auto ratio = [](double a, double b) { return b > 0.0 ? a / b : 0.0; };
  double total_a = 0.0, total_b = 0.0;
  for (const auto &block : blocks)
    for (const auto &[a, b] : block) {
      total_a += a;
      total_b += b;
    }
  Interval out{.estimate = ratio(total_a, total_b)};
  if (blocks.empty() || iterations == 0) {
    out.low = out.high = out.estimate;
    return out;
  }
  std::mt19937_64 engine{seed};
  std::uniform_int_distribution<std::size_t> pick{0, blocks.size() - 1};
  std::vector<double> samples;
  samples.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    double a = 0.0, b = 0.0;
    for (std::size_t k = 0; k < blocks.size(); ++k)
      for (const auto &[x, y] : blocks[pick(engine)]) {
        a += x;
        b += y;
      }
    samples.push_back(ratio(a, b));
  }
  out.low = quantile(samples, 0.025).value_or(out.estimate);
  out.high = quantile(samples, 0.975).value_or(out.estimate);
  return out;
}

}
