#pragma once

#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace rnbt::stats {

template <class R>
concept NumericRange =
    std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, double>;

[[nodiscard]] std::optional<double> quantile(std::vector<double> values, double q);

template <NumericRange R> [[nodiscard]] std::optional<double> median(R &&values) {
  std::vector<double> copy{};
  for (auto &&value : values)
    copy.push_back(static_cast<double>(value));
  return quantile(std::move(copy), 0.5);
}

struct Wilcoxon {
  std::size_t n{};
  double w_plus{};
  double z{};
  double p_two_sided{1.0};
};

[[nodiscard]] Wilcoxon wilcoxon_signed_rank(std::span<const double> differences);

struct Interval {
  double estimate{};
  double low{};
  double high{};
};

[[nodiscard]] Interval
bootstrap_ratio(std::span<const std::vector<std::pair<double, double>>> blocks,
                std::size_t iterations, std::uint64_t seed);

}
