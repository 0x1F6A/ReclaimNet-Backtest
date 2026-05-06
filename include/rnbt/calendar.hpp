#pragma once

#include "rnbt/core.hpp"

#include <string>
#include <string_view>

namespace rnbt {

struct CivilTime {
  std::int64_t year{};
  unsigned month{};
  unsigned day{};
  unsigned hour{};
  unsigned minute{};
  unsigned second{};
  std::int64_t nanos{};
  constexpr auto operator<=>(const CivilTime &) const = default;
};

[[nodiscard]] constexpr std::int64_t days_from_civil(std::int64_t y, unsigned m,
                                                     unsigned d) noexcept {
  y -= m <= 2 ? 1 : 0;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const auto yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

[[nodiscard]] constexpr CivilTime civil_from_days(std::int64_t z) noexcept {
  z += 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const auto doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const std::int64_t y = static_cast<std::int64_t>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  return CivilTime{.year = y + (m <= 2 ? 1 : 0), .month = m, .day = d};
}

inline constexpr std::int64_t kNanosPerDay = 86'400'000'000'000LL;

[[nodiscard]] constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept {
  return a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);
}

[[nodiscard]] constexpr CivilTime civil_from_instant(Instant t, Nanos offset) noexcept {
  const std::int64_t local = ns_of(t) + offset.count();
  const std::int64_t days = floor_div(local, kNanosPerDay);
  const std::int64_t within = local - days * kNanosPerDay;
  auto civil = civil_from_days(days);
  const std::int64_t seconds = within / 1'000'000'000LL;
  civil.hour = static_cast<unsigned>(seconds / 3600);
  civil.minute = static_cast<unsigned>((seconds / 60) % 60);
  civil.second = static_cast<unsigned>(seconds % 60);
  civil.nanos = within % 1'000'000'000LL;
  return civil;
}

static_assert(days_from_civil(1970, 1, 1) == 0);
static_assert(days_from_civil(2026, 3, 1) == 20513);
static_assert(civil_from_days(20513) == CivilTime{.year = 2026, .month = 3, .day = 1});

struct ParsedTime {
  Instant instant{};
  Nanos offset{};
};

[[nodiscard]] Expected<ParsedTime> parse_iso8601(std::string_view text);
[[nodiscard]] std::string format_iso8601(Instant t, Nanos offset);

class Calendar {
public:
  constexpr Calendar() noexcept = default;
  constexpr Calendar(Instant origin, Nanos offset) noexcept : origin_{origin}, offset_{offset} {}

  [[nodiscard]] constexpr Instant origin() const noexcept { return origin_; }
  [[nodiscard]] constexpr Nanos offset() const noexcept { return offset_; }

  [[nodiscard]] constexpr std::int64_t day_index(Instant t) const noexcept {
    const std::int64_t origin_local = ns_of(origin_) + offset_.count();
    const std::int64_t origin_day = floor_div(origin_local, kNanosPerDay);
    const std::int64_t local = ns_of(t) + offset_.count();
    return floor_div(local, kNanosPerDay) - origin_day;
  }

  [[nodiscard]] constexpr Instant day_start(std::int64_t index) const noexcept {
    const std::int64_t origin_local = ns_of(origin_) + offset_.count();
    const std::int64_t origin_day = floor_div(origin_local, kNanosPerDay);
    return instant_of((origin_day + index) * kNanosPerDay - offset_.count());
  }

  [[nodiscard]] std::string format(Instant t) const { return format_iso8601(t, offset_); }

private:
  Instant origin_{};
  Nanos offset_{};
};

}
