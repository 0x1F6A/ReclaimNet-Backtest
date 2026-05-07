#include "rnbt/calendar.hpp"

#include <charconv>

namespace rnbt {

namespace {

template <std::integral T>
[[nodiscard]] std::optional<T> digits(std::string_view text, std::size_t at, std::size_t count) {
  if (at + count > text.size())
    return std::nullopt;
  T value{};
  const auto *first = text.data() + at;
  const auto *last = first + count;
  auto [end, ec] = std::from_chars(first, last, value);
  if (ec != std::errc{} || end != last)
    return std::nullopt;
  return value;
}

}

Expected<ParsedTime> parse_iso8601(std::string_view text) {
  const auto year = digits<std::int64_t>(text, 0, 4);
  const auto month = digits<unsigned>(text, 5, 2);
  const auto day = digits<unsigned>(text, 8, 2);
  const auto hour = digits<unsigned>(text, 11, 2);
  const auto minute = digits<unsigned>(text, 14, 2);
  const auto second = digits<unsigned>(text, 17, 2);
  if (!year || !month || !day || !hour || !minute || !second || text.size() < 19 ||
      text[4] != '-' || text[7] != '-' || (text[10] != 'T' && text[10] != ' ') || text[13] != ':' ||
      text[16] != ':')
    return failure(std::format("invalid ISO-8601 timestamp '{}'", text));
  std::size_t cursor = 19;
  std::int64_t fraction = 0;
  if (cursor < text.size() && text[cursor] == '.') {
    std::int64_t scale = 100'000'000;
    ++cursor;
    while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
      fraction += (text[cursor] - '0') * scale;
      scale /= 10;
      ++cursor;
    }
  }
  Nanos offset{};
  if (cursor < text.size()) {
    const char sign = text[cursor];
    if (sign == 'Z') {
      ++cursor;
    } else if (sign == '+' || sign == '-') {
      const auto oh = digits<unsigned>(text, cursor + 1, 2);
      const auto om = digits<unsigned>(text, cursor + 4, 2);
      if (!oh || !om || text.size() < cursor + 6 || text[cursor + 3] != ':')
        return failure(std::format("invalid UTC offset in '{}'", text));
      offset = std::chrono::hours{*oh} + std::chrono::minutes{*om};
      if (sign == '-')
        offset = -offset;
      cursor += 6;
    }
  }
  if (cursor != text.size())
    return failure(std::format("trailing characters in timestamp '{}'", text));
  const std::int64_t local_ns =
      days_from_civil(*year, *month, *day) * kNanosPerDay +
      ((static_cast<std::int64_t>(*hour) * 60 + *minute) * 60 + *second) * 1'000'000'000LL +
      fraction;
  return ParsedTime{instant_of(local_ns - offset.count()), offset};
}

std::string format_iso8601(Instant t, Nanos offset) {
  const auto civil = civil_from_instant(t, offset);
  const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(offset);
  const auto absolute = minutes.count() < 0 ? -minutes.count() : minutes.count();
  return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}{}{:02}:{:02}", civil.year,
                     civil.month, civil.day, civil.hour, civil.minute, civil.second,
                     civil.nanos / 1'000'000, minutes.count() < 0 ? '-' : '+', absolute / 60,
                     absolute % 60);
}

}
