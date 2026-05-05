#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <compare>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rnbt {

struct VirtualClock {
  using rep = std::int64_t;
  using period = std::nano;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<VirtualClock>;
  static constexpr bool is_steady = true;
};

using Nanos = VirtualClock::duration;
using Instant = VirtualClock::time_point;

inline constexpr Instant kNever = Instant::max();

[[nodiscard]] constexpr std::int64_t ns_of(Instant t) noexcept {
  return t.time_since_epoch().count();
}

[[nodiscard]] constexpr Instant instant_of(std::int64_t ns) noexcept { return Instant{Nanos{ns}}; }

[[nodiscard]] constexpr double seconds_of(Nanos d) noexcept {
  return std::chrono::duration<double>{d}.count();
}

[[nodiscard]] constexpr Nanos nanos_of(double seconds) noexcept {
  if (seconds != seconds)
    return Nanos::zero();
  if (seconds >= 9.2e9)
    return Nanos::max();
  if (seconds <= -9.2e9)
    return Nanos::min();
  return std::chrono::round<Nanos>(std::chrono::duration<double>{seconds});
}

[[nodiscard]] constexpr Instant saturating_add(Instant t, Nanos d) noexcept {
  if (d > Nanos::zero() && t > kNever - d)
    return kNever;
  return t + d;
}

template <class Tag, class Rep>
  requires std::is_arithmetic_v<Rep>
struct Quantity {
  using rep = Rep;
  Rep value{};

  [[nodiscard]] constexpr auto operator<=>(const Quantity &) const noexcept = default;

  constexpr Quantity &operator+=(const Quantity &other) noexcept {
    value += other.value;
    return *this;
  }
  constexpr Quantity &operator-=(const Quantity &other) noexcept {
    value -= other.value;
    return *this;
  }
  [[nodiscard]] friend constexpr Quantity operator+(Quantity a, const Quantity &b) noexcept {
    return a += b;
  }
  [[nodiscard]] friend constexpr Quantity operator-(Quantity a, const Quantity &b) noexcept {
    return a -= b;
  }
  template <class Scalar>
    requires std::is_arithmetic_v<Scalar>
  [[nodiscard]] friend constexpr Quantity operator*(Quantity a, Scalar s) noexcept {
    return Quantity{static_cast<Rep>(a.value * s)};
  }
  template <class Scalar>
    requires std::is_arithmetic_v<Scalar>
  [[nodiscard]] friend constexpr Quantity operator/(Quantity a, Scalar s) noexcept {
    return Quantity{static_cast<Rep>(a.value / s)};
  }
};

template <class T> struct is_quantity : std::false_type {};
template <class Tag, class Rep> struct is_quantity<Quantity<Tag, Rep>> : std::true_type {};
template <class T>
concept QuantityType = is_quantity<std::remove_cvref_t<T>>::value;

struct ByteTag {};
struct BitRateTag {};
using Bytes = Quantity<ByteTag, std::uint64_t>;
using BitRate = Quantity<BitRateTag, double>;

[[nodiscard]] constexpr double bits_of(Bytes b) noexcept {
  return static_cast<double>(b.value) * 8.0;
}

[[nodiscard]] constexpr Nanos transfer_time(Bytes b, BitRate r) noexcept {
  return r.value > 0.0 ? nanos_of(bits_of(b) / r.value) : Nanos::max();
}

namespace literals {
consteval Bytes operator""_B(unsigned long long v) { return Bytes{v}; }
consteval Bytes operator""_MB(unsigned long long v) { return Bytes{v * 1'000'000ULL}; }
consteval Bytes operator""_GB(unsigned long long v) { return Bytes{v * 1'000'000'000ULL}; }
consteval BitRate operator""_Mbps(unsigned long long v) {
  return BitRate{static_cast<double>(v) * 1e6};
}
consteval BitRate operator""_Gbps(unsigned long long v) {
  return BitRate{static_cast<double>(v) * 1e9};
}
}

struct Error {
  std::string message;
  std::source_location where = std::source_location::current();

  [[nodiscard]] std::string describe() const {
    return std::format("{} [{}:{}]", message, where.file_name(), where.line());
  }
};

template <class T> using Expected = std::expected<T, Error>;
using Status = Expected<void>;

[[nodiscard]] inline std::unexpected<Error>
failure(std::string message, std::source_location where = std::source_location::current()) {
  return std::unexpected<Error>{Error{std::move(message), where}};
}

template <class... Fs> struct overloaded : Fs... {
  using Fs::operator()...;
};

template <class E> struct enum_names;

template <class E>
concept NamedEnum = std::is_scoped_enum_v<E> && requires { enum_names<E>::values; };

template <NamedEnum E> [[nodiscard]] constexpr std::string_view name_of(E value) noexcept {
  const auto index = static_cast<std::size_t>(std::to_underlying(value));
  return index < enum_names<E>::values.size() ? enum_names<E>::values[index]
                                              : std::string_view{"UNKNOWN"};
}

template <NamedEnum E>
[[nodiscard]] constexpr std::optional<E> parse_enum(std::string_view text) noexcept {
  const auto &values = enum_names<E>::values;
  const auto found = std::ranges::find(values, text);
  if (found == values.end())
    return std::nullopt;
  return static_cast<E>(std::distance(values.begin(), found));
}

enum class ReclaimKind : std::uint8_t { scheduled, emergency };
template <> struct enum_names<ReclaimKind> {
  static constexpr std::array<std::string_view, 2> values{"SCHEDULED", "EMERGENCY"};
};

enum class ReplayMode : std::uint8_t { episodes, jobs };
template <> struct enum_names<ReplayMode> {
  static constexpr std::array<std::string_view, 2> values{"episodes", "jobs"};
};

enum class Outcome : std::uint8_t {
  handoff,
  checkpoint_recovery,
  degraded_recovery,
  restart_from_input,
  unresolved,
  not_running
};
template <> struct enum_names<Outcome> {
  static constexpr std::array<std::string_view, 6> values{
      "HANDOFF",    "CHECKPOINT_RECOVERY", "DEGRADED_RECOVERY", "RESTART_FROM_INPUT",
      "UNRESOLVED", "NOT_RUNNING"};
};

template <class T>
concept Timestamped = requires(const T &value) {
  { value.at } -> std::convertible_to<Instant>;
};

}

template <rnbt::NamedEnum E>
struct std::formatter<E, char> : std::formatter<std::string_view, char> {
  auto format(E value, std::format_context &ctx) const {
    return std::formatter<std::string_view, char>::format(rnbt::name_of(value), ctx);
  }
};

template <class Tag, class Rep>
struct std::formatter<rnbt::Quantity<Tag, Rep>, char> : std::formatter<Rep, char> {
  auto format(const rnbt::Quantity<Tag, Rep> &value, std::format_context &ctx) const {
    return std::formatter<Rep, char>::format(value.value, ctx);
  }
};
