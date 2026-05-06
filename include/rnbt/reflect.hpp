#pragma once

#include "rnbt/core.hpp"

#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace rnbt {

using Json = nlohmann::json;

template <std::size_t N> struct FixedString {
  std::array<char, N> chars{};
  consteval FixedString(const char (&text)[N]) { std::copy_n(text, N, chars.begin()); }
  [[nodiscard]] constexpr std::string_view view() const noexcept { return {chars.data(), N - 1}; }
};

template <class M> struct member_pointer_traits;
template <class C, class T> struct member_pointer_traits<T C::*> {
  using owner = C;
  using type = T;
};

enum class Presence : std::uint8_t { required, optional };

template <FixedString Key, auto Member, Presence P = Presence::required, FixedString LegacyKey = "">
  requires std::is_member_object_pointer_v<decltype(Member)>
struct Field {
  using owner = typename member_pointer_traits<decltype(Member)>::owner;
  using type = typename member_pointer_traits<decltype(Member)>::type;
  static constexpr std::string_view key = Key.view();
  static constexpr std::string_view legacy_key = LegacyKey.view();
  static constexpr auto member = Member;
  static constexpr Presence presence = P;
};

template <class T> struct describe;

template <class T>
concept Described = requires { typename describe<std::remove_cvref_t<T>>::fields; };

template <class T> struct is_optional : std::false_type {};
template <class T> struct is_optional<std::optional<T>> : std::true_type {};
template <class T> struct is_vector : std::false_type {};
template <class T, class A> struct is_vector<std::vector<T, A>> : std::true_type {};
template <class T> struct is_string_map : std::false_type {};
template <class V, class C, class A>
struct is_string_map<std::map<std::string, V, C, A>> : std::true_type {};
template <class T> struct is_set : std::false_type {};
template <class T, class C, class A> struct is_set<std::set<T, C, A>> : std::true_type {};

struct DecodeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

template <class V> [[nodiscard]] Json encode(const V &value);
template <class V> [[nodiscard]] V decode(const Json &value);

template <Described T> [[nodiscard]] Json encode_object(const T &value) {
  Json out = Json::object();
  [&]<class... F>(std::type_identity<std::tuple<F...>>) {
    ((out[std::string{F::key}] = encode(value.*F::member)), ...);
  }(std::type_identity<typename describe<T>::fields>{});
  return out;
}

template <class F, class T> void decode_field(const Json &in, T &target) {
  using member = typename F::type;
  auto found = in.find(F::key);
  if constexpr (!F::legacy_key.empty()) {
    const auto legacy = in.find(F::legacy_key);
    if (found == in.end())
      found = legacy;
    else if (legacy != in.end() && *found != *legacy)
      throw DecodeError{std::format("conflicting fields '{}' and '{}'", F::key, F::legacy_key)};
  }
  if (found == in.end() || found->is_null()) {
    if constexpr (F::presence == Presence::optional || is_optional<member>::value)
      return;
    else
      throw DecodeError{std::format("missing field '{}'", F::key)};
  }
  try {
    target.*F::member = decode<member>(*found);
  } catch (const DecodeError &error) {
    throw DecodeError{std::format("{}: {}", F::key, error.what())};
  } catch (const Json::exception &error) {
    throw DecodeError{std::format("{}: {}", F::key, error.what())};
  }
}

template <Described T> void decode_object(const Json &in, T &target) {
  if (!in.is_object())
    throw DecodeError{"expected a JSON object"};
  [&]<class... F>(std::type_identity<std::tuple<F...>>) {
    (decode_field<F>(in, target), ...);
  }(std::type_identity<typename describe<T>::fields>{});
}

template <class V> Json encode(const V &value) {
  if constexpr (std::same_as<V, Instant>)
    return ns_of(value);
  else if constexpr (std::same_as<V, Nanos>)
    return seconds_of(value);
  else if constexpr (NamedEnum<V>)
    return std::string{name_of(value)};
  else if constexpr (QuantityType<V>)
    return value.value;
  else if constexpr (is_optional<V>::value)
    return value ? encode(*value) : Json(nullptr);
  else if constexpr (is_vector<V>::value || is_set<V>::value) {
    Json out = Json::array();
    for (const auto &item : value)
      out.push_back(encode(item));
    return out;
  } else if constexpr (is_string_map<V>::value) {
    Json out = Json::object();
    for (const auto &[key, item] : value)
      out[key] = encode(item);
    return out;
  } else if constexpr (Described<V>)
    return encode_object(value);
  else
    return Json(value);
}

template <class V> V decode(const Json &in) {
  if constexpr (std::same_as<V, Instant>)
    return instant_of(in.get<std::int64_t>());
  else if constexpr (std::same_as<V, Nanos>)
    return nanos_of(in.get<double>());
  else if constexpr (NamedEnum<V>) {
    const auto text = in.get<std::string>();
    if (auto parsed = parse_enum<V>(text))
      return *parsed;
    throw DecodeError{std::format("unknown enumerator '{}'", text)};
  } else if constexpr (QuantityType<V>)
    return V{in.get<typename V::rep>()};
  else if constexpr (is_optional<V>::value) {
    if (in.is_null())
      return std::nullopt;
    return decode<typename V::value_type>(in);
  } else if constexpr (is_vector<V>::value) {
    if (!in.is_array())
      throw DecodeError{"expected a JSON array"};
    V out;
    out.reserve(in.size());
    for (const auto &item : in)
      out.push_back(decode<typename V::value_type>(item));
    return out;
  } else if constexpr (is_set<V>::value) {
    V out;
    for (const auto &item : in)
      out.insert(decode<typename V::value_type>(item));
    return out;
  } else if constexpr (is_string_map<V>::value) {
    V out;
    for (const auto &[key, item] : in.items())
      out.emplace(key, decode<typename V::mapped_type>(item));
    return out;
  } else if constexpr (Described<V>) {
    V out{};
    decode_object(in, out);
    return out;
  } else
    return in.get<V>();
}

template <class V> [[nodiscard]] Expected<V> try_decode(const Json &in) {
  try {
    return decode<V>(in);
  } catch (const std::exception &error) {
    return failure(error.what());
  }
}

}
