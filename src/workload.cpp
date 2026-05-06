#include "rnbt/workload.hpp"

#include "rnbt/jsonl.hpp"

#include <cmath>

namespace rnbt::job {

namespace {

double finite_number(const Json &object, std::string_view field) {
  const auto &value = object.at(field);
  if (!value.is_number())
    throw DecodeError{std::format("{} must be a number", field)};
  const double number = value.get<double>();
  if (!std::isfinite(number))
    throw DecodeError{std::format("{} must be finite", field)};
  return number;
}

std::int64_t nonnegative_timestamp(const Json &object, std::string_view field) {
  const auto &value = object.at(field);
  if (!value.is_number_integer() ||
      (value.is_number_unsigned() &&
       value.get<std::uint64_t>() >
           static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
    throw DecodeError{std::format("{} must be a nonnegative int64 nanosecond timestamp", field)};
  const auto number = value.get<std::int64_t>();
  if (number < 0)
    throw DecodeError{std::format("{} must be nonnegative", field)};
  return number;
}

}

Expected<WorkloadProfile> WorkloadProfile::load(const std::filesystem::path &path) {
  const auto json = read_json_file(path);
  if (!json)
    return std::unexpected{json.error()};
  try {
    if (!json->is_object())
      throw DecodeError{"expected a workload profile object"};
    const auto &schema = json->at("schema_version");
    if (!schema.is_number_integer() || schema != 1)
      throw DecodeError{"schema_version must be 1"};

    WorkloadProfile profile;
    profile.trace_node_ = json->at("trace_node").get<std::string>();
    if (profile.trace_node_.empty())
      throw DecodeError{"trace_node must not be empty"};
    profile.window_start_ns_ = nonnegative_timestamp(*json, "window_start_ns");
    profile.window_end_ns_ = nonnegative_timestamp(*json, "window_end_ns");
    if (profile.window_end_ns_ <= profile.window_start_ns_)
      throw DecodeError{"window_end_ns must be after window_start_ns"};
    const auto &samples = json->at("samples");
    if (!samples.is_array() || samples.empty())
      throw DecodeError{"samples must be a nonempty array"};
    profile.samples_.reserve(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index) {
      try {
        const auto &input = samples.at(index);
        if (!input.is_object())
          throw DecodeError{"expected a sample object"};
        Sample sample;
        sample.offset_s = finite_number(input, "offset_s");
        sample.target.gpu_util_pct = finite_number(input, "gpu_util_pct");
        sample.target.link_tx_bps = finite_number(input, "link_tx_bps");
        sample.target.link_rx_bps = finite_number(input, "link_rx_bps");
        if (sample.offset_s < 0 || sample.offset_s > profile.duration_s())
          throw DecodeError{"offset_s must be within the selected window"};
        if (index == 0 && sample.offset_s != 0)
          throw DecodeError{"the first offset_s must be 0"};
        if (index > 0 && sample.offset_s <= profile.samples_.back().offset_s)
          throw DecodeError{"offset_s must be strictly increasing"};
        if (sample.target.gpu_util_pct < 0 || sample.target.gpu_util_pct > 100)
          throw DecodeError{"gpu_util_pct must be between 0 and 100"};
        if (sample.target.link_tx_bps < 0 || sample.target.link_rx_bps < 0)
          throw DecodeError{"link_tx_bps and link_rx_bps must be nonnegative"};
        profile.samples_.push_back(sample);
      } catch (const std::exception &error) {
        throw DecodeError{std::format("samples[{}]: {}", index, error.what())};
      }
    }
    return profile;
  } catch (const std::exception &error) {
    return failure(std::format("{}: {}", path.string(), error.what()));
  }
}

Expected<WorkloadTarget> WorkloadProfile::at(double offset_s) const {
  if (samples_.empty())
    return failure("workload profile has not been loaded");
  if (!std::isfinite(offset_s) || offset_s < 0 || offset_s > duration_s())
    return failure(
        std::format("workload offset {} is outside [0, {}] seconds", offset_s, duration_s()));
  const auto found = std::upper_bound(
      samples_.begin(), samples_.end(), offset_s,
      [](double offset, const Sample &sample) { return offset < sample.offset_s; });
  return std::prev(found)->target;
}

}
