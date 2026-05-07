#include "rnbt/replay_config.hpp"

#include "rnbt/jsonl.hpp"

#include <cctype>
#include <set>

namespace rnbt {

namespace {

[[nodiscard]] Status apply_override(Json &document, std::string_view assignment) {
  const auto equals = assignment.find('=');
  if (equals == std::string_view::npos || equals == 0 || assignment.front() != '/')
    return failure(std::format("override '{}' must look like /path/to/key=value", assignment));
  Json value = Json::parse(assignment.substr(equals + 1), nullptr, false);
  if (value.is_discarded())
    value = std::string{assignment.substr(equals + 1)};
  try {
    document[Json::json_pointer{std::string{assignment.substr(0, equals)}}] = std::move(value);
  } catch (const Json::exception &error) {
    return failure(std::format("override '{}': {}", assignment, error.what()));
  }
  return {};
}

[[nodiscard]] std::string resolve(const std::filesystem::path &base, std::string_view path) {
  if (path.empty())
    return {};
  const std::filesystem::path candidate{path};
  return (candidate.is_absolute() ? candidate : base / candidate).lexically_normal().string();
}

[[nodiscard]] bool has_gpu_emulation_marker(std::string_view text) {
  std::string lower{text};
  std::ranges::transform(lower, lower.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower.contains("rnbt_fake_") || lower.contains("fake_nvml") || lower.contains("fake-nvml");
}

[[nodiscard]] Status require_physical_gpu_config(const Json &value, std::string path) {
  if (value.is_string() && has_gpu_emulation_marker(value.get_ref<const std::string &>()))
    return failure(
        std::format("{} contains a GPU emulation setting; replay requires physical GPUs", path));
  if (value.is_object()) {
    for (const auto &[key, child] : value.items()) {
      const auto child_path = std::format("{}/{}", path, key);
      if (has_gpu_emulation_marker(key))
        return failure(std::format("{} is a GPU emulation setting; replay requires physical GPUs",
                                   child_path));
      if (auto valid = require_physical_gpu_config(child, child_path); !valid)
        return valid;
    }
  } else if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index)
      if (auto valid = require_physical_gpu_config(value[index], std::format("{}/{}", path, index));
          !valid)
        return valid;
  }
  return {};
}

[[nodiscard]] Status validate(const ReplayConfig &config, const Testbed &testbed) {
  if (config.trace.empty())
    return failure("trace is required");
  if (config.workload_profiles && config.workload_profiles->empty())
    return failure("workload_profiles must be a non-empty directory path");
  if (config.workload_profiles && config.timing.compress_idle)
    return failure("workload profile replay requires compress_idle=false");
  if (config.replay_link_capacity && !config.workload_profiles)
    return failure("replay_link_capacity requires workload_profiles");
  if (config.workload_profiles &&
      (config.run_id.empty() || config.run_id == "." || config.run_id == ".." ||
       !std::ranges::all_of(config.run_id, [](unsigned char c) {
         return std::isalnum(c) || c == '_' || c == '-' || c == '.';
       })))
    return failure(
        "profile replay run_id must be a file name containing only letters, digits, dot, dash or underscore (not . or ..)");
  if (testbed.nodes.empty())
    return failure("the testbed lists no nodes");
  std::set<std::string_view> seen;
  for (const auto &node : testbed.nodes) {
    if (node.trace_node.empty() || node.node_daemon_address.empty())
      return failure("every testbed node needs trace_node and node_daemon_address");
    if (!seen.insert(node.trace_node).second)
      return failure(std::format("trace node {} is mapped twice", node.trace_node));
    if (!testbed.local && node.ssh.empty())
      return failure(
          std::format("{} needs an ssh target unless the testbed is local", node.trace_node));
    if (config.replay_link_capacity &&
        (node.network_interface.empty() ||
         !std::ranges::all_of(node.network_interface, [](unsigned char c) {
           return std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':';
         })))
      return failure(std::format("{} needs a valid dedicated network_interface for link replay",
                                 node.trace_node));
  }
  const auto &t = config.timing;
  for (const double value : {t.node_return_s, t.resolution_cap_s, t.drain_s, t.health_timeout_s})
    if (!(value >= 0.0))
      return failure("timing values must be non-negative");
  if (testbed.parallelism == 0)
    return failure("testbed parallelism must be positive");
  return {};
}

}

Expected<LoadedReplay> load_replay(const std::filesystem::path &config,
                                   std::span<const std::string> overrides) {
  auto document = read_json_file(config);
  if (!document)
    return std::unexpected{document.error()};
  for (const auto &assignment : overrides)
    if (auto applied = apply_override(*document, assignment); !applied)
      return std::unexpected{applied.error()};
  auto replay = try_decode<ReplayConfig>(*document);
  if (!replay)
    return failure(std::format("{}: {}", config.string(), replay.error().message));
  const auto base = std::filesystem::absolute(config).parent_path();
  replay->trace = resolve(base, replay->trace);
  if (replay->workload_profiles)
    replay->workload_profiles = resolve(base, *replay->workload_profiles);
  replay->testbed = resolve(base, replay->testbed);
  replay->output = resolve(base, replay->output);
  if (replay->migration_priors)
    replay->migration_priors = resolve(base, *replay->migration_priors);
  auto testbed_json = read_json_file(replay->testbed);
  if (!testbed_json)
    return std::unexpected{testbed_json.error()};
  if (testbed_json->value("_template", false))
    return failure("fill the physical testbed template and remove _template before replaying");
  if (auto valid = require_physical_gpu_config(*testbed_json, "testbed"); !valid)
    return std::unexpected{valid.error()};
  auto testbed = try_decode<Testbed>(*testbed_json);
  if (!testbed)
    return failure(std::format("{}: {}", replay->testbed, testbed.error().message));
  const auto testbed_base = std::filesystem::path{replay->testbed}.parent_path();
  for (auto *path : {&testbed->credentials.ca, &testbed->credentials.cert,
                     &testbed->credentials.key, &testbed->credentials.jwt_private_key})
    *path = resolve(testbed_base, *path);
  if (auto valid = validate(*replay, *testbed); !valid)
    return std::unexpected{valid.error()};
  if (replay->workload_profiles) {
    const auto suffix = "/runs/" + replay->run_id;
    testbed->paths.log_dir += suffix;
    testbed->paths.workload_root += suffix;
    for (auto &node : testbed->nodes) {
      if (auto custom = node.vars.find("log_dir"); custom != node.vars.end())
        custom->second += suffix;
      if (auto custom = node.vars.find("workload_root"); custom != node.vars.end())
        custom->second += suffix;
    }
  }
  return LoadedReplay{.config = std::move(*replay),
                      .testbed = std::move(*testbed),
                      .raw_config = std::move(*document),
                      .raw_testbed = std::move(*testbed_json),
                      .config_directory = base};
}

}
