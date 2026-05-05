#include "rnbt/scenario.hpp"

#include <algorithm>
#include <ranges>

namespace rnbt {

namespace {

template <class T>
[[nodiscard]] Expected<std::vector<T>> load_records(const std::filesystem::path &path) {
  try {
    std::vector<T> out;
    for (auto &&record : records<T>(path))
      out.push_back(std::move(record));
    return out;
  } catch (const std::exception &error) {
    return failure(std::format("{}: {}", path.string(), error.what()));
  }
}

template <Timestamped T> void sort_by_time(std::vector<T> &items) {
  std::ranges::stable_sort(items, std::less<>{}, &T::at);
}

[[nodiscard]] InputDigest digest(const std::filesystem::path &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  return InputDigest{.file = path.filename().string(), .bytes = ec ? 0 : size};
}

}

Expected<Scenario> Scenario::load(const ScenarioSource &source) {
  Scenario scenario;
  scenario.directory = source.directory;
  const auto at = [&](std::string_view name) { return source.directory / name; };

  auto manifest = read_json_file(at("manifest.json"));
  if (!manifest)
    return std::unexpected{manifest.error()};
  const auto window = [&](std::string_view key) -> Expected<ParsedTime> {
    if (!manifest->contains(key))
      return failure(std::format("manifest lacks {}", key));
    return parse_iso8601((*manifest)[std::string{key}].get<std::string>());
  };
  auto start = window("window_start");
  auto end = window("window_end");
  if (!start || !end)
    return std::unexpected{start ? end.error() : start.error()};
  scenario.window_start = start->instant;
  scenario.window_end = end->instant;
  scenario.calendar = Calendar{start->instant, start->offset};
  scenario.inputs.push_back(digest(at("manifest.json")));

  auto nodes_json = read_json_file(at("nodes.json"));
  if (!nodes_json)
    return std::unexpected{nodes_json.error()};
  auto nodes = try_decode<std::vector<NodeSpec>>(*nodes_json);
  if (!nodes)
    return failure(std::format("nodes.json: {}", nodes.error().message));
  scenario.nodes = std::move(*nodes);
  std::ranges::sort(scenario.nodes, std::less<>{}, &NodeSpec::id);
  scenario.inputs.push_back(digest(at("nodes.json")));

  scenario.topology_file = at("topology.json");
  if (!std::filesystem::exists(scenario.topology_file))
    return failure("scenario lacks topology.json");
  scenario.inputs.push_back(digest(scenario.topology_file));

  auto jobs = load_records<JobSpec>(at("jobs.jsonl"));
  if (!jobs)
    return std::unexpected{jobs.error()};
  scenario.jobs = std::move(*jobs);
  scenario.inputs.push_back(digest(at("jobs.jsonl")));

  auto reclaims = load_records<ReclaimSpec>(at("reclaims.jsonl"));
  if (!reclaims)
    return std::unexpected{reclaims.error()};
  for (auto &&[index, reclaim] : std::views::enumerate(*reclaims))
    if (reclaim.event_id.empty())
      reclaim.event_id = std::format("reclaim-{:04}", index + 1);
  scenario.reclaims = std::move(*reclaims);
  sort_by_time(scenario.reclaims);
  scenario.inputs.push_back(digest(at("reclaims.jsonl")));

  if (source.load_episodes && std::filesystem::exists(at("job_segments.jsonl"))) {
    auto segments = load_records<EpisodeSpec>(at("job_segments.jsonl"));
    if (!segments)
      return std::unexpected{segments.error()};
    scenario.lifecycle_segments = std::move(*segments);
    std::ranges::stable_sort(scenario.lifecycle_segments, std::less<>{}, &EpisodeSpec::start);
    std::ranges::copy_if(scenario.lifecycle_segments, std::back_inserter(scenario.episodes),
                         [](const EpisodeSpec &s) { return s.reason == "PRE_RECLAIM_RUN"; });
    std::ranges::stable_sort(scenario.episodes, std::less<>{}, &EpisodeSpec::start);
    scenario.inputs.push_back(digest(at("job_segments.jsonl")));
  }

  auto node_capacity = load_records<CapacitySample>(at("bandwidth_samples.jsonl"));
  if (!node_capacity)
    return std::unexpected{node_capacity.error()};
  scenario.node_capacity = std::move(*node_capacity);
  sort_by_time(scenario.node_capacity);
  scenario.inputs.push_back(digest(at("bandwidth_samples.jsonl")));

  if (std::filesystem::exists(at("core_bandwidth_samples.jsonl"))) {
    auto core = load_records<CapacitySample>(at("core_bandwidth_samples.jsonl"));
    if (!core)
      return std::unexpected{core.error()};
    scenario.core_capacity = std::move(*core);
    sort_by_time(scenario.core_capacity);
    scenario.inputs.push_back(digest(at("core_bandwidth_samples.jsonl")));
  }

  auto telemetry = load_records<TelemetrySample>(at("telemetry.jsonl"));
  if (!telemetry)
    return std::unexpected{telemetry.error()};
  scenario.telemetry = std::move(*telemetry);
  sort_by_time(scenario.telemetry);
  scenario.inputs.push_back(digest(at("telemetry.jsonl")));

  if (source.load_hazard_samples) {
    auto hazard = load_records<HazardSample>(at("hazard_samples.jsonl"));
    if (!hazard)
      return std::unexpected{hazard.error()};
    scenario.hazard = std::move(*hazard);
    sort_by_time(scenario.hazard);
    scenario.inputs.push_back(digest(at("hazard_samples.jsonl")));
  }

  const auto priors_path =
      source.migration_priors.value_or(at("historical_migration_profiles.jsonl"));
  if (std::filesystem::exists(priors_path)) {
    auto priors = load_records<MigrationPrior>(priors_path);
    if (!priors)
      return std::unexpected{priors.error()};
    for (const auto &prior : *priors)
      scenario.migration_priors_s.insert_or_assign(prior.node, prior.mean_migration_s);
    scenario.inputs.push_back(digest(priors_path));
  }

  scenario.index();
  for (const auto &job : scenario.jobs) {
    if (job.requested_runtime <= Nanos::zero())
      return failure(std::format("job {} has a non-positive requested runtime", job.id));
    if (job.initial_nodes.size() != job.gang_size)
      return failure(std::format("job {} places {} ranks for gang size {}", job.id,
                                 job.initial_nodes.size(), job.gang_size));
    for (const auto &node : job.initial_nodes)
      if (!scenario.find_node(node))
        return failure(std::format("job {} references unknown node {}", job.id, node));
    for (const auto &interval : job.busy_intervals) {
      if (interval.end <= interval.start || interval.nodes.size() != job.gang_size)
        return failure(
            std::format("job {} has an invalid busy interval {}", job.id, interval.segment_id));
      for (const auto &node : interval.nodes)
        if (!scenario.find_node(node))
          return failure(
              std::format("job {} busy interval references unknown node {}", job.id, node));
    }
  }
  for (const auto &reclaim : scenario.reclaims) {
    if (!scenario.find_node(reclaim.node))
      return failure(std::format("{} references unknown node {}", reclaim.event_id, reclaim.node));
    if (reclaim.notice < Nanos::zero())
      return failure(std::format("{} has a negative notice duration", reclaim.event_id));
  }
  for (const auto &episode : scenario.lifecycle_segments) {
    if (!scenario.find_reclaim(episode.event_id) || !scenario.job_index(episode.job))
      return failure(
          std::format("episode {} does not match a reclaim and a job", episode.event_id));
    const auto &job = scenario.jobs[*scenario.job_index(episode.job)];
    if (episode.end <= episode.start || episode.nodes.size() != job.gang_size)
      return failure(
          std::format("episode {} has an invalid interval or rank placement", episode.event_id));
    for (const auto &node : episode.nodes)
      if (!scenario.find_node(node))
        return failure(
            std::format("episode {} references unknown node {}", episode.event_id, node));
  }
  return scenario;
}

void Scenario::index() {
  const auto build = [](auto &target, const auto &items, auto projection) {
    target.clear();
    for (auto &&[position, item] : std::views::enumerate(items))
      target.emplace(std::invoke(projection, item), static_cast<std::size_t>(position));
  };
  build(node_index_, nodes, &NodeSpec::id);
  build(job_index_, jobs, &JobSpec::id);
  build(reclaim_index_, reclaims, &ReclaimSpec::event_id);
  build(episode_index_, episodes, &EpisodeSpec::event_id);
}

const NodeSpec *Scenario::find_node(std::string_view id) const noexcept {
  const auto found = node_index_.find(id);
  return found == node_index_.end() ? nullptr : &nodes[found->second];
}

std::optional<std::size_t> Scenario::job_index(std::string_view id) const noexcept {
  const auto found = job_index_.find(id);
  return found == job_index_.end() ? std::nullopt : std::optional<std::size_t>{found->second};
}

const ReclaimSpec *Scenario::find_reclaim(std::string_view event_id) const noexcept {
  const auto found = reclaim_index_.find(event_id);
  return found == reclaim_index_.end() ? nullptr : &reclaims[found->second];
}

const EpisodeSpec *Scenario::episode_of(std::string_view event_id) const noexcept {
  const auto found = episode_index_.find(event_id);
  return found == episode_index_.end() ? nullptr : &episodes[found->second];
}

}
