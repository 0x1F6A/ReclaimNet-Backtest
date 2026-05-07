#include "rnbt/plan.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <ranges>
#include <set>

namespace rnbt {

std::string_view name_of(const ActionBody &body) noexcept {
  return std::visit([]<class A>(const A &) { return action_name<A>::value; }, body);
}

unsigned action_order(const ActionBody &body) noexcept {
  return std::visit(
      overloaded{[](const ResetTestbed &) { return 0u; },
                 [](const SetLinkCapacity &) { return 1u; }, [](const Return &) { return 2u; },
                 [](const StopInstance &) { return 3u; }, [](const StartInstance &) { return 4u; },
                 [](const InjectReclaim &) { return 5u; }, [](const Depart &) { return 6u; },
                 [](const EndReplay &) { return 7u; }},
      body);
}

Json encode_action(const Action &action) {
  Json out = std::visit([](const auto &body) { return encode(body); }, action.body);
  out["type"] = std::string{name_of(action.body)};
  out["trace_ns"] = ns_of(action.trace_at);
  out["offset_s"] = seconds_of(action.offset);
  out["period"] = action.period;
  return out;
}

Json Plan::to_json(const Calendar &calendar) const {
  Json actions_json = Json::array();
  for (const auto &action : actions) {
    auto entry = encode_action(action);
    entry["trace_time"] = calendar.format(action.trace_at);
    actions_json.push_back(std::move(entry));
  }
  return {{"mode", std::string{rnbt::name_of(mode)}},
          {"from", calendar.format(from)},
          {"to", calendar.format(to)},
          {"origin", calendar.format(origin)},
          {"nodes", nodes},
          {"events", events},
          {"periods", encode(periods)},
          {"stats", encode(stats)},
          {"actions", std::move(actions_json)}};
}

Expected<Instant> parse_window_bound(std::string_view text, const Calendar &calendar) {
  if (text.starts_with("day:")) {
    double day{};
    const auto digits = text.substr(4);
    auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), day);
    if (ec != std::errc{} || end != digits.data() + digits.size() || !std::isfinite(day))
      return failure(std::format("invalid day bound '{}'", text));
    const long double ns = std::round(static_cast<long double>(ns_of(calendar.day_start(0))) +
                                      static_cast<long double>(day) * kNanosPerDay);
    if (ns < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        ns > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
      return failure(std::format("day bound '{}' is outside the supported timestamp range", text));
    return instant_of(static_cast<std::int64_t>(ns));
  }
  if (text.size() > 19 && text[19] == '.') {
    const auto fraction_end = text.find_first_not_of("0123456789", 20);
    const auto digits = (fraction_end == std::string_view::npos ? text.size() : fraction_end) - 20;
    if (digits == 0 || digits > 9)
      return failure(std::format("invalid fractional seconds in window bound '{}'", text));
  }
  auto parsed = parse_iso8601(text);
  if (!parsed)
    return std::unexpected{parsed.error()};
  const auto canonical = format_iso8601(parsed->instant, parsed->offset);
  const bool numeric_offset =
      text.size() >= 6 && (text[text.size() - 6] == '+' || text[text.size() - 6] == '-');
  if (text.substr(0, 10) != canonical.substr(0, 10) ||
      text.substr(11, 8) != canonical.substr(11, 8) ||
      (numeric_offset && parsed->offset != Nanos::zero() &&
       text.substr(text.size() - 6) != canonical.substr(canonical.size() - 6)) ||
      parsed->offset > std::chrono::hours{23} + std::chrono::minutes{59} ||
      parsed->offset < -std::chrono::hours{23} - std::chrono::minutes{59})
    return failure(std::format("invalid ISO-8601 window bound '{}'", text));
  return parsed->instant;
}

namespace {

struct Builder {
  const Scenario &scenario;
  const ReplayConfig &config;
  const Testbed &testbed;
  Plan plan;
  std::map<std::string, std::vector<std::pair<Instant, Instant>>> offline;

  [[nodiscard]] bool mapped(std::span<const std::string> nodes) const {
    return std::ranges::all_of(nodes,
                               [&](const std::string &n) { return testbed.node(n) != nullptr; });
  }

  void add(Instant at, ActionBody body) {
    plan.actions.push_back(
        Action{.trace_at = at, .offset = {}, .period = 0, .body = std::move(body)});
  }

  void reclaim(const ReclaimSpec &event) {
    if (std::ranges::contains(plan.events, event.event_id))
      return;
    add(event.at, InjectReclaim{.event_id = event.event_id,
                                .node = event.node,
                                .kind = event.kind,
                                .notice = event.notice});
    const Instant leave = event.kind == ReclaimKind::emergency ? event.at : event.deadline();
    if (leave < plan.to) {
      add(leave, Depart{.node = event.node, .event_id = event.event_id});
      offline[event.node].emplace_back(
          leave, saturating_add(leave, nanos_of(config.timing.node_return_s)));
    }
    plan.events.push_back(event.event_id);
    ++plan.stats.selected_events;
  }

  void schedule_returns(Instant horizon) {
    std::map<std::string, std::vector<Instant>> uses;
    for (const auto &action : plan.actions)
      std::visit(overloaded{[&](const StartInstance &start) {
                              for (const auto &node : start.nodes)
                                uses[node].push_back(action.trace_at);
                            },
                            [&](const InjectReclaim &injected) {
                              uses[injected.node].push_back(action.trace_at);
                            },
                            [](const auto &) {}},
                 action.body);
    for (auto &[node, stays] : offline) {
      std::ranges::sort(stays);
      auto &used = uses[node];
      std::ranges::sort(used);
      for (const auto &[leave, planned] : stays) {
        const auto next = std::ranges::upper_bound(used, leave);
        Instant back = planned;
        bool forced = false;
        if (next != used.end() && *next < back) {
          back = *next;
          forced = true;
        }
        if (back > horizon)
          continue;
        add(back, Return{.node = node, .forced = forced});
        plan.stats.forced_returns += forced ? 1 : 0;
      }
    }
  }

  void episodes() {
    std::map<std::string, std::vector<const EpisodeSpec *>> by_job;
    for (const auto &episode : scenario.episodes)
      by_job[episode.job].push_back(&episode);
    std::map<const EpisodeSpec *, double> progress_at_start;
    for (auto &[job, list] : by_job) {
      std::ranges::sort(list, std::less<>{}, &EpisodeSpec::start);
      const auto index = scenario.job_index(job);
      const double requested = index ? seconds_of(scenario.jobs[*index].requested_runtime) : 0.0;
      double sum = 0.0;
      for (const auto *episode : list) {
        progress_at_start[episode] = std::min(sum, requested);
        sum += seconds_of(episode->end - episode->start);
      }
    }
    std::vector<std::pair<Instant, Instant>> spans;
    for (const auto &episode : scenario.episodes) {
      const auto *event = scenario.find_reclaim(episode.event_id);
      if (!event || event->at < plan.from || event->at >= plan.to ||
          !config.events.admits(event->event_id, event->kind))
        continue;
      if (!mapped(episode.nodes) || !testbed.node(event->node)) {
        ++plan.stats.skipped_unmapped;
        continue;
      }
      const auto &job = scenario.jobs.at(*scenario.job_index(episode.job));
      const auto instance = std::format("{}@{}", job.id, episode.event_id);
      const Instant start = std::max(episode.start, plan.from);
      const double progress = progress_at_start[&episode] + seconds_of(start - episode.start);
      add(start, StartInstance{.instance = instance,
                               .job = job.id,
                               .nodes = episode.nodes,
                               .progress_s = std::min(progress, seconds_of(job.requested_runtime)),
                               .target_s = seconds_of(job.requested_runtime),
                               .park_after_resume = true,
                               .event_id = episode.event_id});
      reclaim(*event);
      const Instant cap =
          std::min(plan.to, saturating_add(event->at, nanos_of(config.timing.resolution_cap_s)));
      add(cap, StopInstance{.instance = instance});
      spans.emplace_back(start, cap);
      ++plan.stats.instances;
    }
    std::ranges::sort(spans);
    const Nanos budget = nanos_of(config.timing.reset_budget_s);
    for (const auto &[start, end] : spans) {
      const bool split = !plan.periods.empty() && config.reset_between_periods &&
                         start - plan.periods.back().end > budget;
      if (plan.periods.empty() || split) {
        plan.periods.push_back(PeriodInfo{.index = static_cast<std::uint32_t>(plan.periods.size()),
                                          .start = start,
                                          .end = end,
                                          .events = {}});
      } else {
        plan.periods.back().end = std::max(plan.periods.back().end, end);
      }
    }
  }

  void jobs() {
    const auto progress_at = [&](const JobSpec &job, Instant at) {
      double seconds = 0.0;
      if (!job.busy_intervals.empty()) {
        for (const auto &interval : job.busy_intervals)
          if (at > interval.start)
            seconds += seconds_of(std::min(at, interval.end) - interval.start);
      } else {
        bool recorded = false;
        for (const auto &segment : scenario.lifecycle_segments)
          if (segment.job == job.id) {
            recorded = true;
            if (at > segment.start)
              seconds += seconds_of(std::min(at, segment.end) - segment.start);
          }
        if (!recorded && at > job.submit)
          seconds = seconds_of(at - job.submit);
      }
      return seconds;
    };
    const auto launch = [&](const JobSpec &job, std::string instance, Instant original_start,
                            Instant original_end, const std::vector<std::string> &nodes,
                            std::optional<std::string> event_id = std::nullopt) {
      const Instant start = std::max(original_start, plan.from);
      const Instant end = std::min(original_end, plan.to);
      if (start >= end)
        return;
      if (!mapped(nodes)) {
        ++plan.stats.skipped_unmapped;
        return;
      }
      const double progress = progress_at(job, start);
      const double target =
          std::max(seconds_of(job.requested_runtime), progress + seconds_of(end - start) + 1.0);
      add(start, StartInstance{.instance = instance,
                               .job = job.id,
                               .nodes = nodes,
                               .progress_s = progress,
                               .target_s = target,
                               .park_after_resume = false,
                               .event_id = std::move(event_id)});
      add(end, StopInstance{.instance = instance});
      if (end < plan.to)
        add(plan.to, StopInstance{.instance = instance});
      ++plan.stats.instances;
    };
    for (const auto &job : scenario.jobs) {
      std::vector<const EpisodeSpec *> segments;
      for (const auto &segment : scenario.lifecycle_segments)
        if (segment.job == job.id)
          segments.push_back(&segment);
      if (!segments.empty()) {
        std::set<const EpisodeSpec *> grouped;
        for (const auto *pre : segments) {
          if (pre->reason != "PRE_RECLAIM_RUN")
            continue;
          const auto *event = scenario.find_reclaim(pre->event_id);
          if (!event)
            continue;
          grouped.insert(pre);
          const EpisodeSpec *post = nullptr;
          for (const auto *candidate : segments)
            if (candidate->event_id == pre->event_id && candidate != pre &&
                (candidate->reason == "POST_MIGRATION_LIFECYCLE" ||
                 candidate->reason == "FALLBACK_REQUEUE_LIFECYCLE")) {
              grouped.insert(candidate);
              if (!post || candidate->end > post->end)
                post = candidate;
            }
          const Instant end =
              post ? post->end
                   : saturating_add(event->at, nanos_of(config.timing.resolution_cap_s));
          if (event->at < plan.from) {
            if (post)
              launch(job, std::format("{}@{}", job.id, event->event_id), post->start, end,
                     post->nodes);
          } else {
            launch(job, std::format("{}@{}", job.id, event->event_id), pre->start, end, pre->nodes,
                   event->event_id);
          }
        }
        for (const auto *segment : segments)
          if (!grouped.contains(segment))
            launch(
                job,
                std::format("{}@{}", job.id,
                            segment->segment_id.empty() ? segment->event_id : segment->segment_id),
                segment->start, segment->end, segment->nodes);
        continue;
      }
      if (!job.busy_intervals.empty()) {
        for (std::size_t index = 0; index < job.busy_intervals.size(); ++index) {
          const auto &interval = job.busy_intervals[index];
          const auto suffix =
              interval.segment_id.empty() ? std::format("segment-{}", index) : interval.segment_id;
          launch(job, std::format("{}@{}", job.id, suffix), interval.start, interval.end,
                 interval.nodes);
        }
        continue;
      }
      launch(job, job.id, job.submit, saturating_add(job.submit, job.requested_runtime),
             job.initial_nodes);
    }
    for (const auto &event : scenario.reclaims) {
      if (event.at < plan.from || event.at >= plan.to ||
          !config.events.admits(event.event_id, event.kind))
        continue;
      if (!testbed.node(event.node)) {
        ++plan.stats.skipped_unmapped;
        continue;
      }
      reclaim(event);
    }
    if (!plan.actions.empty()) {
      plan.periods.push_back(
          PeriodInfo{.index = 0, .start = plan.from, .end = plan.to, .events = {}});
    }
  }

  [[nodiscard]] Status physical_profiles() {
    if (!config.workload_profiles)
      return {};
    plan.periods = {PeriodInfo{.index = 0, .start = plan.from, .end = plan.to, .events = {}}};
    add(plan.to, EndReplay{});
    if (!config.replay_link_capacity)
      return {};
    for (const auto &host : testbed.nodes) {
      const CapacitySample *initial = nullptr;
      for (const auto &sample : scenario.node_capacity)
        if (sample.node == host.trace_node && sample.at <= plan.from &&
            (!initial || sample.at >= initial->at))
          initial = &sample;
      if (!initial)
        for (const auto &sample : scenario.node_capacity)
          if (sample.node == host.trace_node && sample.at >= plan.from && sample.at < plan.to &&
              (!initial || sample.at < initial->at))
            initial = &sample;
      if (!initial)
        return failure(std::format("node {} has no link-capacity sample before the window end",
                                   host.trace_node));
      if (!std::isfinite(initial->bps) || initial->bps < 0.0)
        return failure(
            std::format("node {} has an invalid initial link capacity", host.trace_node));
      add(plan.from, SetLinkCapacity{.node = host.trace_node, .bps = initial->bps});
      std::map<Instant, double> changes;
      for (const auto &sample : scenario.node_capacity)
        if (sample.node == host.trace_node && sample.at > plan.from && sample.at < plan.to) {
          if (!std::isfinite(sample.bps) || sample.bps < 0.0)
            return failure(
                std::format("node {} has an invalid link-capacity sample", host.trace_node));
          changes.insert_or_assign(sample.at, sample.bps);
        }
      double current_bps = initial->bps;
      for (const auto &[at, bps] : changes)
        if (bps != current_bps) {
          add(at, SetLinkCapacity{.node = host.trace_node, .bps = bps});
          current_bps = bps;
        }
    }
    return {};
  }

  void assign_periods() {
    const Nanos budget = nanos_of(config.timing.reset_budget_s);
    const auto period_of = [&](Instant at) -> std::optional<std::uint32_t> {
      for (const auto &period : plan.periods)
        if (at >= period.start - budget && at <= period.end)
          return period.index;
      return std::nullopt;
    };
    std::erase_if(plan.actions, [&](Action &action) {
      const auto period = period_of(action.trace_at);
      if (!period && std::holds_alternative<Return>(action.body) && config.reset_between_periods)
        return true;
      action.period = period.value_or(static_cast<std::uint32_t>(plan.periods.size() - 1));
      return false;
    });
    for (auto &period : plan.periods)
      for (const auto &action : plan.actions)
        if (action.period == period.index)
          if (const auto *injected = std::get_if<InjectReclaim>(&action.body))
            period.events.push_back(injected->event_id);
    for (const auto &period : plan.periods)
      if (period.index == 0 || config.reset_between_periods)
        plan.actions.push_back(
            Action{.trace_at = (period.index == 0 ? plan.from : period.start) - budget,
                   .offset = {},
                   .period = period.index,
                   .body = ResetTestbed{.period = period.index}});
  }

  void assign_offsets() {
    std::ranges::sort(plan.actions, std::less<>{},
                      [](const Action &a) { return std::tuple{a.trace_at, action_order(a.body)}; });
    if (plan.actions.empty())
      return;
    plan.origin = plan.from - nanos_of(config.timing.reset_budget_s);
    Nanos removed{};
    const Nanos threshold = nanos_of(config.timing.idle_threshold_s);
    const Nanos keep = nanos_of(config.timing.reset_budget_s);
    for (std::size_t i = 0; i < plan.actions.size(); ++i) {
      auto &action = plan.actions[i];
      if (config.timing.compress_idle && i > 0 &&
          std::holds_alternative<ResetTestbed>(action.body)) {
        const Nanos gap = action.trace_at - plan.actions[i - 1].trace_at;
        if (gap > threshold)
          removed += gap - keep;
      }
      action.offset = action.trace_at - plan.origin - removed;
    }
    std::ranges::stable_sort(plan.actions, std::less<>{}, &Action::key);
    plan.stats.trace_span_s = seconds_of(plan.actions.back().trace_at - plan.origin);
    plan.stats.wall_span_s = seconds_of(plan.actions.back().offset);
  }
};

}

Expected<Plan> build_plan(const Scenario &scenario, const ReplayConfig &config,
                          const Testbed &testbed) {
  Builder builder{
      .scenario = scenario, .config = config, .testbed = testbed, .plan = {}, .offline = {}};
  auto &plan = builder.plan;
  for (const double seconds : {config.timing.reset_budget_s, config.timing.resolution_cap_s,
                               config.timing.node_return_s, config.timing.idle_threshold_s})
    if (!std::isfinite(seconds) || seconds < 0.0)
      return failure("replay planning timing values must be finite and non-negative");
  if (config.workload_profiles && config.timing.compress_idle)
    return failure("physical workload profiles require uncompressed trace time");
  plan.mode = config.mode;
  plan.from = scenario.window_start;
  plan.to = scenario.window_end;
  if (config.window.from) {
    auto from = parse_window_bound(*config.window.from, scenario.calendar);
    if (!from)
      return std::unexpected{from.error()};
    plan.from = *from;
  }
  if (config.window.to) {
    auto to = parse_window_bound(*config.window.to, scenario.calendar);
    if (!to)
      return std::unexpected{to.error()};
    plan.to = *to;
  }
  if (plan.to <= plan.from)
    return failure("the replay window is empty");
  if (plan.from < scenario.window_start || plan.to > scenario.window_end)
    return failure(std::format("replay window must lie within trace range [{}, {})",
                               scenario.calendar.format(scenario.window_start),
                               scenario.calendar.format(scenario.window_end)));
  if (config.mode == ReplayMode::episodes) {
    if (scenario.episodes.empty() && !config.workload_profiles)
      return failure("episode replay needs job_segments.jsonl in the trace");
    builder.episodes();
  } else {
    builder.jobs();
  }
  if (auto status = builder.physical_profiles(); !status)
    return std::unexpected{status.error()};
  if (plan.actions.empty())
    return failure("no trace event or job falls inside the selected window on mapped nodes");
  const auto horizon = std::ranges::max(plan.actions | std::views::transform(&Action::trace_at));
  builder.schedule_returns(horizon);
  builder.assign_periods();
  builder.assign_offsets();
  std::set<std::string> nodes;
  if (config.workload_profiles)
    for (const auto &host : testbed.nodes)
      nodes.insert(host.trace_node);
  for (const auto &action : plan.actions)
    std::visit(
        overloaded{[&](const StartInstance &s) { nodes.insert(s.nodes.begin(), s.nodes.end()); },
                   [&](const InjectReclaim &r) { nodes.insert(r.node); },
                   [&](const Depart &d) { nodes.insert(d.node); },
                   [&](const Return &r) { nodes.insert(r.node); }, [](const auto &) {}},
        action.body);
  plan.nodes.assign(nodes.begin(), nodes.end());
  return std::move(builder.plan);
}

}
