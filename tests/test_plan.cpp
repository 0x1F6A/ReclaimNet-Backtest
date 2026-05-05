#include "rnbt/plan.hpp"

#include <gtest/gtest.h>

using namespace rnbt;

namespace {

Scenario mini() {
  auto scenario =
      Scenario::load(ScenarioSource{.directory = std::string{RNBT_TEST_DATA} + "/mini_trace"});
  EXPECT_TRUE(scenario) << (scenario ? "" : scenario.error().describe());
  return std::move(*scenario);
}

Testbed testbed(std::initializer_list<std::string_view> nodes) {
  Testbed out;
  out.local = true;
  out.coordinator =
      CoordinatorHost{.host = "127.0.0.1", .ssh = "", .address = "127.0.0.1:50051", .vars = {}};
  std::uint16_t port = 50061;
  for (const auto node : nodes)
    out.nodes.push_back(HostConfig{.trace_node = std::string{node},
                                   .host = "127.0.0.1",
                                   .ssh = "",
                                   .node_daemon_address = std::format("127.0.0.1:{}", port++),
                                   .migration_port = 7600,
                                   .vars = {}});
  return out;
}

template <class A> std::vector<const Action *> of(const Plan &plan) {
  std::vector<const Action *> out;
  for (const auto &action : plan.actions)
    if (std::holds_alternative<A>(action.body))
      out.push_back(&action);
  return out;
}

}

TEST(Plan, EpisodesFollowTheScriptAndTheNotice) {
  const auto scenario = mini();
  ReplayConfig config;
  config.timing.node_return_s = 20.0;
  config.timing.resolution_cap_s = 120.0;
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  EXPECT_EQ(plan->stats.selected_events, 2u);
  EXPECT_EQ(plan->stats.instances, 2u);
  ASSERT_FALSE(plan->actions.empty());
  EXPECT_TRUE(std::holds_alternative<ResetTestbed>(plan->actions.front().body));
  const auto starts = of<StartInstance>(*plan);
  ASSERT_EQ(starts.size(), 2u);
  EXPECT_EQ(std::get<StartInstance>(starts[0]->body).instance, "job-a@reclaim-0001");
  EXPECT_TRUE(std::get<StartInstance>(starts[0]->body).park_after_resume);
  const auto departs = of<Depart>(*plan);
  ASSERT_EQ(departs.size(), 2u);
  const auto *scheduled = scenario.find_reclaim("reclaim-0001");
  EXPECT_EQ(departs[0]->trace_at, scheduled->deadline());
  const auto *emergency = scenario.find_reclaim("reclaim-0002");
  EXPECT_EQ(departs[1]->trace_at, emergency->at);
  for (std::size_t i = 1; i < plan->actions.size(); ++i)
    EXPECT_LE(plan->actions[i - 1].offset, plan->actions[i].offset);
  EXPECT_EQ(of<ResetTestbed>(*plan).size(), 1u);
}

TEST(Plan, WindowAndKindFiltersSelectEvents) {
  const auto scenario = mini();
  ReplayConfig config;
  config.window.from = "2026-03-01T00:00:00+08:00";
  config.window.to = "2026-03-01T00:02:00+08:00";
  auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->events, std::vector<std::string>{"reclaim-0001"});
  config.window = {};
  config.events.kinds = {ReclaimKind::emergency};
  plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->events, std::vector<std::string>{"reclaim-0002"});
}

TEST(Plan, UnmappedNodesAreSkipped) {
  const auto scenario = mini();
  ReplayConfig config;
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-003"}));
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->stats.skipped_unmapped, 1u);
  EXPECT_EQ(plan->events, std::vector<std::string>{"reclaim-0001"});
}

TEST(Plan, JobsModeStartsAtSubmitAndInjectsNodeEvents) {
  const auto scenario = mini();
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan);
  const auto starts = of<StartInstance>(*plan);
  ASSERT_EQ(starts.size(), 2u);
  EXPECT_EQ(std::get<StartInstance>(starts[0]->body).instance, "job-a@reclaim-0001");
  EXPECT_FALSE(std::get<StartInstance>(starts[0]->body).park_after_resume);
  EXPECT_EQ(of<InjectReclaim>(*plan).size(), 2u);
}

TEST(Plan, DayBoundsUseTheTraceCalendar) {
  const auto scenario = mini();
  const auto bound = parse_window_bound("day:1.5", scenario.calendar);
  ASSERT_TRUE(bound);
  EXPECT_EQ(*bound, scenario.calendar.day_start(1) + std::chrono::hours{12});
  EXPECT_FALSE(parse_window_bound("day:x", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("day:nan", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("day:inf", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("day:1000000000", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("2026-02-30T00:00:00+08:00", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("2026-03-01T25:00:00+08:00", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("2026-03-01T00:00:00+00:99", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("2026-03-01T00:00:00.Z", scenario.calendar));
  EXPECT_FALSE(parse_window_bound("2026-03-01T00:00:00.1234567891Z", scenario.calendar));
}

TEST(Plan, JobsCarryIntoTheWindowAndStopAtItsExclusiveEnd) {
  const auto scenario = mini();
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.window.from = "2026-03-01T00:00:50+08:00";
  config.window.to = "2026-03-01T00:01:00+08:00";
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  const auto starts = of<StartInstance>(*plan);
  ASSERT_EQ(starts.size(), 2u);
  EXPECT_EQ(starts[0]->trace_at, plan->from);
  EXPECT_EQ(starts[1]->trace_at, plan->from);
  EXPECT_EQ(std::get<StartInstance>(starts[0]->body).progress_s, 40.0);
  EXPECT_EQ(std::get<StartInstance>(starts[1]->body).progress_s, 30.0);
  EXPECT_EQ(plan->origin, plan->from - nanos_of(config.timing.reset_budget_s));
  EXPECT_EQ(plan->actions.front().offset, Nanos::zero());
  EXPECT_EQ(starts[0]->offset, nanos_of(config.timing.reset_budget_s));
  const auto stops = of<StopInstance>(*plan);
  ASSERT_EQ(stops.size(), 2u);
  for (const auto *stop : stops)
    EXPECT_EQ(stop->trace_at, plan->to);
  EXPECT_TRUE(of<InjectReclaim>(*plan).empty());
  for (const auto &action : plan->actions) {
    if (!std::holds_alternative<ResetTestbed>(action.body))
      EXPECT_GE(action.trace_at, plan->from);
    EXPECT_LE(action.trace_at, plan->to);
    EXPECT_GE(action.offset, Nanos::zero());
  }
}

TEST(Plan, EpisodesClipBothEndsAndDoNotDepartAfterWindowEnd) {
  const auto scenario = mini();
  ReplayConfig config;
  config.window.from = "2026-03-01T00:01:00+08:00";
  config.window.to = "2026-03-01T00:01:30+08:00";
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  EXPECT_EQ(plan->events, std::vector<std::string>{"reclaim-0001"});
  const auto starts = of<StartInstance>(*plan);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0]->trace_at, plan->from);
  EXPECT_EQ(std::get<StartInstance>(starts[0]->body).progress_s, 50.0);
  const auto stops = of<StopInstance>(*plan);
  ASSERT_EQ(stops.size(), 1u);
  EXPECT_EQ(stops[0]->trace_at, plan->to);
  EXPECT_TRUE(of<Depart>(*plan).empty());
  EXPECT_TRUE(of<Return>(*plan).empty());
  for (const auto &action : plan->actions) {
    if (!std::holds_alternative<ResetTestbed>(action.body))
      EXPECT_GE(action.trace_at, plan->from);
    EXPECT_LE(action.trace_at, plan->to);
  }
}

TEST(Plan, EmergencyNotifiesBeforeRemovingTheNodeAtWindowStart) {
  const auto scenario = mini();
  ReplayConfig config;
  config.window.from = "2026-03-01T00:02:30+08:00";
  config.window.to = "2026-03-01T00:03:00+08:00";
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  EXPECT_EQ(plan->events, std::vector<std::string>{"reclaim-0002"});
  std::vector<std::string_view> at_start;
  for (const auto &action : plan->actions)
    if (action.trace_at == plan->from)
      at_start.push_back(name_of(action.body));
  EXPECT_EQ(at_start, (std::vector<std::string_view>{"start", "reclaim", "depart"}));
}

TEST(Plan, RejectsBoundsOutsideTraceAndEmptySelections) {
  const auto scenario = mini();
  ReplayConfig config;
  const auto hosts = testbed({"node-001", "node-002", "node-003"});
  config.window.from = "day:-1";
  EXPECT_FALSE(build_plan(scenario, config, hosts));
  config.window.from = "day:0";
  config.window.to = "day:2";
  EXPECT_FALSE(build_plan(scenario, config, hosts));
  config.window.to = "day:0";
  EXPECT_FALSE(build_plan(scenario, config, hosts));
  config.window.from = "2026-03-01T01:00:00+08:00";
  config.window.to = "2026-03-01T02:00:00+08:00";
  EXPECT_FALSE(build_plan(scenario, config, hosts));
  config.mode = ReplayMode::jobs;
  config.window.from = "2026-03-01T02:00:00+08:00";
  config.window.to = "2026-03-01T03:00:00+08:00";
  EXPECT_FALSE(build_plan(scenario, config, hosts));
}

TEST(Plan, LifecycleKeepsTheTaskAliveThroughARealReclaim) {
  auto scenario = mini();
  const auto *event = scenario.find_reclaim("reclaim-0001");
  ASSERT_NE(event, nullptr);
  scenario.lifecycle_segments.push_back(EpisodeSpec{.event_id = event->event_id,
                                                    .job = "job-a",
                                                    .nodes = {"node-003"},
                                                    .start = event->at + nanos_of(45),
                                                    .end = event->at + nanos_of(100),
                                                    .reason = "POST_MIGRATION_LIFECYCLE",
                                                    .segment_id = "post-a"});
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.window.to = "2026-03-01T00:03:00+08:00";
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  std::size_t starts_a = 0;
  std::vector<Instant> stops_a;
  for (const auto &action : plan->actions) {
    if (const auto *start = std::get_if<StartInstance>(&action.body);
        start && start->job == "job-a") {
      ++starts_a;
      EXPECT_EQ(start->nodes, std::vector<std::string>{"node-001"});
    }
    if (const auto *stop = std::get_if<StopInstance>(&action.body);
        stop && stop->instance == "job-a@reclaim-0001")
      stops_a.push_back(action.trace_at);
  }
  EXPECT_EQ(starts_a, 1u);
  ASSERT_EQ(stops_a.size(), 2u);
  EXPECT_EQ(stops_a[0], event->at + nanos_of(100));
  EXPECT_EQ(stops_a[1], plan->to);
  EXPECT_GT(stops_a[0], event->deadline());
}

TEST(Plan, LifecycleCarriesInFromTheRecordedMigrationDestination) {
  auto scenario = mini();
  const auto *event = scenario.find_reclaim("reclaim-0001");
  ASSERT_NE(event, nullptr);
  scenario.lifecycle_segments.push_back(EpisodeSpec{.event_id = event->event_id,
                                                    .job = "job-a",
                                                    .nodes = {"node-003"},
                                                    .start = event->at + nanos_of(45),
                                                    .end = event->at + nanos_of(100),
                                                    .reason = "POST_MIGRATION_LIFECYCLE",
                                                    .segment_id = "post-a"});
  auto &job = scenario.jobs[0];
  job.busy_intervals = {
      BusyInterval{
          .segment_id = "pre-a", .start = job.submit, .end = event->at, .nodes = {"node-001"}},
      BusyInterval{.segment_id = "post-a",
                   .start = event->at + nanos_of(45),
                   .end = event->at + nanos_of(100),
                   .nodes = {"node-003"}}};
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.window.from = "2026-03-01T00:02:00+08:00";
  config.window.to = "2026-03-01T00:02:20+08:00";
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  const auto starts = of<StartInstance>(*plan);
  ASSERT_EQ(starts.size(), 2u);
  const auto found = std::ranges::find_if(starts, [](const Action *action) {
    return std::get<StartInstance>(action->body).job == "job-a";
  });
  ASSERT_NE(found, starts.end());
  EXPECT_EQ((*found)->trace_at, plan->from);
  const auto &start = std::get<StartInstance>((*found)->body);
  EXPECT_EQ(start.nodes, std::vector<std::string>{"node-003"});
  EXPECT_DOUBLE_EQ(start.progress_s, 65.0);
  EXPECT_FALSE(start.event_id.has_value());
  EXPECT_TRUE(of<InjectReclaim>(*plan).empty());
  EXPECT_TRUE(of<Depart>(*plan).empty());
}

TEST(Plan, BusyIntervalsReplayLaterActivityInsteadOfTheSubmissionTime) {
  auto scenario = mini();
  scenario.lifecycle_segments.clear();
  scenario.episodes.clear();
  auto &job = scenario.jobs[0];
  const Instant later = scenario.window_start + std::chrono::hours{8};
  job.busy_intervals = {BusyInterval{.segment_id = "early",
                                     .start = job.submit,
                                     .end = job.submit + nanos_of(20),
                                     .nodes = {"node-001"}},
                        BusyInterval{.segment_id = "later",
                                     .start = later,
                                     .end = later + nanos_of(60),
                                     .nodes = {"node-003"}}};
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.window.from = "2026-03-01T08:00:10+08:00";
  config.window.to = "2026-03-01T08:00:20+08:00";
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  const auto starts = of<StartInstance>(*plan);
  ASSERT_EQ(starts.size(), 1u);
  const auto &start = std::get<StartInstance>(starts[0]->body);
  EXPECT_EQ(start.instance, "job-a@later");
  EXPECT_EQ(start.nodes, std::vector<std::string>{"node-003"});
  EXPECT_DOUBLE_EQ(start.progress_s, 30.0);
  EXPECT_EQ(starts[0]->trace_at, plan->from);
  EXPECT_EQ(of<StopInstance>(*plan).front()->trace_at, plan->to);
}

TEST(Plan, PhysicalBackgroundReplaysTheWholeWindowOnAllMappedNodes) {
  auto scenario = mini();
  scenario.jobs.clear();
  scenario.reclaims.clear();
  scenario.episodes.clear();
  scenario.lifecycle_segments.clear();
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.workload_profiles = "profiles";
  config.replay_link_capacity = true;
  config.window.from = "2026-03-01T00:01:00+08:00";
  config.window.to = "2026-03-01T00:02:00+08:00";
  const Instant from = scenario.window_start + nanos_of(60);
  const Instant to = scenario.window_start + nanos_of(120);
  scenario.node_capacity = {
      CapacitySample{.node = "node-001", .bps = 1e9, .at = from - nanos_of(1)},
      CapacitySample{.node = "node-002", .bps = 2e9, .at = from},
      CapacitySample{.node = "node-003", .bps = 3e9, .at = from + nanos_of(1)},
      CapacitySample{.node = "node-001", .bps = 5e9, .at = from + nanos_of(30)},
      CapacitySample{.node = "node-001", .bps = 6e9, .at = from + nanos_of(30)},
      CapacitySample{.node = "node-001", .bps = 7e9, .at = to}};
  const auto plan = build_plan(scenario, config, testbed({"node-001", "node-002", "node-003"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  EXPECT_EQ(plan->nodes, (std::vector<std::string>{"node-001", "node-002", "node-003"}));
  ASSERT_EQ(plan->periods.size(), 1u);
  EXPECT_EQ(plan->periods[0].start, from);
  EXPECT_EQ(plan->periods[0].end, to);
  const auto capacity = of<SetLinkCapacity>(*plan);
  ASSERT_EQ(capacity.size(), 4u);
  EXPECT_EQ(capacity[0]->trace_at, from);
  EXPECT_EQ(capacity[1]->trace_at, from);
  EXPECT_EQ(capacity[2]->trace_at, from);
  EXPECT_EQ(capacity[3]->trace_at, from + nanos_of(30));
  EXPECT_EQ(std::get<SetLinkCapacity>(capacity[3]->body).bps, 6e9);
  EXPECT_TRUE(of<StartInstance>(*plan).empty());
  const auto ends = of<EndReplay>(*plan);
  ASSERT_EQ(ends.size(), 1u);
  EXPECT_EQ(ends.front()->trace_at, to);
  EXPECT_EQ(ends.front()->offset, nanos_of(60 + config.timing.reset_budget_s));
  EXPECT_EQ(name_of(plan->actions.back().body), "end");
}

TEST(Plan, PhysicalCapacityDoesNotBorrowSamplesAfterTheWindow) {
  auto scenario = mini();
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.workload_profiles = "profiles";
  config.replay_link_capacity = true;
  config.window.to = "2026-03-01T00:01:00+08:00";
  scenario.node_capacity = {
      CapacitySample{.node = "node-001", .bps = 1e9, .at = scenario.window_start + nanos_of(60)}};
  EXPECT_FALSE(build_plan(scenario, config, testbed({"node-001"})));
}

TEST(Plan, ZeroLinkCapacityPreservesACompleteNetworkOutage) {
  auto scenario = mini();
  ReplayConfig config;
  config.mode = ReplayMode::jobs;
  config.workload_profiles = "profiles";
  config.replay_link_capacity = true;
  config.window.to = "2026-03-01T00:01:00+08:00";
  scenario.node_capacity = {
      CapacitySample{.node = "node-001", .bps = 0.0, .at = scenario.window_start},
      CapacitySample{.node = "node-001", .bps = 1e9, .at = scenario.window_start + nanos_of(10)},
      CapacitySample{.node = "node-001", .bps = 0.0, .at = scenario.window_start + nanos_of(20)}};
  const auto plan = build_plan(scenario, config, testbed({"node-001"}));
  ASSERT_TRUE(plan) << plan.error().describe();
  const auto capacity = of<SetLinkCapacity>(*plan);
  ASSERT_EQ(capacity.size(), 3u);
  EXPECT_EQ(std::get<SetLinkCapacity>(capacity[0]->body).bps, 0.0);
  EXPECT_EQ(std::get<SetLinkCapacity>(capacity[1]->body).bps, 1e9);
  EXPECT_EQ(std::get<SetLinkCapacity>(capacity[2]->body).bps, 0.0);
}
