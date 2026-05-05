#pragma once

#include "rnbt/replay_config.hpp"
#include "rnbt/scenario.hpp"

#include <variant>

namespace rnbt {

struct ResetTestbed {
  std::uint32_t period{};
};

struct Depart {
  std::string node{};
  std::string event_id{};
};

struct Return {
  std::string node{};
  bool forced{};
};

struct StartInstance {
  std::string instance{};
  std::string job{};
  std::vector<std::string> nodes{};
  double progress_s{};
  double target_s{};
  bool park_after_resume{};
  std::optional<std::string> event_id{};
};

struct InjectReclaim {
  std::string event_id{};
  std::string node{};
  ReclaimKind kind{};
  Nanos notice{};
};

struct StopInstance {
  std::string instance{};
};

struct SetLinkCapacity {
  std::string node{};
  double bps{};
};

struct EndReplay {};

using ActionBody = std::variant<ResetTestbed, Depart, Return, StartInstance, InjectReclaim,
                                StopInstance, SetLinkCapacity, EndReplay>;

template <> struct describe<ResetTestbed> {
  using fields = std::tuple<Field<"period", &ResetTestbed::period>>;
};
template <> struct describe<Depart> {
  using fields = std::tuple<Field<"node", &Depart::node>, Field<"event_id", &Depart::event_id>>;
};
template <> struct describe<Return> {
  using fields = std::tuple<Field<"node", &Return::node>, Field<"forced", &Return::forced>>;
};
template <> struct describe<StartInstance> {
  using S = StartInstance;
  using fields =
      std::tuple<Field<"instance", &S::instance>, Field<"job", &S::job>, Field<"nodes", &S::nodes>,
                 Field<"progress_s", &S::progress_s>, Field<"target_s", &S::target_s>,
                 Field<"park_after_resume", &S::park_after_resume>,
                 Field<"event_id", &S::event_id>>;
};
template <> struct describe<InjectReclaim> {
  using I = InjectReclaim;
  using fields = std::tuple<Field<"event_id", &I::event_id>, Field<"node", &I::node>,
                            Field<"kind", &I::kind>, Field<"notice_s", &I::notice>>;
};
template <> struct describe<StopInstance> {
  using fields = std::tuple<Field<"instance", &StopInstance::instance>>;
};
template <> struct describe<SetLinkCapacity> {
  using fields =
      std::tuple<Field<"node", &SetLinkCapacity::node>, Field<"bps", &SetLinkCapacity::bps>>;
};
template <> struct describe<EndReplay> {
  using fields = std::tuple<>;
};

template <class A> struct action_name;
template <> struct action_name<ResetTestbed> {
  static constexpr std::string_view value = "reset";
};
template <> struct action_name<Depart> {
  static constexpr std::string_view value = "depart";
};
template <> struct action_name<Return> {
  static constexpr std::string_view value = "return";
};
template <> struct action_name<StartInstance> {
  static constexpr std::string_view value = "start";
};
template <> struct action_name<InjectReclaim> {
  static constexpr std::string_view value = "reclaim";
};
template <> struct action_name<StopInstance> {
  static constexpr std::string_view value = "stop";
};
template <> struct action_name<SetLinkCapacity> {
  static constexpr std::string_view value = "link_capacity";
};
template <> struct action_name<EndReplay> {
  static constexpr std::string_view value = "end";
};

template <class A>
concept PlannedAction = Described<A> && requires { action_name<A>::value; };

static_assert([]<class... A>(std::type_identity<std::variant<A...>>) {
  return (PlannedAction<A> && ...);
}(std::type_identity<ActionBody>{}));

[[nodiscard]] std::string_view name_of(const ActionBody &body) noexcept;

[[nodiscard]] unsigned action_order(const ActionBody &body) noexcept;

struct Action {
  Instant trace_at{};
  Nanos offset{};
  std::uint32_t period{};
  ActionBody body{};

  [[nodiscard]] auto key() const noexcept {
    return std::tuple{offset, trace_at, action_order(body)};
  }
};

[[nodiscard]] Json encode_action(const Action &action);

struct PeriodInfo {
  std::uint32_t index{};
  Instant start{};
  Instant end{};
  std::vector<std::string> events{};
};

template <> struct describe<PeriodInfo> {
  using fields =
      std::tuple<Field<"period", &PeriodInfo::index>, Field<"start_ns", &PeriodInfo::start>,
                 Field<"end_ns", &PeriodInfo::end>, Field<"events", &PeriodInfo::events>>;
};

struct PlanStats {
  std::uint64_t selected_events{};
  std::uint64_t skipped_unmapped{};
  std::uint64_t forced_returns{};
  std::uint64_t instances{};
  double trace_span_s{};
  double wall_span_s{};
};

template <> struct describe<PlanStats> {
  using P = PlanStats;
  using fields =
      std::tuple<Field<"selected_events", &P::selected_events>,
                 Field<"skipped_unmapped", &P::skipped_unmapped>,
                 Field<"forced_returns", &P::forced_returns>, Field<"instances", &P::instances>,
                 Field<"trace_span_s", &P::trace_span_s>, Field<"wall_span_s", &P::wall_span_s>>;
};

struct Plan {
  ReplayMode mode{};
  Instant from{};
  Instant to{};
  Instant origin{};
  std::vector<Action> actions{};
  std::vector<PeriodInfo> periods{};
  std::vector<std::string> nodes{};
  std::vector<std::string> events{};
  PlanStats stats{};

  [[nodiscard]] Json to_json(const Calendar &calendar) const;
};

[[nodiscard]] Expected<Instant> parse_window_bound(std::string_view text, const Calendar &calendar);

[[nodiscard]] Expected<Plan> build_plan(const Scenario &scenario, const ReplayConfig &config,
                                        const Testbed &testbed);

}
