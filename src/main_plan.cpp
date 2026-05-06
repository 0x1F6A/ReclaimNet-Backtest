#include "rnbt/plan.hpp"

#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: rnbt-plan <replay.json> [--out plan.json] [--set /key=value]\n";
    return 1;
  }
  std::optional<std::string> output;
  std::vector<std::string> overrides;
  for (int i = 2; i < argc; ++i) {
    const std::string_view key{argv[i]};
    if (i + 1 >= argc || (key != "--out" && key != "--set")) {
      std::cerr << "rnbt-plan: expected --out path or --set /key=value\n";
      return 1;
    }
    if (key == "--out")
      output = argv[++i];
    else
      overrides.emplace_back(argv[++i]);
  }
  const auto replay = rnbt::load_replay(argv[1], overrides);
  if (!replay) {
    std::cerr << "rnbt-plan: " << replay.error().message << '\n';
    return 1;
  }
  const auto scenario = rnbt::Scenario::load(
      {.directory = replay->config.trace, .migration_priors = replay->config.migration_priors});
  if (!scenario) {
    std::cerr << "rnbt-plan: " << scenario.error().message << '\n';
    return 1;
  }
  const auto plan = rnbt::build_plan(*scenario, replay->config, replay->testbed);
  if (!plan) {
    std::cerr << "rnbt-plan: " << plan.error().message << '\n';
    return 1;
  }
  const auto result = plan->to_json(scenario->calendar);
  if (output) {
    if (auto saved = rnbt::write_json_file(*output, result); !saved) {
      std::cerr << "rnbt-plan: " << saved.error().message << '\n';
      return 1;
    }
  } else
    std::cout << result.dump(2) << '\n';
  std::cerr << std::format("{} actions, {} job instances, {} reclaim events; window [{}, {})\n",
                           plan->actions.size(), plan->stats.instances, plan->stats.selected_events,
                           scenario->calendar.format(plan->from),
                           scenario->calendar.format(plan->to));
  return 0;
}
