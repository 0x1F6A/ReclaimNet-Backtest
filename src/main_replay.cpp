#include "rnbt/driver.hpp"

#include <csignal>
#include <iostream>
#include <pthread.h>
#include <thread>

namespace {

struct Cli {
  std::string command{};
  std::string config{};
  std::vector<std::string> overrides{};
  std::optional<std::string> out{};
  std::optional<std::string> run{};
  std::optional<std::string> action{};
  bool dry_run = false;
};

[[nodiscard]] rnbt::Expected<Cli> parse(std::span<char *const> argv) {
  if (argv.size() < 3)
    return rnbt::failure("usage: rnbt-replay <plan|run|collect|render|testbed> <config.json> "
                         "[--set /key=value] [--out path] [--run dir] [--dry-run] [reset|stop]");
  Cli cli{.command = argv[1], .config = argv[2]};
  for (std::size_t i = 3; i < argv.size(); ++i) {
    const std::string_view token{argv[i]};
    const auto value = [&]() -> rnbt::Expected<std::string> {
      if (i + 1 >= argv.size())
        return rnbt::failure(std::format("{} needs a value", token));
      return std::string{argv[++i]};
    };
    if (token == "--set") {
      auto v = value();
      if (!v)
        return std::unexpected{v.error()};
      cli.overrides.push_back(*v);
    } else if (token == "--out") {
      auto v = value();
      if (!v)
        return std::unexpected{v.error()};
      cli.out = *v;
    } else if (token == "--run") {
      auto v = value();
      if (!v)
        return std::unexpected{v.error()};
      cli.run = *v;
    } else if (token == "--dry-run") {
      cli.dry_run = true;
    } else if (!token.starts_with("--")) {
      cli.action = std::string{token};
    } else {
      return rnbt::failure(std::format("unknown option {}", token));
    }
  }
  return cli;
}

[[nodiscard]] int report(const rnbt::Error &error) {
  std::cerr << "rnbt-replay: " << error.message << '\n';
  return 1;
}

}

int main(int argc, char **argv) {
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  std::stop_source stop;
  std::jthread watcher{[&](std::stop_token token) {
    timespec poll{.tv_sec = 0, .tv_nsec = 200'000'000};
    while (!token.stop_requested()) {
      if (sigtimedwait(&signals, nullptr, &poll) > 0) {
        std::cerr << "rnbt-replay: stopping after the current action\n";
        stop.request_stop();
        return;
      }
    }
  }};
  const auto cli = parse({argv, static_cast<std::size_t>(argc)});
  if (!cli)
    return report(cli.error());
  auto replay = rnbt::load_replay(cli->config, cli->overrides);
  if (!replay)
    return report(replay.error());
  auto scenario = rnbt::Scenario::load(rnbt::ScenarioSource{
      .directory = replay->config.trace, .migration_priors = replay->config.migration_priors});
  if (!scenario)
    return report(scenario.error());
  if (cli->command == "render") {
    const auto out = cli->out.value_or(replay->config.output + "/deploy");
    if (auto rendered = rnbt::render_deployment(*replay, *scenario, out); !rendered)
      return report(rendered.error());
    std::cout << "deployment files written to " << out << '\n';
    return 0;
  }
  if (cli->command == "collect") {
    const auto run = cli->run.value_or(replay->config.output);
    if (auto collected = rnbt::collect(*replay, run); !collected)
      return report(collected.error());
    std::cout << "collected into " << run << "/collected\n";
    return 0;
  }
  auto plan = rnbt::build_plan(*scenario, replay->config, replay->testbed);
  if (!plan)
    return report(plan.error());
  if (cli->command == "plan") {
    const auto json = plan->to_json(scenario->calendar);
    if (cli->out) {
      if (auto written = rnbt::write_json_file(*cli->out, json); !written)
        return report(written.error());
    }
    std::cout << std::format(
        "{} actions, {} events, {} instances, {} periods, trace span {:.1f} h, wall span "
        "{:.1f} h, skipped (unmapped) {}\n",
        plan->actions.size(), plan->stats.selected_events, plan->stats.instances,
        plan->periods.size(), plan->stats.trace_span_s / 3600.0, plan->stats.wall_span_s / 3600.0,
        plan->stats.skipped_unmapped);
    return 0;
  }
  rnbt::Driver driver{*replay, *scenario, *plan, replay->config.output, cli->dry_run};
  if (cli->command == "testbed") {
    rnbt::ActionRecord record{.type = cli->action.value_or("")};
    rnbt::Status status;
    if (record.type == "reset")
      status = driver.testbed_reset(0, plan->from, record, stop.get_token());
    else if (record.type == "stop")
      status = driver.testbed_stop(record);
    else
      return report(rnbt::Error{"testbed needs reset or stop"});
    std::cout << rnbt::encode(record).dump(2) << '\n';
    return status ? 0 : 1;
  }
  if (cli->command != "run")
    return report(rnbt::Error{std::format("unknown command {}", cli->command)});
  std::cout << std::format("replaying {} actions over {:.1f} h of wall time into {}\n",
                           plan->actions.size(), plan->stats.wall_span_s / 3600.0,
                           replay->config.output);
  auto result = driver.run(stop.get_token());
  if (!result)
    return report(result.error());
  std::cout << std::format("done: {} actions, {} failed, max lateness {:.2f} s\n", result->actions,
                           result->failed, result->max_lateness_s);
  return result->failed == 0 ? 0 : 2;
}
