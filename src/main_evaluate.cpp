#include "rnbt/evaluate.hpp"
#include "rnbt/jsonl.hpp"

#include <iostream>

int main(int argc, char **argv) {
  const std::span<char *const> args{argv, static_cast<std::size_t>(argc)};
  if (args.size() < 2) {
    std::cerr
        << "usage: rnbt-evaluate <run_dir> [--metrics metrics.json] [--baseline run_dir] [--out file]\n";
    return 1;
  }
  std::filesystem::path run{args[1]};
  std::optional<std::filesystem::path> metrics, baseline, out;
  for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
    const std::string_view key{args[i]};
    (key == "--metrics" ? metrics : key == "--baseline" ? baseline : out) = args[i + 1];
  }
  rnbt::MetricsSpec spec;
  if (metrics) {
    auto json = rnbt::read_json_file(*metrics);
    auto decoded =
        json.and_then([](const rnbt::Json &j) { return rnbt::try_decode<rnbt::MetricsSpec>(j); });
    if (!decoded) {
      std::cerr << "rnbt-evaluate: " << decoded.error().message << '\n';
      return 1;
    }
    spec = *decoded;
  }
  auto measured = rnbt::measure_run(run, spec);
  if (!measured) {
    std::cerr << "rnbt-evaluate: " << measured.error().message << '\n';
    return 1;
  }
  {
    rnbt::JsonlSink sink{run / "outcomes.jsonl"};
    for (const auto &outcome : measured->outcomes)
      sink.write(outcome);
  }
  auto summary = rnbt::summarize(*measured, spec);
  std::cout << rnbt::render_table(summary);
  if (baseline) {
    auto other = rnbt::measure_run(*baseline, spec);
    if (!other) {
      std::cerr << "rnbt-evaluate: " << other.error().message << '\n';
      return 1;
    }
    summary["comparison"] = rnbt::compare(*measured, *other, spec);
    std::cout << summary["comparison"].dump(2) << '\n';
  }
  if (auto written = rnbt::write_json_file(out.value_or(run / "metrics.json"), summary); !written) {
    std::cerr << "rnbt-evaluate: " << written.error().message << '\n';
    return 1;
  }
  return 0;
}
