#pragma once

#include "rnbt/replay_config.hpp"

#include <chrono>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace rnbt {

using Vars = std::map<std::string, std::string, std::less<>>;

[[nodiscard]] Expected<std::string> render(std::string_view pattern, const Vars &vars);
[[nodiscard]] std::string shell_quote(std::string_view text);

struct CommandResult {
  int exit_code = -1;
  std::string output{};
  double seconds{};
  bool timed_out = false;

  [[nodiscard]] bool ok() const noexcept { return exit_code == 0 && !timed_out; }
};

struct Command {
  std::string label{};
  std::string line{};
};

[[nodiscard]] CommandResult run_command(const std::string &line,
                                        std::chrono::duration<double> timeout);

[[nodiscard]] std::vector<CommandResult> run_parallel(std::span<const Command> commands,
                                                      std::size_t parallelism,
                                                      std::chrono::duration<double> timeout);

class Remote {
public:
  explicit Remote(const Testbed &testbed);

  [[nodiscard]] Vars node_vars(const HostConfig &host) const;
  [[nodiscard]] Vars coordinator_vars() const;
  [[nodiscard]] Expected<Command> on_node(const HostConfig &host, std::string_view pattern,
                                          const Vars &extra = {}) const;
  [[nodiscard]] Expected<Command> on_coordinator(std::string_view pattern,
                                                 const Vars &extra = {}) const;
  [[nodiscard]] Expected<Command> fetch(std::string_view ssh, std::string_view remote_path,
                                        std::string_view local_path) const;
  [[nodiscard]] Expected<Command> upload(const HostConfig &host,
                                         const std::filesystem::path &local_path,
                                         std::string_view remote_path) const;
  [[nodiscard]] std::chrono::duration<double> timeout() const noexcept;

private:
  [[nodiscard]] Expected<Command> wrap(std::string label, std::string_view ssh,
                                       std::string inner) const;
  const Testbed &testbed_;
};

}
