#include "rnbt/shell.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <latch>
#include <mutex>
#include <poll.h>
#include <semaphore>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

namespace rnbt {

Expected<std::string> render(std::string_view pattern, const Vars &vars) {
  std::string out;
  out.reserve(pattern.size() + 64);
  for (std::size_t i = 0; i < pattern.size();) {
    if (pattern[i] != '{') {
      out.push_back(pattern[i++]);
      continue;
    }
    const auto close = pattern.find('}', i);
    if (close == std::string_view::npos)
      return failure(std::format("unterminated placeholder in '{}'", pattern));
    const auto key = pattern.substr(i + 1, close - i - 1);
    const auto found = vars.find(key);
    if (found == vars.end())
      return failure(std::format("unknown placeholder {{{}}} in '{}'", key, pattern));
    out += found->second;
    i = close + 1;
  }
  return out;
}

std::string shell_quote(std::string_view text) {
  std::string out{"'"};
  for (const char c : text) {
    if (c == '\'')
      out += "'\\''";
    else
      out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

CommandResult run_command(const std::string &line, std::chrono::duration<double> timeout) {
  CommandResult result;
  const auto started = std::chrono::steady_clock::now();
  std::array<int, 2> pipe_fds{};
#ifdef __linux__
  const bool pipe_failed = pipe2(pipe_fds.data(), O_CLOEXEC) != 0;
#else
  bool pipe_failed = pipe(pipe_fds.data()) != 0;
  if (!pipe_failed && (fcntl(pipe_fds[0], F_SETFD, FD_CLOEXEC) < 0 ||
                       fcntl(pipe_fds[1], F_SETFD, FD_CLOEXEC) < 0)) {
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    pipe_failed = true;
  }
#endif
  if (pipe_failed) {
    result.output = "pipe failed";
    return result;
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDERR_FILENO);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  posix_spawnattr_t attributes;
  posix_spawnattr_init(&attributes);
  posix_spawnattr_setpgroup(&attributes, 0);
  sigset_t child_signal_mask;
  sigemptyset(&child_signal_mask);
  posix_spawnattr_setsigmask(&attributes, &child_signal_mask);
  posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK);
  std::string shell{"/bin/bash"}, flag{"-c"}, body{line};
  std::array<char *, 4> argv{shell.data(), flag.data(), body.data(), nullptr};
  pid_t pid{};
  const int spawned = posix_spawn(&pid, argv[0], &actions, &attributes, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attributes);
  close(pipe_fds[1]);
  if (spawned != 0) {
    close(pipe_fds[0]);
    result.output = std::format("spawn failed: {}", std::strerror(spawned));
    return result;
  }
  const auto deadline =
      started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(timeout);
  std::array<char, 4096> buffer{};
  bool open = true;
  while (open) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      result.timed_out = true;
      kill(-pid, SIGKILL);
      break;
    }
    pollfd descriptor{.fd = pipe_fds[0], .events = POLLIN, .revents = 0};
    const int ready =
        poll(&descriptor, 1, static_cast<int>(std::min<long long>(remaining.count(), 200)));
    if (ready < 0 && errno != EINTR)
      break;
    if (ready > 0) {
      const auto count = read(pipe_fds[0], buffer.data(), buffer.size());
      if (count > 0) {
        if (result.output.size() < (1U << 20))
          result.output.append(buffer.data(), static_cast<std::size_t>(count));
      } else if (count == 0) {
        open = false;
      }
    }
  }
  close(pipe_fds[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  result.exit_code =
      WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

std::vector<CommandResult> run_parallel(std::span<const Command> commands, std::size_t parallelism,
                                        std::chrono::duration<double> timeout) {
  std::vector<CommandResult> results(commands.size());
  if (commands.empty())
    return results;
  std::counting_semaphore<1024> slots{
      static_cast<std::ptrdiff_t>(std::clamp<std::size_t>(parallelism, 1, 1024))};
  std::latch done{static_cast<std::ptrdiff_t>(commands.size())};
  std::vector<std::jthread> workers;
  workers.reserve(commands.size());
  for (std::size_t i = 0; i < commands.size(); ++i)
    workers.emplace_back([&, i] {
      slots.acquire();
      results[i] = run_command(commands[i].line, timeout);
      slots.release();
      done.count_down();
    });
  done.wait();
  return results;
}

Remote::Remote(const Testbed &testbed) : testbed_{testbed} {}

std::chrono::duration<double> Remote::timeout() const noexcept {
  return std::chrono::duration<double>{testbed_.command_timeout_s};
}

Vars Remote::node_vars(const HostConfig &host) const {
  const auto &p = testbed_.paths;
  Vars vars{{"node", host.trace_node},
            {"host", host.host},
            {"ssh", host.ssh},
            {"node_daemon_address", host.node_daemon_address},
            {"agent_address", host.node_daemon_address},
            {"migration_port", std::to_string(host.migration_port)},
            {"reclaimnet", p.reclaimnet},
            {"job_binary", p.job_binary},
            {"load_binary", p.load_binary},
            {"workload_root", p.workload_root},
            {"network_interface", host.network_interface.empty() ? "eth0" : host.network_interface},
            {"state_root", p.state_root},
            {"log_dir", p.log_dir},
            {"checkpoint_dir", p.checkpoint_dir},
            {"coordinator_data", p.coordinator_data},
            {"coordinator_address", testbed_.coordinator.address},
            {"env", ""}};
  for (const auto &[key, value] : host.vars)
    vars.insert_or_assign(key, value);
  for (const auto &[key, suffix] : std::array{std::pair{"workload_state", "/state.json"},
                                              std::pair{"profile_path", "/profile.json"},
                                              std::pair{"workload_log", "/logs/samples.jsonl"}})
    if (!host.vars.contains(key))
      vars.insert_or_assign(key, vars.at("workload_root") + suffix);
  return vars;
}

Vars Remote::coordinator_vars() const {
  const auto &p = testbed_.paths;
  Vars vars{{"host", testbed_.coordinator.host},
            {"ssh", testbed_.coordinator.ssh},
            {"coordinator_address", testbed_.coordinator.address},
            {"reclaimnet", p.reclaimnet},
            {"coordinator_data", p.coordinator_data},
            {"env", ""}};
  for (const auto &[key, value] : testbed_.coordinator.vars)
    vars.insert_or_assign(key, value);
  return vars;
}

Expected<Command> Remote::wrap(std::string label, std::string_view ssh, std::string inner) const {
  if (testbed_.local || ssh.empty())
    return Command{std::move(label), std::move(inner)};
  return render(testbed_.commands.remote,
                Vars{{"ssh", std::string{ssh}}, {"quoted", shell_quote(inner)}})
      .transform([&](std::string line) { return Command{std::move(label), std::move(line)}; });
}

Expected<Command> Remote::on_node(const HostConfig &host, std::string_view pattern,
                                  const Vars &extra) const {
  auto vars = node_vars(host);
  for (const auto &[key, value] : extra)
    vars.insert_or_assign(key, value);
  return render(pattern, vars).and_then([&](std::string inner) {
    return wrap(host.trace_node, host.ssh, std::move(inner));
  });
}

Expected<Command> Remote::on_coordinator(std::string_view pattern, const Vars &extra) const {
  auto vars = coordinator_vars();
  for (const auto &[key, value] : extra)
    vars.insert_or_assign(key, value);
  return render(pattern, vars).and_then([&](std::string inner) {
    return wrap("coordinator", testbed_.coordinator.ssh, std::move(inner));
  });
}

Expected<Command> Remote::fetch(std::string_view ssh, std::string_view remote_path,
                                std::string_view local_path) const {
  if (testbed_.local || ssh.empty())
    return Command{"fetch", std::format("mkdir -p {0} && cp -r {1}/. {0}/", shell_quote(local_path),
                                        shell_quote(remote_path))};
  return render(testbed_.commands.fetch, Vars{{"ssh", std::string{ssh}},
                                              {"remote_path", std::string{remote_path}},
                                              {"local_path", std::string{local_path}}})
      .transform([](std::string line) { return Command{"fetch", std::move(line)}; });
}

Expected<Command> Remote::upload(const HostConfig &host, const std::filesystem::path &local_path,
                                 std::string_view remote_path) const {
  if (testbed_.local || host.ssh.empty())
    return Command{host.trace_node, std::format("cp -- {} {}", shell_quote(local_path.string()),
                                                shell_quote(remote_path))};
  return Command{host.trace_node,
                 std::format("scp -q -- {} {}", shell_quote(local_path.string()),
                             shell_quote(host.ssh + ":" + std::string{remote_path}))};
}

}
