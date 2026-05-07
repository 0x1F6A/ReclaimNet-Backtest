#include "rnbt/gpu.hpp"
#include "rnbt/jsonl.hpp"
#include "rnbt/workload.hpp"

#include <cerrno>
#include <charconv>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/file.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using namespace rnbt;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr double kPeriodSeconds = 0.2;
constexpr std::uint64_t kDeviceBytes = 1U << 20;
volatile std::sig_atomic_t stop_requested = 0;

extern "C" void request_stop(int) { stop_requested = 1; }

std::int64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int fail(std::string_view message) {
  std::cerr << "rnbt-load: " << message << '\n';
  return 1;
}

struct State {
  std::string status = "starting";
  std::string error;
  std::int64_t pid{};
  std::string process_start_ticks;
  std::string executable;
  std::uint64_t samples_written{};
  std::string node;
  std::string profile;
  std::string log;
  std::int64_t start_wall_ns{};
  std::uint32_t gpus{};
};

Json state_json(const State &state) {
  return Json{{"schema_version", 1},
              {"program", "rnbt-load"},
              {"status", state.status},
              {"error", state.error},
              {"pid", state.pid},
              {"process_start_ticks", state.process_start_ticks},
              {"executable", state.executable},
              {"samples_written", state.samples_written},
              {"node", state.node},
              {"profile", state.profile},
              {"log", state.log},
              {"start_wall_ns", state.start_wall_ns},
              {"gpus", state.gpus},
              {"gpu_backend", "cuda"},
              {"telemetry_source", "nvml"},
              {"updated_ns", wall_ns()}};
}

Expected<State> load_state(const fs::path &path) {
  const auto json = read_json_file(path);
  if (!json)
    return std::unexpected{json.error()};
  try {
    if (json->at("schema_version") != 1 || json->at("program") != "rnbt-load")
      return failure("state is not an rnbt-load state file");
    State state;
    state.status = json->at("status").get<std::string>();
    state.error = json->at("error").get<std::string>();
    state.pid = json->at("pid").get<std::int64_t>();
    state.process_start_ticks = json->at("process_start_ticks").get<std::string>();
    state.executable = json->at("executable").get<std::string>();
    state.samples_written = json->at("samples_written").get<std::uint64_t>();
    state.node = json->at("node").get<std::string>();
    state.profile = json->at("profile").get<std::string>();
    state.log = json->at("log").get<std::string>();
    state.start_wall_ns = json->at("start_wall_ns").get<std::int64_t>();
    state.gpus = json->at("gpus").get<std::uint32_t>();
    if (state.pid < 0 || state.pid > std::numeric_limits<pid_t>::max())
      return failure("rnbt-load state has an invalid process ID");
    return state;
  } catch (const std::exception &error) {
    return failure(std::format("{}: {}", path.string(), error.what()));
  }
}

Status write_all(int fd, std::string_view text) {
  while (!text.empty()) {
    const auto count = write(fd, text.data(), text.size());
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return failure(
          std::format("write failed: {}", count == 0 ? "no progress" : std::strerror(errno)));
    text.remove_prefix(static_cast<std::size_t>(count));
  }
  return {};
}

Status save_state(const fs::path &path, const State &state) {
  const auto temporary = fs::path{path}.concat(std::format(".{}.tmp", getpid()));
  const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0)
    return failure(std::format("cannot write {}: {}", temporary.string(), std::strerror(errno)));
  Status result = write_all(fd, state_json(state).dump() + "\n");
  if (result && fsync(fd) != 0)
    result = failure(std::format("cannot flush state {}: {}", path.string(), std::strerror(errno)));
  if (close(fd) != 0 && result)
    result = failure(std::format("cannot close state {}: {}", path.string(), std::strerror(errno)));
  std::error_code error;
  if (result) {
    fs::rename(temporary, path, error);
    if (error)
      result = failure(std::format("cannot replace state {}: {}", path.string(), error.message()));
  }
  if (!result)
    fs::remove(temporary, error);
  return result;
}

struct Process {
  char status = '?';
  std::string start_ticks;
};

Expected<Process> process_info(std::int64_t pid) {
  std::ifstream input{std::format("/proc/{}/stat", pid)};
  std::string line;
  if (!std::getline(input, line))
    return failure("cannot read process identity from /proc");
  const auto close = line.rfind(')');
  if (close == std::string::npos)
    return failure("invalid process identity in /proc");
  std::istringstream fields{line.substr(close + 1)};
  Process process;
  std::string field;
  for (unsigned index = 3; index <= 22; ++index) {
    if (!(fields >> field))
      return failure("truncated process identity in /proc");
    if (index == 3)
      process.status = field.at(0);
    if (index == 22)
      process.start_ticks = field;
  }
  return process;
}

std::string executable_for(std::int64_t pid) {
  std::error_code error;
  const auto executable = fs::read_symlink(std::format("/proc/{}/exe", pid), error);
  return error ? std::string{} : executable.string();
}

bool process_matches(const State &state) {
  if (state.pid <= 0 || state.pid == getpid() || state.process_start_ticks.empty())
    return false;
  const auto process = process_info(state.pid);
  return process && process->status != 'Z' && process->status != 'X' &&
         process->start_ticks == state.process_start_ticks && !state.executable.empty() &&
         state.executable == executable_for(getpid()) &&
         state.executable == executable_for(state.pid);
}

int fail_state(const fs::path &path, State &state, std::string message) {
  state.status = "failed";
  state.error = std::move(message);
  if (auto saved = save_state(path, state); !saved)
    std::cerr << "rnbt-load: " << saved.error().message << '\n';
  return fail(state.error);
}

Status append_sample(int fd, const Json &sample) {
  if (auto written = write_all(fd, sample.dump() + "\n"); !written)
    return written;
  if (fsync(fd) != 0)
    return failure(std::format("cannot flush real GPU sample log: {}", std::strerror(errno)));
  return {};
}

class DeviceLeases {
public:
  ~DeviceLeases() {
    for (const int fd : descriptors_)
      close(fd);
  }
  DeviceLeases() = default;
  DeviceLeases(const DeviceLeases &) = delete;
  DeviceLeases &operator=(const DeviceLeases &) = delete;

  Status acquire(const Json &samples) {
    for (const auto &device : samples) {
      const auto uuid = device.at("uuid").get<std::string>();
      if (uuid.empty() || !std::ranges::all_of(uuid, [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '_';
          }))
        return failure("NVML returned an invalid UUID for the physical GPU actor lease");
      const auto path = std::format("/tmp/rnbt-load-{}-{}.lock", getuid(), uuid);
      const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
      if (fd < 0)
        return failure(
            std::format("cannot open physical GPU actor lease {}: {}", path, std::strerror(errno)));
      if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return failure(std::format(
            "GPU {} is already held by another workload actor; stop the other replay first", uuid));
      }
      descriptors_.push_back(fd);
    }
    return {};
  }

private:
  std::vector<int> descriptors_;
};

void sleep_until(std::chrono::time_point<Clock, std::chrono::duration<double>> deadline) {
  while (!stop_requested && Clock::now() < deadline)
    std::this_thread::sleep_until(std::min(
        deadline, std::chrono::time_point<Clock, std::chrono::duration<double>>{Clock::now()} +
                      std::chrono::duration<double>{0.025}));
}

int daemon_loop(const fs::path &state_path, State state, const job::WorkloadProfile &profile) {
  struct sigaction action{};
  action.sa_handler = request_stop;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, nullptr) != 0 || sigaction(SIGINT, &action, nullptr) != 0)
    return fail_state(state_path, state, "cannot install load actor signal handlers");
  sigset_t handled_signals;
  sigemptyset(&handled_signals);
  sigaddset(&handled_signals, SIGTERM);
  sigaddset(&handled_signals, SIGINT);
  if (sigprocmask(SIG_UNBLOCK, &handled_signals, nullptr) != 0)
    return fail_state(state_path, state, "cannot unblock load actor stop signals");
  state.pid = getpid();
  const auto process = process_info(state.pid);
  state.executable = executable_for(state.pid);
  if (process)
    state.process_start_ticks = process->start_ticks;
  if (auto saved = save_state(state_path, state); !saved)
    return fail_state(state_path, state, saved.error().message);
  job::Gpu gpu;
  if (auto ready = gpu.initialize(kDeviceBytes * state.gpus, state.gpus); !ready)
    return fail_state(state_path, state, ready.error().message);
  if (!process || state.executable.empty())
    return fail_state(state_path, state,
                      "physical load actors require Linux /proc process identity");
  const int fd = open(state.log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd < 0)
    return fail_state(
        state_path, state,
        std::format("cannot open sample log {}: {}", state.log, std::strerror(errno)));
  const auto finish_failure = [&](std::string message) {
    close(fd);
    return fail_state(state_path, state, std::move(message));
  };
  auto actual = gpu.sample();
  if (!actual)
    return finish_failure(actual.error().message);
  DeviceLeases leases;
  if (auto exclusive = leases.acquire(*actual); !exclusive)
    return finish_failure(exclusive.error().message);
  if (auto logged = append_sample(fd, Json{{"event", "ready"},
                                           {"t_ns", wall_ns()},
                                           {"node", state.node},
                                           {"pid", state.pid},
                                           {"gpu_backend", "cuda"},
                                           {"telemetry_source", "nvml"},
                                           {"actual_gpu_samples", *actual}});
      !logged)
    return finish_failure(logged.error().message);
  const auto ready_ns = wall_ns();
  if (ready_ns >= state.start_wall_ns &&
      static_cast<double>(ready_ns - state.start_wall_ns) / 1e9 >= profile.duration_s())
    return finish_failure("entire selected window was missed before physical GPU replay could run");
  state.status = "running";
  state.error.clear();
  if (auto saved = save_state(state_path, state); !saved)
    return finish_failure(saved.error().message);

  std::uint64_t sequence = 0;
  while (!stop_requested) {
    const auto now_ns = wall_ns();
    if (now_ns < state.start_wall_ns) {
      sleep_until(Clock::now() +
                  std::chrono::duration<double>{std::min(
                      kPeriodSeconds, static_cast<double>(state.start_wall_ns - now_ns) / 1e9)});
      continue;
    }
    const double offset = static_cast<double>(now_ns - state.start_wall_ns) / 1e9;
    if (offset >= profile.duration_s()) {
      if (state.samples_written == 0)
        return finish_failure(
            "entire selected window was missed before physical GPU replay could run");
      break;
    }
    const auto target = profile.at(offset);
    if (!target)
      return finish_failure(target.error().message);
    const auto period =
        std::chrono::duration<double>{std::min(kPeriodSeconds, profile.duration_s() - offset)};
    const auto busy = period * (target->gpu_util_pct / 100.0);
    const auto begin = Clock::now();
    if (auto worked = gpu.run_for(busy, ++sequence); !worked)
      return finish_failure(worked.error().message);
    const double busy_elapsed = std::chrono::duration<double>{Clock::now() - begin}.count();
    sleep_until(begin + period);
    actual = gpu.sample();
    if (!actual)
      return finish_failure(actual.error().message);
    double actual_util = 0;
    for (const auto &device : *actual)
      actual_util += device.at("utilization_gpu_percent").get<double>();
    actual_util /= static_cast<double>(actual->size());
    const Json sample{{"event", "sample"},
                      {"t_ns", wall_ns()},
                      {"node", state.node},
                      {"pid", state.pid},
                      {"trace_offset_s", offset},
                      {"profile", state.profile},
                      {"gpu_backend", "cuda"},
                      {"telemetry_source", "nvml"},
                      {"target",
                       {{"gpu_util_pct", target->gpu_util_pct},
                        {"link_tx_bps", target->link_tx_bps},
                        {"link_rx_bps", target->link_rx_bps}}},
                      {"actual_gpu_samples", *actual},
                      {"actual_gpu_util_pct", actual_util},
                      {"gpu_util_error_pct", actual_util - target->gpu_util_pct},
                      {"control_period_s", period.count()},
                      {"busy_budget_s", busy.count()},
                      {"busy_elapsed_s", busy_elapsed}};
    if (auto logged = append_sample(fd, sample); !logged)
      return finish_failure(logged.error().message);
    ++state.samples_written;
    if (auto saved = save_state(state_path, state); !saved)
      return finish_failure(saved.error().message);
  }
  state.status = stop_requested ? "stopped" : "completed";
  if (auto logged = append_sample(fd, Json{{"event", state.status},
                                           {"t_ns", wall_ns()},
                                           {"node", state.node},
                                           {"pid", state.pid},
                                           {"samples_written", state.samples_written}});
      !logged)
    return finish_failure(logged.error().message);
  if (close(fd) != 0)
    return fail_state(state_path, state,
                      std::format("cannot close sample log: {}", std::strerror(errno)));
  if (auto saved = save_state(state_path, state); !saved)
    return fail_state(state_path, state, saved.error().message);
  return 0;
}

using Args = std::map<std::string, std::string, std::less<>>;

Expected<Args> parse_args(int argc, char **argv) {
  Args args;
  for (int index = 2; index < argc; index += 2) {
    const std::string_view key{argv[index]};
    if (!key.starts_with("--") || index + 1 >= argc ||
        std::string_view{argv[index + 1]}.starts_with("--"))
      return failure("options must be --name value pairs");
    if (!args.emplace(std::string{key.substr(2)}, argv[index + 1]).second)
      return failure(std::format("duplicate option {}", key));
  }
  return args;
}

template <class T> Expected<T> integer_arg(const Args &args, std::string_view key) {
  const auto found = args.find(key);
  if (found == args.end())
    return failure(std::format("--{} is required", key));
  T value{};
  const auto [end, error] =
      std::from_chars(found->second.data(), found->second.data() + found->second.size(), value);
  if (error != std::errc{} || end != found->second.data() + found->second.size())
    return failure(std::format("--{} must be a valid integer", key));
  return value;
}

Status create_parent(const fs::path &path) {
  std::error_code error;
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path(), error);
  return error ? Status{failure(std::format("cannot create {}: {}", path.parent_path().string(),
                                            error.message()))}
               : Status{};
}

int start(const Args &args) {
  for (const auto *name : {"profile", "state", "log", "gpus", "start-wall-ns"})
    if (!args.contains(name) || args.at(name).empty())
      return fail(std::format("--{} is required", name));
  for (const auto &[name, value] : args) {
    (void)value;
    if (name != "profile" && name != "state" && name != "log" && name != "gpus" &&
        name != "start-wall-ns")
      return fail(std::format("unknown option --{}", name));
  }
  const auto gpus = integer_arg<std::uint32_t>(args, "gpus");
  const auto origin = integer_arg<std::int64_t>(args, "start-wall-ns");
  if (!gpus || *gpus == 0)
    return fail(gpus ? "--gpus must be positive" : gpus.error().message);
  if (!origin || *origin <= 0)
    return fail(origin ? "--start-wall-ns must be positive" : origin.error().message);
  const auto profile_path = fs::absolute(args.at("profile"));
  const auto state_path = fs::absolute(args.at("state"));
  const auto log_path = fs::absolute(args.at("log"));
  if (profile_path == state_path || profile_path == log_path || state_path == log_path)
    return fail("profile, state and log must be distinct files");
  const auto profile = job::WorkloadProfile::load(profile_path);
  if (!profile)
    return fail(profile.error().message);
  if (auto parent = create_parent(state_path); !parent)
    return fail(parent.error().message);
  if (auto parent = create_parent(log_path); !parent)
    return fail(parent.error().message);
  const auto lock_path = fs::path{state_path}.concat(".lock");
  const int lock = open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (lock < 0)
    return fail(std::format("cannot open actor lock: {}", std::strerror(errno)));
  if (flock(lock, LOCK_EX | LOCK_NB) != 0) {
    close(lock);
    const auto existing = load_state(state_path);
    if (existing && existing->status == "running" && process_matches(*existing) &&
        existing->profile == profile_path.string() && existing->log == log_path.string() &&
        existing->gpus == *gpus && existing->start_wall_ns == *origin)
      return 0;
    return fail("another load actor holds this state file; stop it before changing the profile");
  }
  State state;
  state.node = profile->trace_node();
  state.profile = profile_path.string();
  state.log = log_path.string();
  state.gpus = *gpus;
  state.start_wall_ns = *origin;
  for (const char *key : {"RNBT_FAKE_GPU_DIR", "RNBT_FAKE_GPUS", "RNBT_FAKE_VRAM_BYTES"}) {
    if (const char *value = std::getenv(key); value && *value) {
      close(lock);
      return fail_state(
          state_path, state,
          std::format("{} is not supported; physical replay requires real CUDA/NVML", key));
    }
  }
  if (auto saved = save_state(state_path, state); !saved) {
    close(lock);
    return fail(saved.error().message);
  }
  const pid_t child = fork();
  if (child < 0) {
    close(lock);
    return fail_state(state_path, state, std::format("fork failed: {}", std::strerror(errno)));
  }
  if (child == 0) {
    if (setsid() < 0)
      std::_Exit(fail_state(state_path, state, "cannot detach GPU load actor"));
    const pid_t grandchild = fork();
    if (grandchild < 0)
      std::_Exit(fail_state(state_path, state, "cannot create detached GPU load actor"));
    if (grandchild > 0)
      std::_Exit(0);
    const int null = open("/dev/null", O_RDWR | O_CLOEXEC);
    if (null >= 0) {
      dup2(null, STDIN_FILENO);
      dup2(null, STDOUT_FILENO);
      dup2(null, STDERR_FILENO);
    }
    for (int fd = 3; fd < 1024; ++fd)
      if (fd != lock)
        close(fd);
    int result = 1;
    try {
      result = daemon_loop(state_path, state, *profile);
    } catch (const std::exception &error) {
      if (auto latest = load_state(state_path))
        state = *latest;
      result = fail_state(state_path, state, error.what());
    }
    close(lock);
    std::_Exit(result);
  }
  close(lock);
  int child_status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &child_status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != child || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)
    return fail("could not create the GPU load actor");
  const auto deadline = Clock::now() + std::chrono::seconds{120};
  while (Clock::now() < deadline) {
    auto current = load_state(state_path);
    if (current) {
      if (current->status == "failed")
        return fail(current->error);
      if (current->status == "completed" ||
          (current->status == "running" && process_matches(*current)))
        return 0;
      if (current->status == "stopped")
        return fail("GPU load actor was stopped during initialization");
      if (current->pid > 0) {
        const auto process = process_info(current->pid);
        if ((process && (process->status == 'Z' || process->status == 'X')) ||
            (kill(static_cast<pid_t>(current->pid), 0) != 0 && errno == ESRCH))
          return fail_state(state_path, *current, "GPU load actor exited during initialization");
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{25});
  }
  if (auto current = load_state(state_path)) {
    state = *current;
    if (process_matches(state)) {
      kill(static_cast<pid_t>(state.pid), SIGCONT);
      kill(static_cast<pid_t>(state.pid), SIGKILL);
    }
  }
  return fail_state(state_path, state, "timed out initializing real GPU load actor");
}

int stop(const fs::path &state_path) {
  std::error_code error;
  if (!fs::exists(state_path, error) && !error)
    return 0;
  auto state = load_state(state_path);
  if (!state)
    return fail(state.error().message);
  if ((state->status == "running" || state->status == "starting") && !process_matches(*state)) {
    state->status = "failed";
    state->error = "GPU load actor is no longer running or its process identity does not match";
    if (auto saved = save_state(state_path, *state); !saved)
      return fail(saved.error().message);
    return 0;
  }
  bool forced_kill = false;
  if (process_matches(*state)) {
    if (kill(static_cast<pid_t>(state->pid), SIGCONT) != 0 && errno != ESRCH)
      return fail(std::format("cannot continue GPU load actor: {}", std::strerror(errno)));
    if (process_matches(*state) && kill(static_cast<pid_t>(state->pid), SIGTERM) != 0 &&
        errno != ESRCH)
      return fail(std::format("cannot stop GPU load actor: {}", std::strerror(errno)));
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (process_matches(*state) && Clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds{25});
    if (process_matches(*state)) {
      forced_kill = true;
      if (kill(static_cast<pid_t>(state->pid), SIGKILL) != 0 && errno != ESRCH)
        return fail_state(state_path, *state,
                          std::format("cannot kill GPU load actor: {}", std::strerror(errno)));
      const auto killed_deadline = Clock::now() + std::chrono::seconds{2};
      while (process_matches(*state) && Clock::now() < killed_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{25});
      if (process_matches(*state))
        return fail_state(state_path, *state, "GPU load actor did not exit after SIGKILL");
    }
  }
  if (auto latest = load_state(state_path))
    state = std::move(latest);
  if (forced_kill || state->status == "running" || state->status == "starting") {
    state->status = "failed";
    state->error = forced_kill
                       ? "GPU load actor did not stop gracefully and required SIGKILL"
                       : "GPU load actor exited without recording a final state during stop";
    if (auto saved = save_state(state_path, *state); !saved)
      return fail(saved.error().message);
  }
  return 0;
}

int status(const fs::path &state_path) {
  auto state = load_state(state_path);
  if (!state)
    return fail(state.error().message);
  const auto process = process_info(state->pid);
  const bool initializing_dead = state->status == "starting" && state->pid > 0 &&
                                 ((process && (process->status == 'Z' || process->status == 'X')) ||
                                  (kill(static_cast<pid_t>(state->pid), 0) != 0 && errno == ESRCH));
  if ((state->status == "running" && !process_matches(*state)) || initializing_dead) {
    state->status = "failed";
    state->error = "GPU load actor is no longer running";
    if (auto saved = save_state(state_path, *state); !saved)
      return fail(saved.error().message);
  }
  std::cout << state_json(*state).dump(2) << '\n';
  return state->status == "failed" ? 1 : 0;
}

}

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      return fail(
          "usage: rnbt-load start --profile FILE --state FILE --log FILE --gpus N --start-wall-ns NS | stop/status --state FILE");
    const auto args = parse_args(argc, argv);
    if (!args)
      return fail(args.error().message);
    const std::string_view command{argv[1]};
    if (command == "start")
      return start(*args);
    if (command != "stop" && command != "status")
      return fail(std::format("unknown command '{}'", command));
    if (args->size() != 1 || !args->contains("state") || args->at("state").empty())
      return fail("--state FILE is required");
    return command == "stop" ? stop(args->at("state")) : status(args->at("state"));
  } catch (const std::exception &error) {
    return fail(error.what());
  }
}
