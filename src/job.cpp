#include "rnbt/job.hpp"
#include "rnbt/gpu.hpp"

#include "rnbt/jsonl.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace rnbt::job {

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_pause = 0;
volatile std::sig_atomic_t g_continued = 0;

[[nodiscard]] std::int64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

class Args {
public:
  explicit Args(std::span<char *const> arguments) {
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      std::string_view token{arguments[i]};
      if (!token.starts_with("--"))
        continue;
      token.remove_prefix(2);
      if (i + 1 < arguments.size() && !std::string_view{arguments[i + 1]}.starts_with("--"))
        values_[std::string{token}].emplace_back(arguments[++i]);
      else
        values_[std::string{token}].emplace_back("true");
    }
  }

  template <class T> [[nodiscard]] std::optional<T> get(std::string_view key) const {
    const auto found = values_.find(key);
    if (found == values_.end() || found->second.empty())
      return std::nullopt;
    const std::string &text = found->second.back();
    if constexpr (std::same_as<T, std::string>) {
      return text;
    } else if constexpr (std::same_as<T, bool>) {
      return text == "true" || text == "1";
    } else {
      T value{};
      auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
      if (ec != std::errc{} || end != text.data() + text.size())
        return std::nullopt;
      return value;
    }
  }

  [[nodiscard]] bool contains(std::string_view key) const { return values_.contains(key); }

  [[nodiscard]] std::vector<std::string> all(std::string_view key) const {
    const auto found = values_.find(key);
    return found == values_.end() ? std::vector<std::string>{} : found->second;
  }

private:
  std::map<std::string, std::vector<std::string>, std::less<>> values_;
};

[[nodiscard]] std::string node_name(const Args &args) {
  if (auto node = args.get<std::string>("node"))
    return *node;
  if (const char *env = std::getenv("RNBT_NODE"); env && *env)
    return env;
  std::array<char, 256> host{};
  gethostname(host.data(), host.size() - 1);
  return host.data();
}

[[nodiscard]] bool alive(std::int64_t pid) {
  return pid > 0 && kill(static_cast<pid_t>(pid), 0) == 0;
}

[[nodiscard]] char process_state(std::int64_t pid) {
  std::ifstream stat{std::format("/proc/{}/stat", pid)};
  std::string content;
  std::getline(stat, content);
  const auto close = content.rfind(')');
  return close != std::string::npos && close + 2 < content.size() ? content[close + 2] : '?';
}

[[nodiscard]] fs::path state_file(const fs::path &directory) { return directory / "job.json"; }
[[nodiscard]] fs::path payload_file(const fs::path &directory) { return directory / "payload.bin"; }

Status save_state(const fs::path &directory, JobState &state) {
  state.updated_ns = wall_ns();
  const auto target = state_file(directory);
  const auto temporary = fs::path{target}.concat(std::format(".{}.tmp", getpid()));
  {
    std::ofstream out{temporary, std::ios::trunc};
    if (!out)
      return failure(std::format("cannot write {}", temporary.string()));
    out << encode(state).dump();
    out.flush();
    if (!out)
      return failure(std::format("cannot flush {}", temporary.string()));
  }
  std::error_code ec;
  fs::rename(temporary, target, ec);
  if (ec)
    return failure(std::format("rename {}: {}", target.string(), ec.message()));
  return {};
}

void append_log(const JobState &state, std::string_view event) {
  if (state.log_dir.empty())
    return;
  std::error_code ec;
  fs::create_directories(state.log_dir, ec);
  const auto path =
      fs::path{state.log_dir} / std::format("{}.r{}.jsonl", state.instance, state.rank);
  const auto line = encode(LogLine{.t_ns = wall_ns(),
                                   .instance = state.instance,
                                   .rank = state.rank,
                                   .node = state.node,
                                   .event = std::string{event},
                                   .progress_s = state.progress_s,
                                   .pid = state.pid,
                                   .status = state.status,
                                   .gpu_samples = state.gpu_samples})
                        .dump() +
                    "\n";
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd < 0)
    return;
  [[maybe_unused]] const auto written = write(fd, line.data(), line.size());
  close(fd);
}

[[nodiscard]] Status physical_environment() {
  for (const char *key : {"RNBT_FAKE_GPU_DIR", "RNBT_FAKE_GPUS", "RNBT_FAKE_VRAM_BYTES"})
    if (const char *value = std::getenv(key); value && *value)
      return failure(
          std::format("{} is not supported; physical replay requires real CUDA/NVML", key));
  return {};
}

[[nodiscard]] Status validate_resources(const JobState &state) {
  if (state.gpus == 0 || state.vram_bytes == 0)
    return failure("positive --gpus and --vram-bytes are required for physical GPU replay");
  if (state.checkpoint_bytes <= kHeaderBytes)
    return failure(
        std::format("--checkpoint-bytes must exceed {} so it contains GPU data", kHeaderBytes));
  if (state.checkpoint_bytes - kHeaderBytes > state.vram_bytes)
    return failure("checkpoint GPU payload exceeds the requested GPU memory");
  return {};
}

[[nodiscard]] Status copy_bytes(int source, int target, std::uint64_t bytes) {
  std::array<char, 1 << 20> buffer{};
  while (bytes > 0) {
    const auto size = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, buffer.size()));
    ssize_t count;
    do {
      count = read(source, buffer.data(), size);
    } while (count < 0 && errno == EINTR);
    if (count <= 0)
      return failure(count == 0 ? "truncated GPU payload"
                                : std::format("GPU payload read: {}", std::strerror(errno)));
    std::size_t offset = 0;
    while (offset < static_cast<std::size_t>(count)) {
      ssize_t written;
      do {
        written = write(target, buffer.data() + offset, static_cast<std::size_t>(count) - offset);
      } while (written < 0 && errno == EINTR);
      if (written <= 0)
        return failure(std::format("GPU payload write: {}", std::strerror(errno)));
      offset += static_cast<std::size_t>(written);
    }
    bytes -= static_cast<std::uint64_t>(count);
  }
  return {};
}

[[nodiscard]] Status save_gpu_payload(const fs::path &directory, const JobState &state, Gpu &gpu) {
  const auto target = payload_file(directory);
  const auto temporary = fs::path{target}.concat(std::format(".{}.tmp", getpid()));
  const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0)
    return failure(
        std::format("cannot write GPU payload {}: {}", temporary.string(), std::strerror(errno)));
  auto result = gpu.write_payload(fd, state.checkpoint_bytes - kHeaderBytes);
  if (result && fsync(fd) != 0)
    result = failure(std::format("cannot sync GPU payload: {}", std::strerror(errno)));
  close(fd);
  std::error_code ec;
  if (result) {
    fs::rename(temporary, target, ec);
    if (ec)
      result = failure(std::format("GPU payload rename: {}", ec.message()));
  }
  if (!result)
    fs::remove(temporary, ec);
  return result;
}

[[nodiscard]] Status restore_gpu_payload(const fs::path &directory, const JobState &state,
                                         Gpu &gpu) {
  const auto path = payload_file(directory);
  std::error_code ec;
  const auto size = fs::file_size(path, ec);
  if (ec || size != state.checkpoint_bytes - kHeaderBytes)
    return failure("missing or truncated GPU payload for restore");
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return failure(std::format("cannot read GPU payload {}", path.string()));
  const auto result = gpu.read_payload(fd, size);
  close(fd);
  return result;
}

[[nodiscard]] int fail_state(const fs::path &directory, JobState &state, std::string message) {
  state.error = std::move(message);
  state.status = "failed";
  (void)save_state(directory, state);
  append_log(state, "error:" + state.error);
  std::cerr << "rnbt-job: " << state.error << '\n';
  return 1;
}

extern "C" void on_stop(int) { g_stop = 1; }
extern "C" void on_pause(int) { g_pause = 1; }
extern "C" void on_continue(int) { g_continued = 1; }

void install(int signal, void (*handler)(int)) {
  struct sigaction action{};
  action.sa_handler = handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(signal, &action, nullptr);
}

int daemon_loop(const fs::path &directory, JobState state) {
  install(SIGTERM, on_stop);
  install(SIGINT, on_stop);
  install(SIGUSR1, on_pause);
  install(SIGCONT, on_continue);
  sigset_t controls;
  sigemptyset(&controls);
  for (const int signal : {SIGTERM, SIGINT, SIGUSR1, SIGCONT})
    sigaddset(&controls, signal);
  if (sigprocmask(SIG_UNBLOCK, &controls, nullptr) != 0)
    return fail_state(directory, state, "cannot unblock GPU job control signals");
  state.pid = getpid();
  if (auto saved = save_state(directory, state); !saved)
    return fail_state(directory, state, saved.error().message);
  Gpu gpu;
  if (auto ready = gpu.initialize(state.vram_bytes, state.gpus); !ready)
    return fail_state(directory, state, ready.error().message);
  if (state.restored) {
    if (auto restored = restore_gpu_payload(directory, state, gpu); !restored)
      return fail_state(directory, state, restored.error().message);
  }
  auto samples = gpu.sample();
  if (!samples)
    return fail_state(directory, state, samples.error().message);
  state.gpu_samples = std::move(*samples);
  state.status = "running";
  state.error.clear();
  if (auto saved = save_state(directory, state); !saved)
    return fail_state(directory, state, saved.error().message);
  append_log(state, state.restored ? "resume-start" : "start");
  append_log(state, "gpu:cuda");
  using clock = std::chrono::steady_clock;
  const auto tick = std::chrono::duration<double>{std::max(0.05, state.tick_s)};
  auto last = clock::now();
  double since_log = 0.0;
  double running_s = 0.0;
  std::uint64_t sequence = 0;
  std::string ending = "stop";
  while (true) {
    std::this_thread::sleep_for(tick);
    if (g_stop)
      break;
    if (g_continued) {
      g_continued = 0;
      state.status = "running";
      append_log(state, "continued");
      last = clock::now();
    }
    if (g_pause) {
      g_pause = 0;
      if (auto captured = save_gpu_payload(directory, state, gpu); !captured)
        return fail_state(directory, state, captured.error().message);
      state.status = "paused";
      if (auto saved = save_state(directory, state); !saved)
        return fail_state(directory, state, saved.error().message);
      append_log(state, "pause");
      raise(SIGSTOP);
      last = clock::now();
      continue;
    }
    if (!state.external_load)
      if (auto computed = gpu.step(++sequence); !computed)
        return fail_state(directory, state, computed.error().message);
    samples = gpu.sample();
    if (!samples)
      return fail_state(directory, state, samples.error().message);
    state.gpu_samples = std::move(*samples);
    const auto now = clock::now();
    const double delta = std::chrono::duration<double>(now - last).count();
    last = now;
    state.progress_s = std::min(state.target_s, state.progress_s + delta);
    running_s += delta;
    since_log += delta;
    if (state.progress_s >= state.target_s) {
      ending = "complete";
      break;
    }
    if (state.restored && state.exit_after_resume_s && running_s >= *state.exit_after_resume_s) {
      ending = "park";
      break;
    }
    if (auto saved = save_state(directory, state); !saved)
      return fail_state(directory, state, saved.error().message);
    if (since_log >= state.log_every_s) {
      since_log = 0.0;
      append_log(state, "tick");
    }
  }
  state.status = ending == "complete" ? "completed" : ending == "park" ? "parked" : "stopped";
  (void)save_state(directory, state);
  append_log(state, ending);
  return 0;
}

[[nodiscard]] int fail(std::string_view message) {
  std::cerr << "rnbt-job: " << message << '\n';
  return 1;
}

[[nodiscard]] int start(const Args &args) {
  const auto directory = args.get<std::string>("state");
  if (!directory)
    return fail("--state is required");
  const fs::path root{*directory};
  std::error_code ec;
  fs::create_directories(root, ec);
  JobState state;
  std::optional<JobState> active;
  if (auto existing = load_state(root)) {
    state = *existing;
    if (alive(state.pid) && (state.status == "running" || state.status == "paused"))
      active = state;
  } else {
    state.progress_s = args.get<double>("progress-s").value_or(0.0);
  }
  if (args.contains("gpus") && !args.get<std::uint32_t>("gpus"))
    return fail_state(root, state, "--gpus must be a valid unsigned integer");
  for (const auto *key : {"vram-bytes", "checkpoint-bytes"})
    if (args.contains(key) && !args.get<std::uint64_t>(key))
      return fail_state(root, state, std::format("--{} must be a valid unsigned integer", key));
  state.instance = args.get<std::string>("instance").value_or(state.instance);
  state.rank = args.get<std::uint32_t>("rank").value_or(state.rank);
  state.node = node_name(args);
  state.target_s = args.get<double>("target-s").value_or(state.target_s);
  state.checkpoint_bytes =
      args.get<std::uint64_t>("checkpoint-bytes").value_or(state.checkpoint_bytes);
  state.vram_bytes = args.get<std::uint64_t>("vram-bytes").value_or(state.vram_bytes);
  state.gpus = args.get<std::uint32_t>("gpus").value_or(state.gpus);
  state.log_dir = args.get<std::string>("log-dir").value_or(state.log_dir);
  state.tick_s = args.get<double>("tick-s").value_or(state.tick_s);
  state.log_every_s = args.get<double>("log-every-s").value_or(state.log_every_s);
  if (args.contains("external-load")) {
    const auto value = args.get<std::string>("external-load").value_or("");
    if (value != "true" && value != "false" && value != "1" && value != "0")
      return fail_state(root, state, "--external-load must be true or false");
    state.external_load = value == "true" || value == "1";
  }
  if (auto park = args.get<double>("exit-after-resume-s"))
    state.exit_after_resume_s = *park;
  if (state.instance.empty() || state.target_s <= 0.0)
    return fail("--instance and a positive --target-s are required");
  if (auto environment = physical_environment(); !environment)
    return fail_state(root, state, environment.error().message);
  if (auto resources = validate_resources(state); !resources)
    return fail_state(root, state, resources.error().message);
  if (!(state.tick_s > 0.0) || !(state.log_every_s > 0.0))
    return fail_state(root, state, "tick and log intervals must be positive");
  if (active) {
    if (!active->gpu_samples.is_array() || active->gpu_samples.empty())
      return fail("the existing process has no verified CUDA/NVML samples");
    if (state.gpus != active->gpus || state.vram_bytes != active->vram_bytes ||
        state.checkpoint_bytes != active->checkpoint_bytes || state.instance != active->instance)
      return fail("cannot change GPU resources or instance while a job is running");
    return 0;
  }
  state.error.clear();
  state.pid = 0;
  state.status = "starting";
  if (auto saved = save_state(root, state); !saved)
    return fail(saved.error().message);
  const pid_t child = fork();
  if (child < 0)
    return fail("fork failed");
  if (child == 0) {
    setsid();
    const pid_t grandchild = fork();
    if (grandchild != 0)
      std::_Exit(grandchild < 0 ? 1 : 0);
    const int null = open("/dev/null", O_RDWR | O_CLOEXEC);
    if (null >= 0) {
      dup2(null, STDIN_FILENO);
      dup2(null, STDOUT_FILENO);
      dup2(null, STDERR_FILENO);
    }
    for (int fd = 3; fd < 1024; ++fd)
      close(fd);
    const int result = daemon_loop(root, state);
    std::_Exit(result);
  }
  int status = 0;
  waitpid(child, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return fail_state(root, state, "could not create the GPU job process");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{120};
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto current = load_state(root)) {
      if (current->status == "running" && alive(current->pid))
        return 0;
      if (current->status == "failed")
        return fail(current->error);
      if (current->pid > 0 && !alive(current->pid))
        return fail_state(root, *current, "GPU job exited during initialization");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{25});
  }
  if (auto current = load_state(root); current && current->pid > 0) {
    kill(static_cast<pid_t>(current->pid), SIGKILL);
    state = *current;
  }
  return fail_state(root, state, "timed out initializing real GPU resources");
}

[[nodiscard]] int signal_job(const Args &args, std::string_view action) {
  const auto directory = args.get<std::string>("state");
  if (!directory)
    return fail("--state is required");
  auto state = load_state(*directory);
  if (!state)
    return action == "stop" ? 0 : fail(state.error().message);
  if (!alive(state->pid))
    return action == "stop" ? 0 : fail("the job process is not running");
  const auto pid = static_cast<pid_t>(state->pid);
  if (action == "pause") {
    if (state->status == "paused" && process_state(state->pid) == 'T')
      return 0;
    kill(pid, SIGUSR1);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds{120};
    while (std::chrono::steady_clock::now() < until) {
      auto current = load_state(*directory);
      if (current && current->status == "failed")
        return fail(current->error);
      if (!alive(state->pid))
        return fail("GPU job exited before checkpoint capture");
      if (current && current->status == "paused" && process_state(state->pid) == 'T')
        return 0;
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    return fail("timed out waiting for GPU checkpoint capture");
  }
  if (action == "resume") {
    kill(pid, SIGCONT);
    return 0;
  }
  kill(pid, SIGCONT);
  kill(pid, SIGTERM);
  const double timeout = args.get<double>("timeout-s").value_or(10.0);
  const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>{timeout};
  while (alive(state->pid) && std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  if (alive(state->pid))
    kill(pid, SIGKILL);
  return 0;
}

[[nodiscard]] int checkpoint(const Args &args) {
  const auto directory = args.get<std::string>("state");
  const auto out = args.get<std::string>("out");
  if (!directory || !out)
    return fail("--state and --out are required");
  auto state = load_state(*directory);
  if (!state)
    return fail(state.error().message);
  if (!alive(state->pid) || (state->status != "running" && state->status != "paused"))
    return fail("the GPU application is not running");
  const bool resume_after = state->status == "running";
  if (resume_after && signal_job(args, "pause") != 0)
    return 1;
  state = load_state(*directory);
  const auto resume = [&] {
    if (resume_after)
      (void)signal_job(args, "resume");
  };
  if (!state || state->status != "paused") {
    resume();
    return fail("GPU application did not capture a paused checkpoint");
  }
  const auto payload = payload_file(*directory);
  std::error_code ec;
  const auto payload_bytes = fs::file_size(payload, ec);
  if (ec || payload_bytes != state->checkpoint_bytes - kHeaderBytes) {
    resume();
    return fail("missing or truncated GPU checkpoint payload");
  }
  const CheckpointHeader header{.external_load = state->external_load,
                                .instance = state->instance,
                                .rank = state->rank,
                                .progress_s = state->progress_s,
                                .target_s = state->target_s,
                                .checkpoint_bytes = state->checkpoint_bytes,
                                .vram_bytes = state->vram_bytes,
                                .gpus = state->gpus,
                                .captured_ns = wall_ns(),
                                .node = state->node};
  std::string head = encode(header).dump();
  if (head.size() >= kHeaderBytes) {
    resume();
    return fail("checkpoint header too large");
  }
  head.resize(kHeaderBytes, ' ');
  head.back() = '\n';
  const auto temporary = fs::path{*out}.concat(std::format(".{}.tmp", getpid()));
  const int target = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  const int source = open(payload.c_str(), O_RDONLY | O_CLOEXEC);
  Status result;
  if (target < 0 || source < 0) {
    result = failure(std::format("cannot open GPU checkpoint files: {}", std::strerror(errno)));
  } else {
    std::size_t offset = 0;
    while (offset < head.size()) {
      const auto count = write(target, head.data() + offset, head.size() - offset);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0) {
        result = failure("checkpoint header write failed");
        break;
      }
      offset += static_cast<std::size_t>(count);
    }
    if (result)
      result = copy_bytes(source, target, payload_bytes);
    if (result && fsync(target) != 0)
      result = failure("checkpoint sync failed");
  }
  if (target >= 0)
    close(target);
  if (source >= 0)
    close(source);
  if (result) {
    fs::rename(temporary, *out, ec);
    if (ec)
      result = failure(std::format("checkpoint rename: {}", ec.message()));
  }
  if (!result)
    fs::remove(temporary, ec);
  else
    append_log(*state, "checkpoint");
  resume();
  return result ? 0 : fail(result.error().message);
}

[[nodiscard]] int restore(const Args &args) {
  const auto directory = args.get<std::string>("state");
  const auto in = args.get<std::string>("in");
  if (!directory || !in)
    return fail("--state and --in are required");
  if (auto environment = physical_environment(); !environment)
    return fail(environment.error().message);
  auto header = read_checkpoint(*in);
  if (!header)
    return fail(header.error().message);
  const fs::path root{*directory};
  std::error_code ec;
  fs::create_directories(root, ec);
  JobState state;
  if (auto existing = load_state(root))
    state = *existing;
  if (alive(state.pid)) {
    kill(static_cast<pid_t>(state.pid), SIGKILL);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (alive(state.pid) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    if (alive(state.pid))
      return fail("could not stop the previous GPU process before restore");
  }
  state.instance = header->instance;
  state.rank = header->rank;
  state.progress_s = header->progress_s;
  state.target_s = header->target_s;
  state.checkpoint_bytes = header->checkpoint_bytes;
  state.vram_bytes = header->vram_bytes;
  state.gpus = header->gpus;
  state.external_load = header->external_load;
  state.node = node_name(args);
  state.restored = true;
  state.pid = 0;
  state.status = "restoring";
  state.error.clear();
  state.log_dir = args.get<std::string>("log-dir").value_or(state.log_dir);
  if (state.log_dir.empty())
    if (const char *env = std::getenv("RNBT_LOG_DIR"); env && *env)
      state.log_dir = env;
  if (auto resources = validate_resources(state); !resources)
    return fail_state(root, state, resources.error().message);
  const auto temporary = fs::path{payload_file(root)}.concat(std::format(".{}.tmp", getpid()));
  const int source = open(in->c_str(), O_RDONLY | O_CLOEXEC);
  const int target = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  Status result;
  if (source < 0 || target < 0 || lseek(source, static_cast<off_t>(kHeaderBytes), SEEK_SET) < 0)
    result = failure(std::format("cannot open restore payload: {}", std::strerror(errno)));
  else
    result = copy_bytes(source, target, state.checkpoint_bytes - kHeaderBytes);
  if (result && fsync(target) != 0)
    result = failure("restore payload sync failed");
  if (source >= 0)
    close(source);
  if (target >= 0)
    close(target);
  if (result) {
    fs::rename(temporary, payload_file(root), ec);
    if (ec)
      result = failure(std::format("restore payload rename: {}", ec.message()));
  }
  if (!result) {
    fs::remove(temporary, ec);
    return fail_state(root, state, result.error().message);
  }
  Gpu preflight;
  if (auto ready = preflight.initialize(std::max<std::uint64_t>(1048576, state.gpus), state.gpus);
      !ready)
    return fail_state(root, state, ready.error().message);
  auto samples = preflight.sample();
  if (!samples)
    return fail_state(root, state, samples.error().message);
  state.gpu_samples = std::move(*samples);
  state.status = "restored";
  if (auto saved = save_state(root, state); !saved)
    return fail(saved.error().message);
  append_log(state, "restore");
  return 0;
}

[[nodiscard]] int probe(const Args &args) {
  if (auto environment = physical_environment(); !environment)
    return fail(environment.error().message);
  const auto bytes = args.get<std::uint64_t>("vram-bytes").value_or(1048576);
  const auto gpus = args.get<std::uint32_t>("gpus").value_or(1);
  if (args.contains("vram-bytes") && !args.get<std::uint64_t>("vram-bytes"))
    return fail("--vram-bytes must be a valid unsigned integer");
  if (args.contains("gpus") && !args.get<std::uint32_t>("gpus"))
    return fail("--gpus must be a valid unsigned integer");
  Gpu gpu;
  if (auto ready = gpu.initialize(bytes, gpus); !ready)
    return fail(ready.error().message);
  auto samples = gpu.sample();
  if (!samples)
    return fail(samples.error().message);
  std::cout << Json{{"gpu_backend", "cuda"},
                    {"telemetry_source", "nvml"},
                    {"gpu_samples", *samples}}
                   .dump(2)
            << '\n';
  return 0;
}

[[nodiscard]] std::vector<fs::path> discover(const std::vector<std::string> &roots) {
  std::vector<fs::path> found;
  for (const auto &root : roots) {
    std::error_code ec;
    if (!fs::exists(root, ec))
      continue;
    for (auto it =
             fs::recursive_directory_iterator{root, fs::directory_options::skip_permission_denied,
                                              ec};
         it != fs::recursive_directory_iterator{}; it.increment(ec)) {
      if (ec)
        break;
      if (it.depth() > 4)
        it.disable_recursion_pending();
      if (it->path().filename() == "job.json")
        found.push_back(it->path().parent_path());
    }
  }
  return found;
}

[[nodiscard]] int terminate_all(const Args &args, std::optional<std::string> only, int signal) {
  for (const auto &directory : discover(args.all("root"))) {
    auto state = load_state(directory);
    if (!state || (only && state->instance != *only))
      continue;
    if (alive(state->pid)) {
      kill(static_cast<pid_t>(state->pid), SIGCONT);
      kill(static_cast<pid_t>(state->pid), signal);
      for (int attempt = 0; attempt < 500 && alive(state->pid); ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
      if (alive(state->pid))
        kill(static_cast<pid_t>(state->pid), SIGKILL);
      if (signal == SIGKILL) {
        if (auto latest = load_state(directory))
          state = latest;
        state->status = "killed";
        (void)save_state(directory, *state);
        append_log(*state, "kill");
      }
    }
  }
  return 0;
}

}

Expected<JobState> load_state(const fs::path &directory) {
  auto json = read_json_file(state_file(directory));
  if (!json)
    return std::unexpected{json.error()};
  return try_decode<JobState>(*json);
}

Expected<CheckpointHeader> read_checkpoint(const fs::path &file) {
  std::ifstream in{file, std::ios::binary};
  if (!in)
    return failure(std::format("cannot open {}", file.string()));
  std::string head(kHeaderBytes, '\0');
  in.read(head.data(), static_cast<std::streamsize>(head.size()));
  head.resize(static_cast<std::size_t>(in.gcount()));
  const auto json = Json::parse(head, nullptr, false);
  if (json.is_discarded())
    return failure(std::format("{} is not an rnbt-job checkpoint", file.string()));
  auto header = try_decode<CheckpointHeader>(json);
  if (!header)
    return header;
  if (header->magic != CheckpointHeader{}.magic || header->gpu_backend != "cuda")
    return failure(std::format("{} is not a real CUDA GPU checkpoint", file.string()));
  if (header->checkpoint_bytes <= kHeaderBytes || header->gpus == 0 || header->vram_bytes == 0 ||
      header->checkpoint_bytes - kHeaderBytes > header->vram_bytes)
    return failure("checkpoint contains invalid GPU resource sizes");
  std::error_code ec;
  const auto size = fs::file_size(file, ec);
  if (ec || size != header->checkpoint_bytes)
    return failure("GPU checkpoint file is truncated or has an unexpected size");
  return header;
}

int run(std::span<char *const> arguments) {
  if (arguments.size() < 2)
    return fail(
        "usage: rnbt-job <probe|start|pause|resume|stop|checkpoint|restore|status|kill-all|stop-instance> ...");
  const std::string_view command{arguments[1]};
  const Args args{arguments.subspan(2)};
  using Handler = int (*)(const Args &);
  static constexpr std::array<std::pair<std::string_view, Handler>, 10> table{{
      {"probe", [](const Args &a) { return probe(a); }},
      {"start", [](const Args &a) { return start(a); }},
      {"pause", [](const Args &a) { return signal_job(a, "pause"); }},
      {"resume", [](const Args &a) { return signal_job(a, "resume"); }},
      {"stop", [](const Args &a) { return signal_job(a, "stop"); }},
      {"checkpoint", [](const Args &a) { return checkpoint(a); }},
      {"restore", [](const Args &a) { return restore(a); }},
      {"status",
       [](const Args &a) {
         auto state = load_state(a.get<std::string>("state").value_or("."));
         if (!state)
           return fail(state.error().message);
         std::cout << encode(*state).dump(2) << '\n';
         return 0;
       }},
      {"kill-all", [](const Args &a) { return terminate_all(a, std::nullopt, SIGKILL); }},
      {"stop-instance",
       [](const Args &a) { return terminate_all(a, a.get<std::string>("instance"), SIGTERM); }},
  }};
  const auto found =
      std::ranges::find(table, command, &std::pair<std::string_view, Handler>::first);
  if (found == table.end())
    return fail(std::format("unknown command '{}'", command));
  return found->second(args);
}

}
