#include "rnbt/gpu.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rnbt::job {

namespace {

struct NvmlDevice;
using NvmlHandle = NvmlDevice *;
struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};
struct NvmlUtilization {
  unsigned int gpu;
  unsigned int memory;
};

constexpr int kCudaHostToDevice = 1;
constexpr int kCudaDeviceToHost = 2;
constexpr std::size_t kTransferChunkBytes = 4U << 20;

template <class Function> Status load_symbol(void *library, const char *name, Function &target) {
  dlerror();
  target = reinterpret_cast<Function>(dlsym(library, name));
  const char *error = dlerror();
  if (error || !target)
    return failure(std::format("required GPU symbol {} is unavailable: {}", name,
                               error ? error : "null symbol"));
  return {};
}

Status write_all(int fd, const char *data, std::size_t bytes) {
  std::size_t offset = 0;
  while (offset < bytes) {
    const auto written = write(fd, data + offset, bytes - offset);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      return failure(std::format("write GPU checkpoint payload: {}", std::strerror(errno)));
    }
    if (written == 0)
      return failure("write GPU checkpoint payload made no progress");
    offset += static_cast<std::size_t>(written);
  }
  return {};
}

Status read_all(int fd, char *data, std::size_t bytes) {
  std::size_t offset = 0;
  while (offset < bytes) {
    const auto received = read(fd, data + offset, bytes - offset);
    if (received < 0) {
      if (errno == EINTR)
        continue;
      return failure(std::format("read GPU checkpoint payload: {}", std::strerror(errno)));
    }
    if (received == 0)
      return failure("GPU checkpoint payload is truncated");
    offset += static_cast<std::size_t>(received);
  }
  return {};
}

}

struct Gpu::Impl {
  struct Allocation {
    int index{};
    void *pointer{};
    std::size_t bytes{};
    NvmlHandle nvml{};
    std::string pci_bus_id;
    std::string uuid;
    std::string name;
  };

  void *cuda_library{};
  void *nvml_library{};
  bool nvml_initialized{};
  std::uint64_t total_bytes{};
  std::vector<Allocation> allocations;

  int (*cuda_count)(int *){};
  int (*cuda_set)(int){};
  int (*cuda_memory_info)(std::size_t *, std::size_t *){};
  int (*cuda_malloc)(void **, std::size_t){};
  int (*cuda_memset)(void *, int, std::size_t){};
  int (*cuda_free)(void *){};
  int (*cuda_synchronize)(){};
  int (*cuda_memcpy)(void *, const void *, std::size_t, int){};
  int (*cuda_pci_bus_id)(char *, int, int){};
  const char *(*cuda_error_string)(int){};

  int (*nvml_init)(){};
  int (*nvml_shutdown)(){};
  int (*nvml_handle_by_pci)(const char *, NvmlHandle *){};
  int (*nvml_uuid)(NvmlHandle, char *, unsigned int){};
  int (*nvml_name)(NvmlHandle, char *, unsigned int){};
  int (*nvml_memory_info)(NvmlHandle, NvmlMemory *){};
  int (*nvml_utilization)(NvmlHandle, NvmlUtilization *){};
  const char *(*nvml_error_string)(int){};

  ~Impl() {
    for (const auto &allocation : allocations) {
      if (allocation.pointer && cuda_set && cuda_free && cuda_set(allocation.index) == 0)
        (void)cuda_free(allocation.pointer);
    }
    if (nvml_initialized && nvml_shutdown)
      (void)nvml_shutdown();
    if (nvml_library)
      dlclose(nvml_library);
    if (cuda_library)
      dlclose(cuda_library);
  }

  [[nodiscard]] Status cuda_status(int code, std::string_view operation, int device = -1) const {
    if (code == 0)
      return {};
    const char *detail = cuda_error_string ? cuda_error_string(code) : nullptr;
    if (device >= 0)
      return failure(std::format("GPU {}: {} failed: CUDA error {} ({})", device, operation, code,
                                 detail ? detail : "unknown error"));
    return failure(std::format("{} failed: CUDA error {} ({})", operation, code,
                               detail ? detail : "unknown error"));
  }

  [[nodiscard]] Status nvml_status(int code, std::string_view operation, int device = -1) const {
    if (code == 0)
      return {};
    const char *detail = nvml_error_string ? nvml_error_string(code) : nullptr;
    if (device >= 0)
      return failure(std::format("GPU {}: {} failed: NVML error {} ({})", device, operation, code,
                                 detail ? detail : "unknown error"));
    return failure(std::format("{} failed: NVML error {} ({})", operation, code,
                               detail ? detail : "unknown error"));
  }

  Status load() {
    for (const char *name :
         {"libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"}) {
      cuda_library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (cuda_library)
        break;
    }
    if (!cuda_library) {
      const char *error = dlerror();
      return failure(std::format("CUDA runtime is required; cannot load libcudart.so "
                                 "(.13, .12 or .11.0): {}",
                                 error ? error : "library not found"));
    }
    if (auto status = load_symbol(cuda_library, "cudaGetDeviceCount", cuda_count); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaSetDevice", cuda_set); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaMemGetInfo", cuda_memory_info); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaMalloc", cuda_malloc); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaMemset", cuda_memset); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaFree", cuda_free); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaDeviceSynchronize", cuda_synchronize); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaMemcpy", cuda_memcpy); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaDeviceGetPCIBusId", cuda_pci_bus_id); !status)
      return status;
    if (auto status = load_symbol(cuda_library, "cudaGetErrorString", cuda_error_string); !status)
      return status;

    nvml_library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!nvml_library) {
      const char *error = dlerror();
      return failure(std::format("real GPU telemetry requires libnvidia-ml.so.1: {}",
                                 error ? error : "library not found"));
    }
    if (auto status = load_symbol(nvml_library, "nvmlInit_v2", nvml_init); !status)
      return status;
    if (auto status = load_symbol(nvml_library, "nvmlShutdown", nvml_shutdown); !status)
      return status;
    if (auto status =
            load_symbol(nvml_library, "nvmlDeviceGetHandleByPciBusId_v2", nvml_handle_by_pci);
        !status)
      return status;
    if (auto status = load_symbol(nvml_library, "nvmlDeviceGetUUID", nvml_uuid); !status)
      return status;
    if (auto status = load_symbol(nvml_library, "nvmlDeviceGetName", nvml_name); !status)
      return status;
    if (auto status = load_symbol(nvml_library, "nvmlDeviceGetMemoryInfo", nvml_memory_info);
        !status)
      return status;
    if (auto status = load_symbol(nvml_library, "nvmlDeviceGetUtilizationRates", nvml_utilization);
        !status)
      return status;
    if (auto status = load_symbol(nvml_library, "nvmlErrorString", nvml_error_string); !status)
      return status;
    if (auto status = nvml_status(nvml_init(), "nvmlInit_v2"); !status)
      return status;
    nvml_initialized = true;
    return {};
  }

  Status update(std::uint64_t sequence) {
    for (const auto &allocation : allocations) {
      if (auto status = cuda_status(cuda_set(allocation.index), "cudaSetDevice", allocation.index);
          !status)
        return status;
      const int pattern =
          1 + static_cast<int>(
                  (sequence % 255 + static_cast<std::uint64_t>(allocation.index) % 255) % 255);
      if (auto status = cuda_status(cuda_memset(allocation.pointer, pattern, allocation.bytes),
                                    "cudaMemset", allocation.index);
          !status)
        return status;
      if (auto status = cuda_status(cuda_synchronize(), "cudaDeviceSynchronize", allocation.index);
          !status)
        return status;
    }
    return {};
  }

  [[nodiscard]] Expected<Json> telemetry() const {
    Json result = Json::array();
    for (const auto &allocation : allocations) {
      NvmlMemory memory{};
      NvmlUtilization utilization{};
      if (auto status = nvml_status(nvml_memory_info(allocation.nvml, &memory),
                                    "nvmlDeviceGetMemoryInfo", allocation.index);
          !status)
        return std::unexpected{status.error()};
      if (auto status = nvml_status(nvml_utilization(allocation.nvml, &utilization),
                                    "nvmlDeviceGetUtilizationRates", allocation.index);
          !status)
        return std::unexpected{status.error()};
      result.push_back(Json{{"cuda_index", allocation.index},
                            {"pci_bus_id", allocation.pci_bus_id},
                            {"uuid", allocation.uuid},
                            {"name", allocation.name},
                            {"allocated_bytes", allocation.bytes},
                            {"memory_total_bytes", memory.total},
                            {"memory_free_bytes", memory.free},
                            {"memory_used_bytes", memory.used},
                            {"utilization_gpu_percent", utilization.gpu},
                            {"utilization_memory_percent", utilization.memory}});
    }
    return result;
  }

  Status transfer(int fd, std::uint64_t bytes, bool to_file) {
    if (bytes > total_bytes)
      return failure(std::format("checkpoint payload requests {} bytes, but only {} bytes "
                                 "are allocated on real GPUs",
                                 bytes, total_bytes));
    if (fd < 0)
      return failure("GPU checkpoint payload requires a valid file descriptor");
    if (bytes == 0)
      return {};
    std::vector<char> buffer(
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes, kTransferChunkBytes)));
    std::uint64_t remaining = bytes;
    for (const auto &allocation : allocations) {
      if (remaining == 0)
        break;
      if (auto status = cuda_status(cuda_set(allocation.index), "cudaSetDevice", allocation.index);
          !status)
        return status;
      const auto device_bytes =
          static_cast<std::size_t>(std::min<std::uint64_t>(remaining, allocation.bytes));
      for (std::size_t offset = 0; offset < device_bytes;) {
        const auto count = std::min(buffer.size(), device_bytes - offset);
        auto *device = static_cast<std::byte *>(allocation.pointer) + offset;
        if (to_file) {
          if (auto status =
                  cuda_status(cuda_memcpy(buffer.data(), device, count, kCudaDeviceToHost),
                              "cudaMemcpy device-to-host", allocation.index);
              !status)
            return status;
          if (auto status = write_all(fd, buffer.data(), count); !status)
            return status;
        } else {
          if (auto status = read_all(fd, buffer.data(), count); !status)
            return status;
          if (auto status =
                  cuda_status(cuda_memcpy(device, buffer.data(), count, kCudaHostToDevice),
                              "cudaMemcpy host-to-device", allocation.index);
              !status)
            return status;
        }
        offset += count;
      }
      if (auto status = cuda_status(cuda_synchronize(), "cudaDeviceSynchronize", allocation.index);
          !status)
        return status;
      remaining -= device_bytes;
    }
    return {};
  }
};

Gpu::Gpu() = default;
Gpu::~Gpu() = default;

Status Gpu::initialize(std::uint64_t bytes, std::uint32_t gpus) {
  if (impl_)
    return failure("real GPU allocations are already initialized");
  if (gpus == 0)
    return failure("a real GPU job requires at least one GPU");
  if (bytes < gpus)
    return failure(std::format("real GPU jobs require at least one allocated byte per GPU; "
                               "requested {} bytes across {} GPUs",
                               bytes, gpus));
  if (bytes > std::numeric_limits<std::size_t>::max())
    return failure("requested GPU memory exceeds this platform's addressable size");
  auto candidate = std::make_unique<Impl>();
  if (auto status = candidate->load(); !status)
    return status;
  int count = 0;
  if (auto status = candidate->cuda_status(candidate->cuda_count(&count), "cudaGetDeviceCount");
      !status)
    return status;
  if (count <= 0 || static_cast<std::uint64_t>(count) < gpus)
    return failure(
        std::format("{} real CUDA GPUs are required, but only {} are visible", gpus, count));

  struct Available {
    int index;
    std::uint64_t free_bytes;
    NvmlHandle nvml;
    std::string pci_bus_id;
  };
  std::vector<Available> available;
  available.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    std::array<char, 64> bus_id{};
    if (auto status = candidate->cuda_status(
            candidate->cuda_pci_bus_id(bus_id.data(), static_cast<int>(bus_id.size()), index),
            "cudaDeviceGetPCIBusId", index);
        !status)
      return status;
    NvmlHandle nvml{};
    if (auto status = candidate->nvml_status(candidate->nvml_handle_by_pci(bus_id.data(), &nvml),
                                             "nvmlDeviceGetHandleByPciBusId_v2", index);
        !status)
      return status;
    NvmlMemory memory{};
    if (auto status = candidate->nvml_status(candidate->nvml_memory_info(nvml, &memory),
                                             "nvmlDeviceGetMemoryInfo", index);
        !status)
      return status;
    available.push_back({index, memory.free, nvml, bus_id.data()});
  }
  std::ranges::sort(available, [](const Available &left, const Available &right) {
    return left.free_bytes != right.free_bytes ? left.free_bytes > right.free_bytes
                                               : left.index < right.index;
  });
  candidate->allocations.reserve(gpus);
  for (std::uint32_t ordinal = 0; ordinal < gpus; ++ordinal) {
    const auto &selected = available[ordinal];
    const int index = selected.index;
    const auto required = static_cast<std::size_t>(bytes / gpus + (ordinal < bytes % gpus ? 1 : 0));
    if (selected.free_bytes < required)
      return failure(std::format("GPU {}: insufficient device memory: requested {} bytes, "
                                 "available {} bytes; allocation will not be reduced",
                                 index, required, selected.free_bytes));
    for (const auto &allocation : candidate->allocations) {
      if (allocation.pci_bus_id == selected.pci_bus_id)
        return failure(std::format("GPU {}: PCI bus {} was already selected; jobs require "
                                   "distinct physical GPUs",
                                   index, selected.pci_bus_id));
    }
    if (auto status = candidate->cuda_status(candidate->cuda_set(index), "cudaSetDevice", index);
        !status)
      return status;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (auto status = candidate->cuda_status(candidate->cuda_memory_info(&free_bytes, &total_bytes),
                                             "cudaMemGetInfo", index);
        !status)
      return status;
    if (free_bytes < required)
      return failure(
          std::format("GPU {}: insufficient device memory after CUDA initialization: "
                      "requested {} bytes, available {} bytes; allocation will not be reduced",
                      index, required, free_bytes));
    candidate->allocations.emplace_back();
    auto &allocation = candidate->allocations.back();
    allocation.index = index;
    allocation.bytes = required;
    allocation.nvml = selected.nvml;
    allocation.pci_bus_id = selected.pci_bus_id;
    if (auto status = candidate->cuda_status(candidate->cuda_malloc(&allocation.pointer, required),
                                             std::format("cudaMalloc({} bytes)", required), index);
        !status)
      return status;

    std::array<char, 96> uuid{};
    std::array<char, 96> name{};
    if (auto status =
            candidate->nvml_status(candidate->nvml_uuid(allocation.nvml, uuid.data(),
                                                        static_cast<unsigned int>(uuid.size())),
                                   "nvmlDeviceGetUUID", index);
        !status)
      return status;
    if (auto status =
            candidate->nvml_status(candidate->nvml_name(allocation.nvml, name.data(),
                                                        static_cast<unsigned int>(name.size())),
                                   "nvmlDeviceGetName", index);
        !status)
      return status;
    allocation.uuid = uuid.data();
    allocation.name = name.data();
  }
  candidate->total_bytes = bytes;
  if (auto status = candidate->update(0); !status)
    return status;
  if (auto telemetry = candidate->telemetry(); !telemetry)
    return std::unexpected{telemetry.error()};
  impl_ = std::move(candidate);
  return {};
}

Status Gpu::step(std::uint64_t sequence) {
  if (!impl_)
    return failure("real GPUs have not been initialized");
  return impl_->update(sequence);
}

Status Gpu::run_for(std::chrono::duration<double> busy, std::uint64_t sequence) {
  if (!impl_)
    return failure("real GPUs have not been initialized");
  if (!std::isfinite(busy.count()) || busy.count() < 0)
    return failure("GPU busy-time budget must be finite and nonnegative");
  if (busy.count() == 0)
    return {};

  std::vector<Status> statuses(impl_->allocations.size());
  const auto run_device = [this, busy, sequence, &statuses](std::size_t ordinal) {
    auto &status = statuses[ordinal];
    try {
      const auto &allocation = impl_->allocations[ordinal];
      status =
          impl_->cuda_status(impl_->cuda_set(allocation.index), "cudaSetDevice", allocation.index);
      if (!status)
        return;
      const auto deadline = std::chrono::steady_clock::now() + busy;
      std::uint64_t iteration = 0;
      do {
        const int pattern =
            1 + static_cast<int>((sequence % 255 + iteration % 255 +
                                  static_cast<std::uint64_t>(allocation.index) % 255) %
                                 255);
        status =
            impl_->cuda_status(impl_->cuda_memset(allocation.pointer, pattern, allocation.bytes),
                               "cudaMemset workload", allocation.index);
        if (!status)
          return;
        status = impl_->cuda_status(impl_->cuda_synchronize(), "cudaDeviceSynchronize workload",
                                    allocation.index);
        if (!status)
          return;
        ++iteration;
      } while (std::chrono::steady_clock::now() < deadline);
    } catch (const std::exception &error) {
      status = failure(std::format("GPU workload worker failed: {}", error.what()));
    }
  };
  if (impl_->allocations.size() == 1) {
    run_device(0);
  } else {
    try {
      std::vector<std::jthread> workers;
      workers.reserve(impl_->allocations.size());
      for (std::size_t ordinal = 0; ordinal < impl_->allocations.size(); ++ordinal)
        workers.emplace_back(run_device, ordinal);
    } catch (const std::exception &error) {
      return failure(std::format("cannot launch GPU workload workers: {}", error.what()));
    }
  }
  for (const auto &status : statuses) {
    if (!status)
      return status;
  }
  return {};
}

Status Gpu::write_payload(int fd, std::uint64_t bytes) {
  if (!impl_)
    return failure("cannot checkpoint before real GPUs are initialized");
  return impl_->transfer(fd, bytes, true);
}

Status Gpu::read_payload(int fd, std::uint64_t bytes) {
  if (!impl_)
    return failure("cannot restore before real GPUs are initialized");
  return impl_->transfer(fd, bytes, false);
}

Expected<Json> Gpu::sample() {
  if (!impl_)
    return failure("cannot sample before real GPUs are initialized");
  return impl_->telemetry();
}

}
