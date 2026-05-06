#pragma once

#include "rnbt/reflect.hpp"

#include <cstdint>
#include <memory>

namespace rnbt::job {

class Gpu {
public:
  Gpu();
  ~Gpu();
  Gpu(const Gpu &) = delete;
  Gpu &operator=(const Gpu &) = delete;

  [[nodiscard]] Status initialize(std::uint64_t bytes, std::uint32_t gpus);
  [[nodiscard]] Status step(std::uint64_t sequence);
  [[nodiscard]] Status run_for(std::chrono::duration<double> busy, std::uint64_t sequence);
  [[nodiscard]] Status write_payload(int fd, std::uint64_t bytes);
  [[nodiscard]] Status read_payload(int fd, std::uint64_t bytes);
  [[nodiscard]] Expected<Json> sample();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}
