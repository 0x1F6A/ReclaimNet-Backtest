#include "rnbt/gpu.hpp"

#include <cstdio>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unistd.h>

using namespace rnbt;

TEST(Gpu, RejectsZeroDevicesWithoutHardware) {
  job::Gpu gpu;
  const auto initialized = gpu.initialize(1048576, 0);
  ASSERT_FALSE(initialized);
  EXPECT_FALSE(initialized.error().message.empty());
}

TEST(Gpu, RejectsZeroVramWithoutHardware) {
  job::Gpu gpu;
  const auto initialized = gpu.initialize(0, 1);
  ASSERT_FALSE(initialized);
  EXPECT_FALSE(initialized.error().message.empty());
}

TEST(Gpu, OperationsRequireRealInitialization) {
  job::Gpu gpu;
  EXPECT_FALSE(gpu.step(0));
  EXPECT_FALSE(gpu.run_for(std::chrono::duration<double>{0}, 0));
  EXPECT_FALSE(gpu.sample());
  EXPECT_FALSE(gpu.write_payload(-1, 0));
  EXPECT_FALSE(gpu.read_payload(-1, 0));
}

TEST(Gpu, WorkloadBudgetRequiresRealGpuAndHonorsZeroBusyTime) {
  job::Gpu gpu;
  if (auto initialized = gpu.initialize(1048576, 1); !initialized)
    GTEST_SKIP() << "A real CUDA GPU and NVML are required: " << initialized.error().message;
  EXPECT_FALSE(gpu.run_for(std::chrono::duration<double>{-1}, 0));
  EXPECT_FALSE(
      gpu.run_for(std::chrono::duration<double>{std::numeric_limits<double>::infinity()}, 0));

  using TemporaryFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
  TemporaryFile before{std::tmpfile(), &std::fclose};
  TemporaryFile after{std::tmpfile(), &std::fclose};
  ASSERT_TRUE(before);
  ASSERT_TRUE(after);
  ASSERT_TRUE(gpu.write_payload(fileno(before.get()), 8192));
  ASSERT_TRUE(gpu.run_for(std::chrono::duration<double>{0}, 100));
  ASSERT_TRUE(gpu.write_payload(fileno(after.get()), 8192));
  std::rewind(before.get());
  std::rewind(after.get());
  for (std::size_t index = 0; index < 8192; ++index)
    ASSERT_EQ(std::fgetc(before.get()), std::fgetc(after.get())) << index;
  ASSERT_TRUE(gpu.run_for(std::chrono::duration<double>{0.01}, 2));
  ASSERT_TRUE(gpu.sample());
}

TEST(Gpu, DevicePayloadRoundTripAndTelemetry) {
  job::Gpu gpu;
  if (auto initialized = gpu.initialize(1048576, 1); !initialized)
    GTEST_SKIP() << "A real CUDA GPU and NVML are required: " << initialized.error().message;

  const auto sample = gpu.sample();
  ASSERT_TRUE(sample);
  ASSERT_TRUE(sample->is_array());
  ASSERT_EQ(sample->size(), 1u);
  const auto &device = sample->at(0);
  EXPECT_FALSE(device.at("uuid").get<std::string>().empty());
  EXPECT_FALSE(device.at("pci_bus_id").get<std::string>().empty());
  EXPECT_EQ(device.at("allocated_bytes").get<std::uint64_t>(), 1048576u);
  EXPECT_GE(device.at("memory_total_bytes").get<std::uint64_t>(), 1048576u);

  using TemporaryFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
  TemporaryFile original{std::tmpfile(), &std::fclose};
  TemporaryFile changed{std::tmpfile(), &std::fclose};
  TemporaryFile restored{std::tmpfile(), &std::fclose};
  ASSERT_TRUE(original);
  ASSERT_TRUE(changed);
  ASSERT_TRUE(restored);
  constexpr std::uint64_t payload_bytes = 8192;
  ASSERT_TRUE(gpu.write_payload(fileno(original.get()), payload_bytes));
  ASSERT_TRUE(gpu.step(1));
  ASSERT_TRUE(gpu.write_payload(fileno(changed.get()), payload_bytes));
  ASSERT_EQ(lseek(fileno(original.get()), 0, SEEK_SET), 0);
  ASSERT_TRUE(gpu.read_payload(fileno(original.get()), payload_bytes));
  ASSERT_TRUE(gpu.write_payload(fileno(restored.get()), payload_bytes));

  const auto read_file = [](std::FILE *file) {
    std::rewind(file);
    std::string contents(payload_bytes, '\0');
    contents.resize(std::fread(contents.data(), 1, contents.size(), file));
    return contents;
  };
  const auto first = read_file(original.get());
  const auto second = read_file(changed.get());
  const auto third = read_file(restored.get());
  ASSERT_EQ(first.size(), payload_bytes);
  ASSERT_EQ(second.size(), payload_bytes);
  ASSERT_EQ(third.size(), payload_bytes);
  EXPECT_TRUE(std::ranges::any_of(first, [](char byte) { return byte != '\0'; }));
  EXPECT_NE(first, second) << "A GPU step must change the device-generated payload";
  EXPECT_EQ(first, third) << "Restoring must copy the checkpoint payload back into device memory";
}
