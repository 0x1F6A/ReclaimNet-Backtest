cmake_minimum_required(VERSION 3.25)
project(ReclaimNetBacktest VERSION 2.0.0 LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(RNBT_WITH_DRIVER "Build rnbt-replay against the ReclaimNet gRPC protocol" ON)
option(RNBT_BUILD_TESTS "Build the backtest test suite" ON)
set(RECLAIMNET_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ICNP'26_Code"
    CACHE PATH "ReclaimNet source tree deployed on the testbed")

find_package(nlohmann_json 3.11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

function(rnbt_defaults target)
  target_compile_features(${target} PUBLIC cxx_std_23)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
endfunction()

add_library(rnbt_base STATIC
  src/calendar.cpp
  src/jsonl.cpp
  src/scenario.cpp
  src/replay_config.cpp
  src/plan.cpp
  src/shell.cpp
  src/stats.cpp
  src/evaluate.cpp
  src/gpu.cpp
  src/workload.cpp
  src/job.cpp)
target_include_directories(rnbt_base PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(rnbt_base PUBLIC nlohmann_json::nlohmann_json Threads::Threads ${CMAKE_DL_LIBS})
rnbt_defaults(rnbt_base)

add_executable(rnbt-job src/main_job.cpp)
target_link_libraries(rnbt-job PRIVATE rnbt_base ${CMAKE_DL_LIBS})
rnbt_defaults(rnbt-job)

add_executable(rnbt-load src/main_load.cpp)
target_link_libraries(rnbt-load PRIVATE rnbt_base)
rnbt_defaults(rnbt-load)

add_executable(rnbt-plan src/main_plan.cpp)
target_link_libraries(rnbt-plan PRIVATE rnbt_base)
rnbt_defaults(rnbt-plan)

add_executable(rnbt-evaluate src/main_evaluate.cpp)
target_link_libraries(rnbt-evaluate PRIVATE rnbt_base)
rnbt_defaults(rnbt-evaluate)

if(RNBT_WITH_DRIVER)
  cmake_path(ABSOLUTE_PATH RECLAIMNET_SOURCE_DIR NORMALIZE OUTPUT_VARIABLE RNBT_SUT_DIR)
  if(NOT EXISTS "${RNBT_SUT_DIR}/proto/reclaimnet.proto")
    message(FATAL_ERROR "RECLAIMNET_SOURCE_DIR does not point at the ReclaimNet tree: ${RNBT_SUT_DIR}")
  endif()
  set(RECLAIMNET_ENABLE_GRPC ON CACHE BOOL "" FORCE)
  set(RECLAIMNET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${RNBT_SUT_DIR}" reclaimnet EXCLUDE_FROM_ALL SYSTEM)
  add_executable(rnbt-replay src/driver.cpp src/rpc.cpp src/main_replay.cpp)
  target_link_libraries(rnbt-replay PRIVATE rnbt_base reclaimnet_common reclaimnet_proto)
  rnbt_defaults(rnbt-replay)
  add_custom_target(testbed_binaries DEPENDS coordinator node_agent rnbt-job rnbt-load rnbt-replay rnbt-evaluate)
endif()

if(RNBT_BUILD_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()
  add_executable(rnbt_tests tests/test_plan.cpp tests/test_job.cpp tests/test_stats.cpp tests/test_shell.cpp
                            tests/test_replay_config.cpp tests/test_gpu.cpp tests/test_workload.cpp
                            tests/test_workload_evaluate.cpp)
  target_link_libraries(rnbt_tests PRIVATE rnbt_base GTest::gtest_main)
  target_compile_definitions(rnbt_tests PRIVATE
    RNBT_JOB_BINARY="$<TARGET_FILE:rnbt-job>"
    RNBT_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_dependencies(rnbt_tests rnbt-job rnbt-load)
  rnbt_defaults(rnbt_tests)
  include(GoogleTest)
  gtest_discover_tests(rnbt_tests DISCOVERY_TIMEOUT 60)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME TraceArchiveImporter
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prepare_trace.py")
    add_test(NAME PhysicalLoadCli
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_load_cli.py"
                     "$<TARGET_FILE:rnbt-load>")
  endif()
endif()
