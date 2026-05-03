#include "rnbt/shell.hpp"

#include <csignal>
#include <gtest/gtest.h>
#include <pthread.h>

using namespace rnbt;
using namespace std::chrono_literals;

TEST(Shell, RendersKnownPlaceholdersOnly) {
  const Vars vars{{"node", "node-001"}, {"dir", "/var/x"}};
  EXPECT_EQ(*render("stop {node} in {dir}", vars), "stop node-001 in /var/x");
  EXPECT_FALSE(render("stop {missing}", vars));
  EXPECT_FALSE(render("stop {node", vars));
}

TEST(Shell, QuotesForRemoteShells) {
  EXPECT_EQ(shell_quote("it's"), "'it'\\''s'");
  const auto echoed = run_command("printf '%s\\n' " + shell_quote("it's"), 5s);
  EXPECT_TRUE(echoed.ok());
  EXPECT_EQ(echoed.output, "it's\n");
}

TEST(Shell, TimesOutAndRunsInParallel) {
  const auto slow = run_command("sleep 5", 200ms);
  EXPECT_TRUE(slow.timed_out);
  EXPECT_FALSE(slow.ok());
  const std::vector<Command> commands{{"a", "sleep 0.3; echo a"},
                                      {"b", "sleep 0.3; echo b"},
                                      {"c", "sleep 0.3; echo c"},
                                      {"d", "exit 3"}};
  const auto started = std::chrono::steady_clock::now();
  const auto results = run_parallel(commands, 4, 5s);
  EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
  EXPECT_EQ(results[0].output, "a\n");
  EXPECT_EQ(results[3].exit_code, 3);
}

TEST(Shell, ChildCommandsReceiveControlSignalsWhenDriverBlocksThem) {
  sigset_t controls, previous;
  sigemptyset(&controls);
  sigaddset(&controls, SIGTERM);
  ASSERT_EQ(pthread_sigmask(SIG_BLOCK, &controls, &previous), 0);
  const auto result = run_command("trap 'exit 42' TERM; kill -TERM $$; exit 7", 5s);
  EXPECT_EQ(pthread_sigmask(SIG_SETMASK, &previous, nullptr), 0);
  EXPECT_FALSE(result.timed_out);
  EXPECT_EQ(result.exit_code, 42);
}

TEST(Shell, RemoteWrapsCommandsUnlessLocal) {
  Testbed testbed;
  testbed.nodes.push_back(HostConfig{.trace_node = "node-001",
                                     .host = "10.0.0.1",
                                     .ssh = "ubuntu@10.0.0.1",
                                     .node_daemon_address = "10.0.0.1:50052",
                                     .migration_port = 7600,
                                     .vars = {}});
  const Remote remote{testbed};
  const auto command = remote.on_node(testbed.nodes.front(), "echo {node}");
  ASSERT_TRUE(command);
  EXPECT_EQ(command->line,
            "ssh -o BatchMode=yes -o ConnectTimeout=10 ubuntu@10.0.0.1 'echo node-001'");
  testbed.local = true;
  EXPECT_EQ(Remote{testbed}.on_node(testbed.nodes.front(), "echo {node}")->line, "echo node-001");
  const auto addresses =
      Remote{testbed}.on_node(testbed.nodes.front(), "echo {node_daemon_address} {agent_address}");
  ASSERT_TRUE(addresses);
  EXPECT_EQ(addresses->line, "echo 10.0.0.1:50052 10.0.0.1:50052");
}
