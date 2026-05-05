#pragma once

#include "rnbt/replay_config.hpp"

#include <map>
#include <memory>
#include <string>

namespace grpc {
class Channel;
}

namespace reclaimnet::proto {
class RuntimeRegistration;
}

namespace rnbt {

class Channels {
public:
  Channels(const CredentialConfig &credentials, double timeout_s);
  ~Channels();
  Channels(const Channels &) = delete;
  Channels &operator=(const Channels &) = delete;

  [[nodiscard]] Status register_runtime(const std::string &node_daemon,
                                        const reclaimnet::proto::RuntimeRegistration &registration);
  [[nodiscard]] Status reclaim(const std::string &node_daemon, const std::string &node,
                               ReclaimKind kind, Nanos notice, std::int64_t timestamp_ns);
  [[nodiscard]] Expected<bool> healthy(const std::string &node_daemon);
  [[nodiscard]] Status report_reclaim(const std::string &coordinator, const std::string &node,
                                      ReclaimKind kind, Nanos notice, std::int64_t timestamp_ns);

private:
  struct State;
  std::unique_ptr<State> state_;
};

}
