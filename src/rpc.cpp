#include "rnbt/rpc.hpp"

#include "reclaimnet/mtls.h"
#include "reclaimnet/security.h"

#include <chrono>
#include <grpcpp/grpcpp.h>
#include <mutex>
#include <reclaimnet.grpc.pb.h>

namespace rnbt {

namespace pb = reclaimnet::proto;

struct Channels::State {
  CredentialConfig credentials;
  std::chrono::milliseconds timeout;
  std::shared_ptr<grpc::ChannelCredentials> channel_credentials;
  std::string jwt_key;
  std::mutex mutex;
  std::map<std::string, std::shared_ptr<grpc::Channel>> channels;

  std::shared_ptr<grpc::Channel> channel(const std::string &address) {
    std::lock_guard lock{mutex};
    auto &slot = channels[address];
    if (!slot) {
      grpc::ChannelArguments arguments;
      if (!credentials.target_name_override.empty())
        arguments.SetSslTargetNameOverride(credentials.target_name_override);
      slot = grpc::CreateCustomChannel(address, channel_credentials, arguments);
    }
    return slot;
  }

  void prepare(grpc::ClientContext &context, std::string_view identity) const {
    context.set_deadline(std::chrono::system_clock::now() + timeout);
    if (jwt_key.empty())
      return;
    const auto issued = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    auto token = reclaimnet::issue_jwt_rs256(
        {std::string{identity}, std::string{identity}, issued, issued + 120}, jwt_key);
    if (token)
      context.AddMetadata("authorization", "Bearer " + token.value());
  }
};

namespace {

[[nodiscard]] Status status_of(const grpc::Status &status, std::string_view what) {
  if (status.ok())
    return {};
  return failure(std::format("{}: {} ({})", what, status.error_message(),
                             static_cast<int>(status.error_code())));
}

[[nodiscard]] std::string type_of(ReclaimKind kind) {
  return kind == ReclaimKind::emergency ? "EMERGENCY" : "SCHEDULED";
}

}

Channels::Channels(const CredentialConfig &credentials, double timeout_s)
    : state_{std::make_unique<State>()} {
  state_->credentials = credentials;
  state_->timeout = std::chrono::milliseconds{static_cast<long long>(timeout_s * 1000.0)};
  state_->channel_credentials = reclaimnet::insecure_channel_credentials();
  if (credentials.tls) {
    auto bundle = reclaimnet::load_tls_bundle(credentials.cert, credentials.key, credentials.ca);
    if (!bundle)
      throw std::runtime_error{std::format("TLS bundle: {}", bundle.error().message)};
    auto made = reclaimnet::make_client_credentials(bundle.value());
    if (!made)
      throw std::runtime_error{std::format("TLS credentials: {}", made.error().message)};
    state_->channel_credentials = made.value();
  }
  if (!credentials.jwt_private_key.empty()) {
    auto key = reclaimnet::read_pem_file(credentials.jwt_private_key);
    if (!key)
      throw std::runtime_error{std::format("JWT key: {}", key.error().message)};
    state_->jwt_key = key.value();
  }
}

Channels::~Channels() = default;

Status Channels::register_runtime(const std::string &node_daemon,
                                  const pb::RuntimeRegistration &registration) {
  grpc::ClientContext context;
  state_->prepare(context, "rnbt-replay");
  pb::Empty response;
  return status_of(pb::NodeAgentService::NewStub(state_->channel(node_daemon))
                       ->RegisterRuntime(&context, registration, &response),
                   std::format("RegisterRuntime {}", node_daemon));
}

Status Channels::reclaim(const std::string &node_daemon, const std::string &node, ReclaimKind kind,
                         Nanos notice, std::int64_t timestamp_ns) {
  grpc::ClientContext context;
  state_->prepare(context, "rnbt-replay");
  pb::ReclaimEvent request;
  request.set_node_id(node);
  request.set_tau_seconds(seconds_of(notice));
  request.set_timestamp(timestamp_ns);
  request.set_type(type_of(kind));
  pb::Empty response;
  return status_of(pb::NodeAgentService::NewStub(state_->channel(node_daemon))
                       ->Reclaim(&context, request, &response),
                   std::format("Reclaim {}", node_daemon));
}

Expected<bool> Channels::healthy(const std::string &node_daemon) {
  grpc::ClientContext context;
  state_->prepare(context, "rnbt-replay");
  pb::Empty request;
  pb::HealthStatus response;
  const auto status = pb::NodeAgentService::NewStub(state_->channel(node_daemon))
                          ->HealthCheck(&context, request, &response);
  if (!status.ok())
    return failure(status.error_message());
  return response.healthy();
}

Status Channels::report_reclaim(const std::string &coordinator, const std::string &node,
                                ReclaimKind kind, Nanos notice, std::int64_t timestamp_ns) {
  grpc::ClientContext context;
  state_->prepare(context, node);
  pb::ReclaimEvent request;
  request.set_node_id(node);
  request.set_tau_seconds(seconds_of(notice));
  request.set_timestamp(timestamp_ns);
  request.set_type(type_of(kind));
  pb::Empty response;
  return status_of(pb::CoordinatorService::NewStub(state_->channel(coordinator))
                       ->ReportReclaim(&context, request, &response),
                   std::format("ReportReclaim {}", node));
}

}
