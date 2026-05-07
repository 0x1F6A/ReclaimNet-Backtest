#include "rnbt/jsonl.hpp"

#include <stdexcept>

namespace rnbt {

Generator<Json> json_lines(std::filesystem::path path) {
  std::ifstream input{path};
  if (!input)
    throw std::runtime_error{std::format("cannot open {}", path.string())};
  std::string line;
  std::size_t number = 0;
  while (std::getline(input, line)) {
    ++number;
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    try {
      co_yield Json::parse(line);
    } catch (const Json::parse_error &error) {
      throw std::runtime_error{std::format("{}:{}: {}", path.string(), number, error.what())};
    }
  }
}

Expected<Json> read_json_file(const std::filesystem::path &path) {
  std::ifstream input{path};
  if (!input)
    return failure(std::format("cannot open {}", path.string()));
  try {
    return Json::parse(input);
  } catch (const Json::exception &error) {
    return failure(std::format("{}: {}", path.string(), error.what()));
  }
}

JsonlSink::JsonlSink(const std::filesystem::path &path)
    : out_{std::make_unique<std::ofstream>(path, std::ios::trunc)} {
  if (!*out_)
    throw std::runtime_error{std::format("cannot write {}", path.string())};
}

void JsonlSink::write_json(const Json &value) {
  if (!out_)
    return;
  *out_ << value.dump() << '\n';
  ++lines_;
}

void JsonlSink::flush() {
  if (out_)
    out_->flush();
}

Status write_json_file(const std::filesystem::path &path, const Json &value) {
  std::ofstream out{path, std::ios::trunc};
  if (!out)
    return failure(std::format("cannot write {}", path.string()));
  out << value.dump(2) << '\n';
  return {};
}

}
