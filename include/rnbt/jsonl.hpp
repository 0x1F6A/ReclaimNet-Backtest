#pragma once

#include "rnbt/reflect.hpp"

#include <coroutine>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ranges>

namespace rnbt {

template <class T> class Generator {
public:
  struct promise_type {
    std::optional<T> current;
    std::exception_ptr error;

    Generator get_return_object() noexcept {
      return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() const noexcept { return {}; }
    std::suspend_always final_suspend() const noexcept { return {}; }
    template <std::convertible_to<T> U> std::suspend_always yield_value(U &&value) {
      current.emplace(std::forward<U>(value));
      return {};
    }
    void return_void() const noexcept {}
    void unhandled_exception() noexcept { error = std::current_exception(); }
    template <class U> std::suspend_never await_transform(U &&) = delete;
  };

  class iterator {
  public:
    using iterator_concept = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;

    iterator() = default;
    explicit iterator(std::coroutine_handle<promise_type> handle) : handle_{handle} { advance(); }
    iterator(iterator &&) noexcept = default;
    iterator &operator=(iterator &&) noexcept = default;

    [[nodiscard]] T &operator*() const { return *handle_.promise().current; }
    iterator &operator++() {
      advance();
      return *this;
    }
    void operator++(int) { advance(); }
    [[nodiscard]] friend bool operator==(const iterator &it, std::default_sentinel_t) noexcept {
      return !it.handle_ || it.handle_.done();
    }

  private:
    void advance() {
      handle_.promise().current.reset();
      handle_.resume();
      if (auto error = std::exchange(handle_.promise().error, nullptr))
        std::rethrow_exception(error);
    }
    std::coroutine_handle<promise_type> handle_{};
  };

  explicit Generator(std::coroutine_handle<promise_type> handle) noexcept : handle_{handle} {}
  Generator(Generator &&other) noexcept : handle_{std::exchange(other.handle_, {})} {}
  Generator &operator=(Generator &&other) noexcept {
    if (this != &other) {
      if (handle_)
        handle_.destroy();
      handle_ = std::exchange(other.handle_, {});
    }
    return *this;
  }
  Generator(const Generator &) = delete;
  Generator &operator=(const Generator &) = delete;
  ~Generator() {
    if (handle_)
      handle_.destroy();
  }

  [[nodiscard]] iterator begin() { return iterator{handle_}; }
  [[nodiscard]] std::default_sentinel_t end() const noexcept { return {}; }

private:
  std::coroutine_handle<promise_type> handle_;
};

[[nodiscard]] Generator<Json> json_lines(std::filesystem::path path);

template <class T> [[nodiscard]] Generator<T> records(std::filesystem::path path) {
  for (auto &line : json_lines(std::move(path)))
    co_yield decode<T>(line);
}

[[nodiscard]] Expected<Json> read_json_file(const std::filesystem::path &path);

class JsonlSink {
public:
  JsonlSink() = default;
  explicit JsonlSink(const std::filesystem::path &path);
  JsonlSink(JsonlSink &&) noexcept = default;
  JsonlSink &operator=(JsonlSink &&) noexcept = default;

  template <class Record> void write(const Record &record) { write_json(encode(record)); }
  void write_json(const Json &value);
  void flush();
  [[nodiscard]] std::size_t lines() const noexcept { return lines_; }

private:
  std::unique_ptr<std::ofstream> out_;
  std::size_t lines_ = 0;
};

Status write_json_file(const std::filesystem::path &path, const Json &value);

}
