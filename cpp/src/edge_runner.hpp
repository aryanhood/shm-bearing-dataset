#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace bearingbench {

inline constexpr std::size_t kChannels = 4;
inline constexpr std::size_t kBins = 400;
inline constexpr std::size_t kInputSize = kChannels * kBins;
inline constexpr std::size_t kNumClasses = 5;

inline constexpr std::array<std::string_view, kNumClasses> kClassLabels{
    "healthy",
    "inner",
    "outer",
    "ball",
    "combined",
};

struct Prediction {
  std::array<float, kNumClasses> probs{};
  std::size_t predicted_index = 0;
  std::string_view label = "healthy";
  std::string_view status = "HEALTHY";
};

struct LatencyReport {
  double mean_us = 0.0;
  double p50_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  std::size_t model_bytes = 0;
};

class EdgeRunner {
 public:
  explicit EdgeRunner(const std::filesystem::path& model_path);
  ~EdgeRunner();

  EdgeRunner(const EdgeRunner&) = delete;
  EdgeRunner& operator=(const EdgeRunner&) = delete;
  EdgeRunner(EdgeRunner&&) noexcept;
  EdgeRunner& operator=(EdgeRunner&&) noexcept;

  [[nodiscard]] Prediction predict(std::span<const float> order_ses) const;
  [[nodiscard]] LatencyReport benchmark(std::span<const float> sample,
                                        int warmup = 50,
                                        int runs = 500) const;
  [[nodiscard]] std::size_t model_bytes() const noexcept { return model_bytes_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::size_t model_bytes_ = 0;
};

[[nodiscard]] std::string_view status_for_class(std::string_view label) noexcept;

}  // namespace bearingbench
