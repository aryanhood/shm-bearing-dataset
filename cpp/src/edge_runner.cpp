#include "edge_runner.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace bearingbench {

namespace {

std::array<float, kNumClasses> softmax_5(std::span<const float> logits) {
  if (logits.size() != kNumClasses) {
    throw std::runtime_error("Unexpected ONNX output logit count");
  }
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::array<float, kNumClasses> probs{};
  double sum = 0.0;
  for (std::size_t i = 0; i < kNumClasses; ++i) {
    const double e = std::exp(static_cast<double>(logits[i] - max_logit));
    probs[i] = static_cast<float>(e);
    sum += e;
  }
  if (sum <= 0.0) {
    throw std::runtime_error("Softmax normalization sum is non-positive");
  }
  for (float& p : probs) {
    p = static_cast<float>(static_cast<double>(p) / sum);
  }
  return probs;
}

double percentile(const std::vector<double>& sorted_vals, double pct) {
  if (sorted_vals.empty()) return 0.0;
  const double pos = (pct / 100.0) * static_cast<double>(sorted_vals.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) return sorted_vals[lo];
  const double frac = pos - static_cast<double>(lo);
  return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

}  // namespace

struct EdgeRunner::Impl {
  Ort::Env env;
  Ort::SessionOptions session_opts;
  Ort::Session session{nullptr};
  Ort::MemoryInfo mem_info{nullptr};
  std::string input_name;
  std::string output_name;

  explicit Impl(const std::filesystem::path& model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "bearingbench"),
        mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetInterOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    const std::wstring native_path = model_path.wstring();
    session = Ort::Session(env, native_path.c_str(), session_opts);
#else
    const std::string native_path = model_path.string();
    session = Ort::Session(env, native_path.c_str(), session_opts);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_ptr = session.GetInputNameAllocated(0, allocator);
    auto out_ptr = session.GetOutputNameAllocated(0, allocator);
    input_name = in_ptr.get();
    output_name = out_ptr.get();
  }

  std::array<float, kNumClasses> run_raw(std::span<float> input_buf) const {
    const std::array<std::int64_t, 3> shape{
        1,
        static_cast<std::int64_t>(kChannels),
        static_cast<std::int64_t>(kBins),
    };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_buf.data(), input_buf.size(), shape.data(), shape.size());

    const char* in_names[] = {input_name.c_str()};
    const char* out_names[] = {output_name.c_str()};

    auto outputs = const_cast<Ort::Session&>(session).Run(
        Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
      throw std::runtime_error("ONNX Runtime returned invalid output tensor");
    }

    const float* raw = outputs[0].GetTensorData<float>();
    std::array<float, kNumClasses> logits{};
    std::copy_n(raw, kNumClasses, logits.begin());
    return logits;
  }
};

std::string_view status_for_class(std::string_view label) noexcept {
  if (label == "healthy") return "HEALTHY";
  if (label == "combined") return "STOP_AND_INSPECT";
  return "WATCH";
}

EdgeRunner::EdgeRunner(const std::filesystem::path& model_path) {
  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("ONNX model not found: " + model_path.string());
  }
  model_bytes_ = static_cast<std::size_t>(std::filesystem::file_size(model_path));
  impl_ = std::make_unique<Impl>(model_path);
}

EdgeRunner::~EdgeRunner() = default;
EdgeRunner::EdgeRunner(EdgeRunner&&) noexcept = default;
EdgeRunner& EdgeRunner::operator=(EdgeRunner&&) noexcept = default;

Prediction EdgeRunner::predict(std::span<const float> order_ses) const {
  if (order_ses.size() != kInputSize) {
    throw std::invalid_argument("Expected 1600 floats (4 x 400 Order-SES input)");
  }
  std::array<float, kInputSize> buf{};
  for (std::size_t i = 0; i < kInputSize; ++i) {
    if (!std::isfinite(order_ses[i])) {
      throw std::invalid_argument("Input Order-SES tensor contains non-finite value");
    }
    buf[i] = order_ses[i];
  }

  const auto logits = impl_->run_raw(buf);
  const auto probs = softmax_5(logits);

  const auto best_it = std::max_element(probs.begin(), probs.end());
  const auto best_idx = static_cast<std::size_t>(std::distance(probs.begin(), best_it));
  const std::string_view label = kClassLabels[best_idx];

  return Prediction{
      .probs = probs,
      .predicted_index = best_idx,
      .label = label,
      .status = status_for_class(label),
  };
}

LatencyReport EdgeRunner::benchmark(std::span<const float> sample,
                                    int warmup,
                                    int runs) const {
  if (sample.size() != kInputSize) {
    throw std::invalid_argument("Benchmark sample must have 1600 floats");
  }
  if (runs <= 0) {
    throw std::invalid_argument("Benchmark runs must be positive");
  }
  std::array<float, kInputSize> buf{};
  std::copy(sample.begin(), sample.end(), buf.begin());

  for (int i = 0; i < warmup; ++i) {
    (void)impl_->run_raw(buf);
  }

  std::vector<double> samples_us;
  samples_us.reserve(static_cast<std::size_t>(runs));

  for (int i = 0; i < runs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)impl_->run_raw(buf);
    const auto t1 = std::chrono::steady_clock::now();
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    samples_us.push_back(us);
  }

  std::sort(samples_us.begin(), samples_us.end());
  const double total = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);

  return LatencyReport{
      .mean_us = total / static_cast<double>(samples_us.size()),
      .p50_us = percentile(samples_us, 50.0),
      .p95_us = percentile(samples_us, 95.0),
      .p99_us = percentile(samples_us, 99.0),
      .model_bytes = model_bytes_,
  };
}

}  // namespace bearingbench
