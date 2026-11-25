#include "edge_runner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ReferenceRow {
  std::string stem;
  std::string true_class;
  double rpm = 0.0;
  std::array<float, bearingbench::kNumClasses> py_probs{};
  std::array<float, bearingbench::kInputSize> tensor{};
};

std::vector<ReferenceRow> load_reference_csv(const std::filesystem::path& csv_path) {
  std::ifstream in(csv_path);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open reference CSV: " + csv_path.string());
  }

  std::string line;
  if (!std::getline(in, line)) {
    throw std::runtime_error("Reference CSV is empty: " + csv_path.string());
  }

  std::vector<ReferenceRow> rows;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> cells;
    cells.reserve(8 + bearingbench::kInputSize);
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      cells.push_back(cell);
    }
    if (cells.size() != 8 + bearingbench::kInputSize) {
      throw std::runtime_error("Unexpected column count in reference CSV row");
    }

    ReferenceRow row;
    row.stem = cells[0];
    row.true_class = cells[1];
    row.rpm = std::stod(cells[2]);
    for (std::size_t k = 0; k < bearingbench::kNumClasses; ++k) {
      row.py_probs[k] = std::stof(cells[3 + k]);
    }
    for (std::size_t i = 0; i < bearingbench::kInputSize; ++i) {
      row.tensor[i] = std::stof(cells[8 + i]);
    }
    rows.push_back(std::move(row));
  }

  if (rows.empty()) {
    throw std::runtime_error("No data rows in reference CSV");
  }
  return rows;
}

void print_usage() {
  std::cout << "Usage: bearingbench-edge --model <onnx_path> --verify-ref <csv_path> "
               "[--runs 500] [--warmup 50] [--tol 1e-4]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_path = "artifacts/order_mrl_int8.onnx";
  std::filesystem::path ref_csv = "cpp/test_data/reference_windows.csv";
  int runs = 500;
  int warmup = 50;
  double tol = 1e-4;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_arg = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + flag);
      }
      return argv[++i];
    };
    if (arg == "--model") {
      model_path = next_arg(arg);
    } else if (arg == "--verify-ref") {
      ref_csv = next_arg(arg);
    } else if (arg == "--runs") {
      runs = std::stoi(next_arg(arg));
    } else if (arg == "--warmup") {
      warmup = std::stoi(next_arg(arg));
    } else if (arg == "--tol") {
      tol = std::stod(next_arg(arg));
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage();
      return 2;
    }
  }

  try {
    bearingbench::EdgeRunner runner(model_path);
    const auto rows = load_reference_csv(ref_csv);

    double worst_diff = 0.0;
    std::size_t matched = 0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== BearingBench C++20 Edge Verification ===\n";
    std::cout << "Model: " << model_path.string() << " ("
              << (static_cast<double>(runner.model_bytes()) / 1024.0) << " KB)\n\n";

    for (const auto& row : rows) {
      const auto pred = runner.predict(row.tensor);
      double row_max_diff = 0.0;
      for (std::size_t k = 0; k < bearingbench::kNumClasses; ++k) {
        const double diff =
            std::abs(static_cast<double>(pred.probs[k]) - static_cast<double>(row.py_probs[k]));
        row_max_diff = std::max(row_max_diff, diff);
      }
      worst_diff = std::max(worst_diff, row_max_diff);

      const bool class_ok = (pred.label == row.true_class);
      if (class_ok) ++matched;

      std::cout << "  " << std::setw(6) << row.stem
                << " | rpm=" << std::setw(7) << std::setprecision(1) << row.rpm
                << " | true=" << std::setw(8) << row.true_class
                << " | pred=" << std::setw(8) << pred.label
                << " | conf=" << std::setprecision(4) << pred.probs[pred.predicted_index]
                << " | status=" << std::setw(16) << pred.status
                << " | max_py_diff=" << std::scientific << std::setprecision(2) << row_max_diff
                << std::fixed << "\n";
    }

    const auto report = runner.benchmark(rows.front().tensor, warmup, runs);
    std::cout << "\n=== C++20 Single-Thread CPU Latency (" << runs << " runs, "
              << warmup << " warmup) ===\n";
    std::cout << std::fixed << std::setprecision(2)
              << "  size_kb=" << (static_cast<double>(report.model_bytes) / 1024.0)
              << " | mean=" << report.mean_us << " us (" << std::setprecision(4) << (report.mean_us / 1000.0) << " ms)"
              << std::setprecision(2)
              << " | p50=" << report.p50_us << " us (" << std::setprecision(4) << (report.p50_us / 1000.0) << " ms)"
              << std::setprecision(2)
              << " | p95=" << report.p95_us << " us"
              << " | p99=" << report.p99_us << " us\n";

    std::cout << "\nParity check: " << matched << "/" << rows.size()
              << " classes matched, worst probability diff = "
              << std::scientific << std::setprecision(3) << worst_diff << "\n";

    if (matched != rows.size() || worst_diff > tol) {
      std::cerr << "FAIL: Verification exceeded tolerance or mismatched classes.\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
