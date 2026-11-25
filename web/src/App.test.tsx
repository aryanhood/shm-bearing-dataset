import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

describe("Motor Health Check App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders title and diagnostic mode tabs cleanly", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          model: {
            file: "/models/order_mrl_int8.onnx",
            size_kb: 35.77,
            classes: ["healthy", "inner", "outer", "ball", "combined"],
            in_channels: 4,
            n_bins: 400,
          },
          onnx_samples: [],
          healthy_baseline: {
            machine_name: "Healthy",
            fs_hz: 12000,
            units: "g",
            n_windows: 6,
            healthy_rms: 0.05,
            healthy_kurtosis: 3.0,
            feature_median: [0, 3, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1],
            feature_scale: [0.1, 0.6, 0.5, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
          },
          reference_examples: {
            healthy: {
              file: "motor_healthy.csv",
              url: "/examples/motor_healthy.csv",
              fs_hz: 12000,
              duration_s: 1.5,
              quality_ok: true,
              iso_velocity_mm_s: 0.15,
              iso_zone: "A",
              drift_score: 0,
              raw_distance: 0.2,
              rms_ratio: 1,
              kurtosis_delta: 0,
              spectral_shift: 0,
            },
            inner_fault: {
              file: "motor_inner_fault.csv",
              url: "/examples/motor_inner_fault.csv",
              fs_hz: 12000,
              duration_s: 1.5,
              quality_ok: true,
              iso_velocity_mm_s: 0.216,
              iso_zone: "A",
              drift_score: 53.75,
              raw_distance: 3.7,
              rms_ratio: 4.2,
              kurtosis_delta: 2.8,
              spectral_shift: 0.7,
            },
          },
        }),
      }),
    );

    render(<App />);
    expect(screen.getByText("Motor Health Check")).toBeInTheDocument();
    expect(await screen.findByText(/1\. Pick a real motor recording/i)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Live Variable-Speed Check/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Check Your Own File/i })).toBeInTheDocument();
  });
});
