"""Export deterministic ONNX, uOttawa, baseline, and C++ reference assets."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bearingbench.baselines.unsupervised import (
    audit_signal,
    compute_iso_velocity,
    learn_baseline,
    load_signal,
    score_against_baseline,
)
from bearingbench.edge.onnx_edge import EdgeClassifier

CLASS_TITLES: dict[str, str] = {
    "healthy": "Healthy Motor (Speed Ramp)",
    "inner": "Worn Inner Ring",
    "outer": "Worn Outer Ring",
    "ball": "Worn Bearing Ball",
    "combined": "Severe Combined Wear",
}

CLASS_STEM_PREFIX: dict[int, str] = {
    0: "H",
    1: "I",
    2: "O",
    3: "B",
    4: "C",
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    onnx_src = root / "artifacts" / "order_mrl_int8.onnx"
    meta_src = root / "artifacts" / "order_mrl_meta.json"
    cache_path = root / "data" / "uottawa" / "prepared_cache.npz"
    ablation_path = root / "data" / "uottawa" / "ablation_cache.npz"
    healthy_csv = root / "examples" / "motor_healthy.csv"
    fault_csv = root / "examples" / "motor_inner_fault.csv"

    for required in (onnx_src, meta_src, cache_path, healthy_csv, fault_csv):
        if not required.exists():
            raise FileNotFoundError(f"Required source asset missing: {required}")

    web_models = root / "web" / "public" / "models"
    web_examples = root / "web" / "public" / "examples"
    web_data = root / "web" / "public" / "data"
    cpp_data = root / "cpp" / "test_data"
    for folder in (web_models, web_examples, web_data, cpp_data):
        folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_src, web_models / onnx_src.name)
    shutil.copy2(meta_src, web_models / meta_src.name)
    shutil.copy2(healthy_csv, web_examples / healthy_csv.name)
    shutil.copy2(fault_csv, web_examples / fault_csv.name)

    clf = EdgeClassifier(onnx_src, meta_src)

    with np.load(cache_path, allow_pickle=False) as pr:
        cnn_x = np.asarray(pr["cnn_x"], dtype=np.float32)
        y = np.asarray(pr["y"], dtype=np.int64)
        trial = np.asarray(pr["trial"], dtype=np.int64)
        profile_code = np.asarray(pr["profile_code"], dtype=np.str_)
        rpm = np.asarray(pr["rpm"], dtype=np.float64)

    window_stems: list[str]
    if ablation_path.exists():
        with np.load(ablation_path, allow_pickle=False) as ab:
            window_stems = [str(s) for s in ab["window_stem"]]
    else:
        window_stems = [
            f"{CLASS_STEM_PREFIX[int(y[i])]}-{profile_code[i]}-{int(trial[i])}"
            for i in range(len(y))
        ]

    all_probs = clf.predict_proba(cnn_x)

    onnx_samples: list[dict[str, Any]] = []
    cpp_rows: list[list[str]] = []

    for c_idx, cname in enumerate(clf.classes):
        candidates = np.where((trial == 3) & (profile_code == "C") & (y == c_idx))[0]
        if len(candidates) == 0:
            raise RuntimeError(f"No Trial-3 Profile-C candidates found for class {cname}")
        best_idx = int(max(candidates, key=lambda idx: float(all_probs[idx, c_idx])))
        if int(np.argmax(all_probs[best_idx])) != c_idx:
            raise RuntimeError(f"Selected window {best_idx} for {cname} misclassified")

        tensor_f32 = cnn_x[best_idx].astype(np.float32)
        flat_exact = tensor_f32.ravel()
        flat_rounded = [round(float(v), 5) for v in flat_exact]
        rounded_arr = np.asarray(flat_rounded, dtype=np.float32).reshape(1, 4, 400)
        json_probs = clf.predict_proba(rounded_arr)[0]
        exact_probs = all_probs[best_idx]

        stem = window_stems[best_idx]
        win_rpm = round(float(rpm[best_idx]), 1)

        onnx_samples.append(
            {
                "id": f"{cname}_c3",
                "title": CLASS_TITLES[cname],
                "stem": stem,
                "true_class": cname,
                "rpm": win_rpm,
                "python_probs": [round(float(v), 6) for v in json_probs],
                "tensor_4x400": flat_rounded,
            }
        )

        cpp_row = [
            stem,
            cname,
            f"{float(rpm[best_idx]):.2f}",
            *(f"{float(p):.8f}" for p in exact_probs),
            *(f"{float(v):.9g}" for v in flat_exact),
        ]
        cpp_rows.append(cpp_row)

    sig_healthy = load_signal(healthy_csv)
    sig_fault = load_signal(fault_csv)
    baseline = learn_baseline(sig_healthy, machine_name="Healthy Reference (12 kHz)")

    ref_examples: dict[str, dict[str, Any]] = {}
    for key, path, sig in (
        ("healthy", healthy_csv, sig_healthy),
        ("inner_fault", fault_csv, sig_fault),
    ):
        qual = audit_signal(sig)
        iso = compute_iso_velocity(sig)
        drift = score_against_baseline(sig, baseline)
        ref_examples[key] = {
            "file": path.name,
            "url": f"/examples/{path.name}",
            "fs_hz": round(sig.fs_hz, 3),
            "duration_s": round(sig.duration_s, 3),
            "quality_ok": qual.ok,
            "iso_velocity_mm_s": iso.velocity_rms_mm_s if iso else None,
            "iso_zone": iso.zone if iso else None,
            "drift_score": drift.drift_score,
            "raw_distance": drift.raw_distance,
            "rms_ratio": drift.rms_ratio,
            "kurtosis_delta": drift.kurtosis_delta,
            "spectral_shift": drift.spectral_shift,
        }

    baseline_dict = asdict(baseline)
    baseline_dict["healthy_rms"] = round(float(baseline.healthy_rms), 6)
    baseline_dict["healthy_kurtosis"] = round(float(baseline.healthy_kurtosis), 4)
    baseline_dict["feature_median"] = [round(float(v), 6) for v in baseline.feature_median]
    baseline_dict["feature_scale"] = [round(float(v), 6) for v in baseline.feature_scale]

    payload = {
        "model": {
            "file": f"/models/{onnx_src.name}",
            "size_kb": round(onnx_src.stat().st_size / 1024.0, 2),
            "classes": list(clf.classes),
            "in_channels": clf.in_channels,
            "n_bins": clf.n_bins,
        },
        "onnx_samples": onnx_samples,
        "healthy_baseline": baseline_dict,
        "reference_examples": ref_examples,
    }

    samples_json = web_data / "motor_samples.json"
    samples_json.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    cpp_csv = cpp_data / "reference_windows.csv"
    header = [
        "stem",
        "true_class",
        "rpm",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        *(f"f{i}" for i in range(1600)),
    ]
    with cpp_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(cpp_rows)

    print(f"Wrote {samples_json} ({samples_json.stat().st_size / 1024.0:.2f} KB)")
    print(f"Wrote {cpp_csv} ({cpp_csv.stat().st_size / 1024.0:.2f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
