"""Export the browser model and the demo recordings for the web dashboard.

The web model is trained on motor loads 0-2 HP only. The demo streams the
3 HP recordings, so everything you see on the live site is unseen data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import CATALOG, load_recording, load_windows, window
from .features import FEATURE_NAMES, envelope_spectrum, extract, extract_many
from .model import Model, train
from .physics import (
    BPFI_ORDER,
    BPFO_ORDER,
    BSF_ORDER,
    CLASS_TITLES,
    CLASSES,
    SAMPLE_RATE_HZ,
    WINDOW,
)

DEMO_LOAD_HP = 3
DEMO_WINDOWS = 12  # ~2 s of signal per recording
INT16_MAX = 32767


def export_web(raw_dir: Path, web_dir: Path, model_path: Path) -> Model:
    data = load_windows(raw_dir)
    train_mask = data.load != DEMO_LOAD_HP
    model = train(extract_many(data.x[train_mask], data.rpm[train_mask]), data.y[train_mask])
    model.save(model_path)
    web_dir.mkdir(parents=True, exist_ok=True)
    model.save(web_dir / "model.json")

    demo_ids = sorted(
        (fid for fid, (_, _, load) in CATALOG.items() if load == DEMO_LOAD_HP),
        key=lambda fid: (CLASSES.index(CATALOG[fid][0]), CATALOG[fid][1]),
    )
    chunks: list[np.ndarray] = []
    recordings: list[dict[str, object]] = []
    golden: list[dict[str, object]] = []
    offset = 0
    for fid in demo_ids:
        rec = load_recording(raw_dir, fid)
        segment = window(rec.signal)[:DEMO_WINDOWS].ravel()
        scale = float(np.abs(segment).max()) / INT16_MAX
        q = np.round(segment / scale).astype(np.int16)
        chunks.append(q)
        title = CLASS_TITLES[CLASSES.index(rec.label)]
        size = f'{rec.diameter_mils / 1000:.3f}"' if rec.diameter_mils else ""
        recordings.append(
            {
                "id": fid,
                "label": rec.label,
                "title": title,
                "diameter": size,
                "load_hp": rec.load_hp,
                "rpm": rec.rpm,
                "offset": offset,
                "length": len(q),
                "scale": scale,
            }
        )
        offset += len(q)
        # parity fixtures for the TypeScript port, computed on the dequantized signal
        w = q[:WINDOW].astype(np.float64) * scale
        f = extract(w, rec.rpm)
        golden.append(
            {
                "id": fid,
                "rpm": rec.rpm,
                "window": w.tolist(),
                "features": f.tolist(),
                "proba": model.predict_proba(f)[0].tolist(),
                "envelope_head": envelope_spectrum(w).spectrum[:64].tolist(),
            }
        )

    np.concatenate(chunks).astype("<i2").tofile(web_dir / "demo.i16")
    meta = {
        "sample_rate": SAMPLE_RATE_HZ,
        "window": WINDOW,
        "classes": list(CLASSES),
        "class_titles": list(CLASS_TITLES),
        "features": list(FEATURE_NAMES),
        "orders": {"bpfi": BPFI_ORDER, "bpfo": BPFO_ORDER, "bsf": BSF_ORDER},
        "train_loads_hp": sorted(set(data.load[train_mask].tolist())),
        "demo_load_hp": DEMO_LOAD_HP,
        "recordings": recordings,
    }
    (web_dir / "demo.json").write_text(json.dumps(meta, indent=2) + "\n")
    fixtures = Path("web/src/lib/__fixtures__")
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "golden.json").write_text(json.dumps(golden[::3]))
    print(f"exported model + {len(recordings)} demo recordings to {web_dir}")
    return model
