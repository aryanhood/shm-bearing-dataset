"""University of Ottawa time-varying speed bearing dataset loader & downloader.

Reference:
    H. Huang and N. Baddour, "Bearing vibration data collected under time-varying
    rotational speed conditions," Data in Brief, vol. 21, pp. 1745-1749, 2018.
    Mendeley Data DOI: 10.17632/v43hmbwxpm.2
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

SAMPLE_RATE_HZ = 200_000.0
ENCODER_PPR = 1024

CLASSES: tuple[str, ...] = ("healthy", "inner", "outer", "ball", "combined")
CLASS_TITLES: tuple[str, ...] = (
    "Healthy",
    "Inner race",
    "Outer race",
    "Ball",
    "Combined",
)
CLASS_CODES: dict[str, str] = {
    "H": "healthy",
    "I": "inner",
    "O": "outer",
    "B": "ball",
    "C": "combined",
}
SPEED_PROFILES: dict[str, str] = {
    "A": "increasing",
    "B": "decreasing",
    "C": "inc_dec",
    "D": "dec_inc",
}
TRIALS: tuple[int, ...] = (1, 2, 3)

MENDELEY_FILES_API = "https://data.mendeley.com/api/datasets/v43hmbwxpm/files?version=2"


@dataclass(frozen=True)
class RecordingSpec:
    """Metadata parsed from a uOttawa recording filename (e.g. 'I-A-1')."""

    stem: str
    label: str
    class_idx: int
    profile_code: str
    profile: str
    trial: int

    @property
    def filename(self) -> str:
        return f"{self.stem}.mat"


def _build_catalog() -> tuple[RecordingSpec, ...]:
    specs: list[RecordingSpec] = []
    for class_code, label in CLASS_CODES.items():
        class_idx = CLASSES.index(label)
        for profile_code, profile in SPEED_PROFILES.items():
            for trial in TRIALS:
                stem = f"{class_code}-{profile_code}-{trial}"
                specs.append(
                    RecordingSpec(
                        stem=stem,
                        label=label,
                        class_idx=class_idx,
                        profile_code=profile_code,
                        profile=profile,
                        trial=trial,
                    )
                )
    return tuple(specs)


CATALOG: tuple[RecordingSpec, ...] = _build_catalog()
SPEC_BY_STEM: dict[str, RecordingSpec] = {s.stem: s for s in CATALOG}


@dataclass(frozen=True)
class Recording:
    """Dual-channel raw sensor recording (accelerometer + 1024-CPR encoder)."""

    spec: RecordingSpec
    fs_hz: float
    vibration: NDArray[np.float64]
    encoder: NDArray[np.float64]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_file(entry: dict[str, object], dest_dir: Path) -> Path:
    filename = str(entry["filename"])
    details = entry["content_details"]
    if not isinstance(details, dict):
        raise ValueError(f"unexpected Mendeley entry for {filename}")
    url = str(details["download_url"])
    expected_sha = str(details.get("sha256_hash", ""))
    expected_size = int(details.get("size", 0))

    target = dest_dir / filename
    if (
        target.exists()
        and target.stat().st_size == expected_size
        and (not expected_sha or _sha256_file(target) == expected_sha)
    ):
        return target

    part = dest_dir / f"{filename}.part"
    req = urllib.request.Request(url, headers={"User-Agent": "orderguard/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, part.open("wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)

    if expected_sha and _sha256_file(part) != expected_sha:
        part.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch downloading {filename}")
    part.replace(target)
    return target


def download_uottawa(
    dest_dir: Path,
    stems: Sequence[str] | None = None,
    *,
    max_workers: int = 6,
) -> list[Path]:
    """Download the uOttawa dataset from Mendeley Data (DOI 10.17632/v43hmbwxpm.2)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        MENDELEY_FILES_API,
        headers={"User-Agent": "orderguard/0.1", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        entries: list[dict[str, object]] = json.loads(resp.read().decode("utf-8"))

    by_name = {str(e["filename"]): e for e in entries}
    wanted_specs = [SPEC_BY_STEM[s] for s in stems] if stems is not None else list(CATALOG)
    missing_remote = [s.filename for s in wanted_specs if s.filename not in by_name]
    if missing_remote:
        raise KeyError(f"missing files in Mendeley catalog: {missing_remote}")

    selected_entries = [by_name[s.filename] for s in wanted_specs]
    paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        future_map = {
            pool.submit(_fetch_file, entry, dest_dir): str(entry["filename"])
            for entry in selected_entries
        }
        for i, fut in enumerate(as_completed(future_map), start=1):
            fname = future_map[fut]
            paths[fname] = fut.result()
            print(f"[{i:02d}/{len(selected_entries):02d}] {fname}")
    return [paths[s.filename] for s in wanted_specs]


def parse_stem(path: Path) -> RecordingSpec:
    """Resolve a RecordingSpec from a filename stem such as 'I-A-1'."""
    stem = path.stem
    if stem in SPEC_BY_STEM:
        return SPEC_BY_STEM[stem]
    return RecordingSpec(
        stem=stem,
        label="unknown",
        class_idx=-1,
        profile_code="?",
        profile="unknown",
        trial=0,
    )


def load_mat_file(path: Path, *, fs_hz: float = SAMPLE_RATE_HZ) -> Recording:
    """Load an arbitrary uOttawa-schema .mat file ('Channel_1' + 'Channel_2')."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `orderguard download` first")
    mat = loadmat(path)
    if "Channel_1" not in mat or "Channel_2" not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"{path.name}: expected Channel_1 and Channel_2, found {keys}")
    vibration = np.asarray(mat["Channel_1"], dtype=np.float64).ravel()
    encoder = np.asarray(mat["Channel_2"], dtype=np.float64).ravel()
    if len(vibration) != len(encoder):
        raise ValueError(
            f"{path.name}: channel length mismatch ({len(vibration)} vs {len(encoder)})"
        )
    return Recording(
        spec=parse_stem(path),
        fs_hz=fs_hz,
        vibration=vibration,
        encoder=encoder,
    )


def load_recording(raw_dir: Path, stem: str) -> Recording:
    """Load a catalogued uOttawa recording by stem (e.g. 'I-A-1')."""
    if stem not in SPEC_BY_STEM:
        raise KeyError(f"unknown recording stem {stem!r}")
    return load_mat_file(raw_dir / f"{stem}.mat")


def available_specs(raw_dir: Path) -> list[RecordingSpec]:
    """Return catalog specs whose .mat files are present in raw_dir."""
    return [s for s in CATALOG if (raw_dir / s.filename).exists()]
