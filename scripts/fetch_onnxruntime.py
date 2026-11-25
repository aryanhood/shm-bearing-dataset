"""Download and extract official Microsoft.ML.OnnxRuntime C/C++ headers and native libraries."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ORT_VERSION = "1.32.0"
NUGET_URL = (
    f"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/"
    f"{ORT_VERSION}/microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg"
)


def _platform_runtime_prefix() -> str:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if sys_name == "windows":
        return "runtimes/win-x64/native/"
    if sys_name == "linux":
        if "aarch64" in machine or "arm64" in machine:
            return "runtimes/linux-arm64/native/"
        return "runtimes/linux-x64/native/"
    if sys_name == "darwin":
        return "runtimes/osx-arm64/native/"
    raise RuntimeError(f"Unsupported platform for OnnxRuntime fetch: {sys_name}/{machine}")


def _sentinel_paths(lib_dir: Path) -> tuple[Path, ...]:
    sys_name = platform.system().lower()
    if sys_name == "windows":
        return (lib_dir / "onnxruntime.dll",)
    if sys_name == "darwin":
        return (lib_dir / "libonnxruntime.dylib",)
    return (lib_dir / "libonnxruntime.so", lib_dir / "libonnxruntime.so.1")


def _ensure_linux_soname(lib_dir: Path) -> None:
    """Create libonnxruntime.so.1 alias expected by DT_SONAME on Linux."""
    if platform.system().lower() != "linux":
        return
    so_path = lib_dir / "libonnxruntime.so"
    soname_path = lib_dir / "libonnxruntime.so.1"
    if not so_path.exists() or soname_path.exists():
        return
    try:
        os.symlink(so_path.name, soname_path)
    except OSError:
        shutil.copy2(so_path, soname_path)


def fetch_onnxruntime(root: Path) -> Path:
    """Ensure ONNX Runtime headers and native libraries exist under `.cache/onnxruntime-<ver>`."""
    ort_root = root / ".cache" / f"onnxruntime-{ORT_VERSION}"
    inc_dir = ort_root / "include"
    lib_dir = ort_root / "lib"

    if (inc_dir / "onnxruntime_cxx_api.h").exists() and all(
        p.exists() for p in _sentinel_paths(lib_dir)
    ):
        return ort_root

    ort_root.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    nupkg_path = root / ".cache" / f"microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg"
    if not nupkg_path.exists():
        print(f"Downloading {NUGET_URL} ...", file=sys.stderr)
        req = urllib.request.Request(NUGET_URL, headers={"User-Agent": "BearingBench-Fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, nupkg_path.open("wb") as out:
            shutil.copyfileobj(resp, out)

    native_prefix = _platform_runtime_prefix()
    inc_prefix = "build/native/include/"

    with zipfile.ZipFile(nupkg_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename.replace("\\", "/")
            if name.startswith(inc_prefix):
                rel = name[len(inc_prefix) :]
                dest = inc_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            elif name.startswith(native_prefix):
                rel = name[len(native_prefix) :]
                dest = lib_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    _ensure_linux_soname(lib_dir)

    if not (inc_dir / "onnxruntime_cxx_api.h").exists() or not all(
        p.exists() for p in _sentinel_paths(lib_dir)
    ):
        raise RuntimeError(f"Extraction incomplete under {ort_root}")

    return ort_root


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ort_root = fetch_onnxruntime(root)
    print(ort_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
