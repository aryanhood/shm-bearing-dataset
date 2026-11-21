"""Download the 40 CWRU drive-end recordings (about 120 MB) used by BearingScope."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

from .data import CATALOG

BASE_URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"


def download(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_id in sorted(CATALOG):
        target = out_dir / f"{file_id}.mat"
        if target.exists() and target.stat().st_size > 0:
            continue
        url = BASE_URL.format(file_id=file_id)
        partial = target.with_suffix(".part")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, partial)
                partial.replace(target)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        print(f"  downloaded {target.name}")
    print(f"{len(CATALOG)} recordings ready in {out_dir}")
