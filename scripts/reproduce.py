"""Single deterministic command to reproduce every table and figure from raw .mat data."""

from __future__ import annotations

from pathlib import Path

from bearingbench.evaluation.plots import generate_all_figures
from bearingbench.evaluation.runner import run_all_experiments


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    run_all_experiments(root)
    generate_all_figures(root)


if __name__ == "__main__":
    main()
