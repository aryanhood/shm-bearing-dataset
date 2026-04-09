"""Backward-compatible wrapper for legacy experiment entrypoint."""
from pipelines.train import main


if __name__ == "__main__":
    main()
