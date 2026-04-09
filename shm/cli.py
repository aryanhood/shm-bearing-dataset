"""Command line interface for the SHM repository."""
from __future__ import annotations

import argparse
import sys

import uvicorn

from src.utils.config import init_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shm", description="SHM command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "evaluate", "predict", "ablation"):
        command = subparsers.add_parser(name)
        command.add_argument("args", nargs=argparse.REMAINDER)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=str, default=None)
    serve.add_argument("--host", type=str, default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return

    command = argv[0]
    remainder = argv[1:]

    if command == "train":
        from pipelines.train import main as train_main

        train_main(remainder)
        return

    if command == "evaluate":
        from pipelines.evaluate import main as evaluate_main

        evaluate_main(remainder)
        return

    if command == "predict":
        from pipelines.predict import main as predict_main

        predict_main(remainder)
        return

    if command == "ablation":
        from pipelines.ablation import main as ablation_main

        ablation_main(remainder)
        return

    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = init_config(args.config, force=bool(args.config))
    from src.api.main import create_app

    uvicorn.run(
        create_app(),
        host=args.host or cfg["api"]["host"],
        port=args.port or int(cfg["api"]["port"]),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
