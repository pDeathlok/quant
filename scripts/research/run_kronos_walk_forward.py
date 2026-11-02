#!/usr/bin/env python3
"""Audit and run the Kronos daily walk-forward research experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant.research.kronos_walk_forward import audit_experiment, load_config, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--folds", nargs="+")
            command.add_argument("--epochs", type=int)
            command.add_argument("--max-train-samples", type=int)
            command.add_argument("--max-eval-origins", type=int)
            command.add_argument("--output-dir", type=Path)
            command.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "audit":
        result = audit_experiment(config)
    else:
        result = run_experiment(
            config,
            fold_names=args.folds,
            epochs_override=args.epochs,
            max_train_samples=args.max_train_samples,
            max_eval_origins=args.max_eval_origins,
            output_dir_override=args.output_dir,
            device_override=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
