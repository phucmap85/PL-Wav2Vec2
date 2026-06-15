from __future__ import annotations

import argparse
import json

from config import load_config
from vocab import build_vocab_file


def _parse_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, value = item.split("=", 1)
        cfg[key] = _parse_value(value)
    return cfg


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LingWav2Vec2 + pitch encoder for Vietnamese MDD Challenge."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    vocab_parser = subparsers.add_parser("make-vocab", help="Build vocabulary from lexicon")
    vocab_parser.add_argument("--input", default="metadata/lexicon_vmd.txt")
    vocab_parser.add_argument("--output", default="vocab.json")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", default="config.json")
    train_parser.add_argument("--set", action="append", default=[], help="Override config with key=value")

    predict_parser = subparsers.add_parser("predict", help="Create an output CSV with predictions")
    predict_parser.add_argument("--config", default="config.json")
    predict_parser.add_argument("--data", default="metadata/test_phones.csv")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--output", default="results.csv")
    predict_parser.add_argument("--max-samples", type=int, default=None)
    predict_parser.add_argument("--set", action="append", default=[], help="Override config with key=value")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)

    if args.command == "make-vocab":
        vocab = build_vocab_file(args.input, args.output)
        print(f"Wrote {len(vocab)} tokens to {args.output}")
        return

    if args.command == "train":
        from train import run_training

        cfg = _apply_overrides(load_config(args.config), args.set)
        run_training(cfg)
        return

    if args.command == "predict":
        from predict import predict

        cfg = _apply_overrides(load_config(args.config), args.set)
        predict(
            cfg=cfg,
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            output_csv=args.output,
            max_samples=args.max_samples,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
