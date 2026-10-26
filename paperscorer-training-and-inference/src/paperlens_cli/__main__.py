"""``paperlens`` CLI entrypoint.

Three subcommands:

  paperlens setup-model --hf_repo skonan/paperscorer-3b-vision-arxiv
        Snapshot-download a published PaperLens checkpoint to the local
        HuggingFace cache. Idempotent.

  paperlens setup-data --dataset_keys k1,k2,... [--all] [--local_dir ...]
        Materialize one or more LlamaFactory sharegpt datasets from the
        published HF release (or a local hf_release_local/ Parquet).
        Wraps scripts/reconstruction.py.

  paperlens serve --config configs/serve.yaml [--port 8002]
        Start a long-running FastAPI server that exposes POST /score over
        the same llamafactory.data.get_dataset path scripts/vllm_infer.py
        uses. Tokenization parity with training + offline batch.
"""
from __future__ import annotations

import argparse
import sys


def _add_setup_model(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "setup-model",
        help="Snapshot-download a PaperLens checkpoint to the HF cache.",
    )
    p.add_argument("--hf_repo", help="HF repo id (e.g. skonan/paperscorer-3b-vision-arxiv)")
    p.add_argument("--local_dir", default=None,
                   help="Optional local target dir (otherwise uses the default HF cache)")
    p.add_argument(
        "--list", action="store_true",
        help="List the known PaperLens HF repos and exit.",
    )
    p.add_argument("--revision", default=None, help="Optional revision (default: main)")
    p.set_defaults(func=_cmd_setup_model)


def _cmd_setup_model(args: argparse.Namespace) -> int:
    from . import setup_model
    return setup_model.run(args)


def _add_setup_data(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "setup-data",
        help="Materialize LlamaFactory sharegpt datasets from the published HF release.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset_keys", nargs="+",
                     help="dataset_info.json keys to materialize")
    src.add_argument("--all", action="store_true",
                     help="Materialize every key listed in the release manifest")
    p.add_argument("--local_dir", default=None,
                   help="Path to a locally-built hf_release_local/ (skip HF Hub fetch)")
    p.add_argument("--hf_text_repo", default="skonan/PaperScorer-Text")
    p.add_argument("--hf_vision_repo", default="skonan/PaperScorer-Vision")
    p.add_argument("--data_root", default="data",
                   help="Where to write the reconstructed sharegpt + images")
    p.set_defaults(func=_cmd_setup_data)


def _cmd_setup_data(args: argparse.Namespace) -> int:
    from . import setup_data
    return setup_data.run(args)


def _add_serve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "serve",
        help="Start the FastAPI /score server.",
    )
    p.add_argument("--config", default="configs/serve.yaml",
                   help="Path to the serve config YAML")
    p.add_argument("--host", default=None, help="Override server.host from config")
    p.add_argument("--port", type=int, default=None, help="Override server.port from config")
    p.set_defaults(func=_cmd_serve)


def _cmd_serve(args: argparse.Namespace) -> int:
    from . import serve
    return serve.run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperlenstraininfer",
        description="PaperLens deployment CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    _add_setup_model(sub)
    _add_setup_data(sub)
    _add_serve(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
