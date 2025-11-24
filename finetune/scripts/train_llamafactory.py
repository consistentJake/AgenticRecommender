"""Thin wrapper for launching LLaMA-Factory with repo-local configs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a LLaMA-Factory YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name passed straight to the trainer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory defined in the YAML.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Force a device mode. 'auto' relies on LLaMA-Factory defaults.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra CLI overrides, e.g. --override per_device_train_batch_size=2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    argv = [
        "llamafactory-cli",
        "train",
        str(args.config.resolve()),
    ]
    if args.run_name:
        argv += ["--run_name", args.run_name]
    if args.output_dir:
        argv += ["--output_dir", str(args.output_dir)]
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override string: {item}")
        key, value = item.split("=", 1)
        flag = f"--{key.lstrip('-')}"
        argv += [flag, value]

    # We need to guard against torch builds lacking torch.mps.* attributes (common on CPU installs).
    patch = """
import sys
import types
import torch
if not hasattr(torch, "mps"):
    torch.mps = types.SimpleNamespace()
if not hasattr(torch.mps, "device_count"):
    torch.mps.device_count = staticmethod(lambda: 0)
if not hasattr(torch.mps, "current_allocated_memory"):
    torch.mps.current_allocated_memory = staticmethod(lambda: 0)
if not hasattr(torch.mps, "recommended_max_memory"):
    torch.mps.recommended_max_memory = staticmethod(lambda: 0)
if not hasattr(torch, "backends"):
    torch.backends = types.SimpleNamespace()
if not hasattr(torch.backends, "mps"):
    torch.backends.mps = types.SimpleNamespace()
if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    torch.backends.mps.is_macos_or_newer = staticmethod(lambda *args, **kwargs: False)
if not hasattr(torch.backends.mps, "is_macos13_or_newer"):
    torch.backends.mps.is_macos13_or_newer = staticmethod(lambda: False)
from llamafactory.cli import main as _main
sys.argv = {argv}
_main()
""".strip().format(argv=repr(argv))
    return [sys.executable, "-c", patch]


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif args.device == "cuda":
        if not env.get("CUDA_VISIBLE_DEVICES"):
            # Leave to runtime detection; user can still export manually.
            pass
    print(f"[llamafactory] {' '.join(shlex.quote(part) for part in cmd)}")
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
