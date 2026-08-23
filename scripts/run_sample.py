#!/usr/bin/env python3
"""
scripts/run_sample.py

CapNav sample runner: identical to scripts/run.py, restricted to the curated
scene allowlist in src/utils/dataset_sample.py (~200 prompts). Useful for
sanity checks, debugging, and low-resource environments.

    python scripts/run_sample.py --model InternVL3_5-8B --num_frames 32 --thinking on

Model validation, routing and the local-checkpoint rules are reused from
run.py rather than duplicated, so the two entry points cannot drift apart.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import build_parser, route_and_run  # noqa: E402
from src.utils.dataset_sample import CAPNAV_SAMPLE_200  # noqa: E402


def main() -> None:
    parser = build_parser()
    parser.description = (
        "CapNav sample runner: same arguments as run.py, but evaluates only the "
        f"curated '{CAPNAV_SAMPLE_200.name}' scene subset "
        f"({len(CAPNAV_SAMPLE_200.scenes)} scenes)."
    )
    args = parser.parse_args()

    if args.model_path and not args.backend:
        raise ValueError("When using --model_path, you must also specify --backend.")
    if not args.model_path and not args.model:
        raise ValueError("When not using --model_path, you must provide --model (Hugging Face id).")

    print(f"[SAMPLE] scenes: {', '.join(CAPNAV_SAMPLE_200.scenes)}")

    route_and_run(
        model=args.model,
        num_frames=args.num_frames,
        thinking=args.thinking,
        model_path=args.model_path,
        backend=args.backend,
        scenes_allowlist=list(CAPNAV_SAMPLE_200.scenes),
    )


if __name__ == "__main__":
    main()
