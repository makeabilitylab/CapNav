#!/usr/bin/env python3
"""
scripts/run.py

CapNav unified runner (Scheme 1):
  - Default: evaluate via Hugging Face repo ids (strict allowlists/patterns).
  - Optional: evaluate via local checkpoint directory (--model_path) with explicit adapter selection (--backend).

Security / safety goals:
  1) No shell execution, no arbitrary downloads beyond Hugging Face's official libraries.
  2) Strict model id validation to prevent typos/hangs and to keep the benchmark scope controlled.
  3) Local checkpoints require explicit backend to avoid fuzzy inference and accidental adapter mismatch.
  4) Optional .env loading (repo-root/.env) for HF cache / token configuration; never overwrites user env.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, List

# ----------------------------
# Repo root (since this file is scripts/run.py)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ----------------------------
# Load .env early (repo-root/.env), optional
# ----------------------------
def _load_dotenv_if_present(repo_root: Path) -> None:
    """
    Loads repo_root/.env if python-dotenv is installed.
    - override=False: never overrides existing environment variables.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return

    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


_load_dotenv_if_present(REPO_ROOT)

# ----------------------------
# Import adapters
# ----------------------------
from src.model_adapters.glm4v_thinking_adapter import run_glm4v_thinking
from src.model_adapters.internvl3_5_adapter import run_internvl3_5
from src.model_adapters.mimo_vl_adapter import run_mimo_vl
from src.model_adapters.qwen3_vl_adapter import run_qwen3_vl
from src.model_adapters.spatial_mllm_adapter import run_spatial_mllm
from src.model_adapters.videor1_adapter import run_videor1

# ----------------------------
# Strict allowlist / patterns (HF repo ids only)
# ----------------------------

# GLM: exact allowlist only
ALLOWED_MODELS_EXACT = {
    "GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.1V-9B-Thinking",
}

# InternVL3_5: strict prefix + safe chars, scalable across sizes/variants
INTERNVL3_5_PATTERN = re.compile(r"^(?:OpenGVLab/)?InternVL3_5-[A-Za-z0-9._-]+$")

# MiMo-VL: strict prefix + safe chars, scalable across sizes/variants
MIMOVL_PATTERN = re.compile(r"^(?:XiaomiMiMo/)?MiMo-VL-[A-Za-z0-9._-]+$")

# Qwen3-VL: strict prefix + safe chars, scalable across sizes/variants
QWEN3_VL_PATTERN = re.compile(r"^(?:Qwen/)?Qwen3-VL-[A-Za-z0-9._-]+$")

# Spatial-MLLM: exact allowlist (copy-paste friendly; prevents HF hangs due to typos)
ALLOWED_SPATIAL_MLLM_EXACT = {
    "Diankun/Spatial-MLLM-subset-sft",
}

# Video-R1: exact allowlist (pinned)
ALLOWED_VIDEOR1_EXACT = {
    "Video-R1/Video-R1-7B",
}

# ----------------------------
# Backend enum for local checkpoints (Scheme 1)
# ----------------------------
BACKENDS = ("glm", "internvl", "mimo", "qwen3", "spatial_mllm", "videor1")


# ----------------------------
# Canonicalization (HF ids)
# ----------------------------
def canonicalize_model(model: str) -> str:
    """
    Convert allowed aliases to canonical identifiers for downstream adapters.
    IMPORTANT: No fuzzy matching. Input must already be valid by exact allowlist
    or strict regex patterns.
    """
    # GLM: allow org form and canonicalize to non-org for routing
    if model == "zai-org/GLM-4.1V-9B-Thinking":
        return "GLM-4.1V-9B-Thinking"

    # InternVL3_5: canonicalize to OpenGVLab/<...>
    if INTERNVL3_5_PATTERN.match(model):
        return model if model.startswith("OpenGVLab/") else f"OpenGVLab/{model}"

    # MiMo-VL: canonicalize to XiaomiMiMo/<...>
    if MIMOVL_PATTERN.match(model):
        return model if model.startswith("XiaomiMiMo/") else f"XiaomiMiMo/{model}"

    # Qwen3-VL: canonicalize to Qwen/<...>
    if QWEN3_VL_PATTERN.match(model):
        return model if model.startswith("Qwen/") else f"Qwen/{model}"

    # Spatial-MLLM + Video-R1 are exact allowlists, keep as-is
    return model


# ----------------------------
# Type checks (HF ids only)
# ----------------------------
def _is_glm(model: str) -> bool:
    return model in ALLOWED_MODELS_EXACT


def _is_internvl3_5(model: str) -> bool:
    return INTERNVL3_5_PATTERN.match(model) is not None


def _is_mimo_vl(model: str) -> bool:
    return MIMOVL_PATTERN.match(model) is not None


def _is_qwen3_vl(model: str) -> bool:
    return QWEN3_VL_PATTERN.match(model) is not None


def _is_spatial_mllm(model: str) -> bool:
    return model in ALLOWED_SPATIAL_MLLM_EXACT


def _is_videor1(model: str) -> bool:
    return model in ALLOWED_VIDEOR1_EXACT


# ----------------------------
# Qwen3-VL thinking mode validation (HF ids)
# ----------------------------
def _qwen3_vl_checkpoint_mode(canonical_model: str) -> str:
    """
    Determine whether the Qwen3-VL checkpoint is Thinking or Instruct.
    Returns: "thinking" or "instruct"
    """
    name = canonical_model.split("/", 1)[-1]  # remove org
    if "Thinking" in name:
        return "thinking"
    if "Instruct" in name:
        return "instruct"
    raise ValueError(
        "Qwen3-VL checkpoint name must include either 'Thinking' or 'Instruct'.\n"
        f"Received: {canonical_model}\n"
        "Examples:\n"
        "  - Qwen/Qwen3-VL-30B-A3B-Thinking\n"
        "  - Qwen/Qwen3-VL-30B-A3B-Instruct\n"
        "  - Qwen/Qwen3-VL-30B-A3B-Thinking-FP8\n"
    )


def _enforce_qwen3_vl_thinking_hf(canonical_model: str, thinking: str) -> None:
    mode = _qwen3_vl_checkpoint_mode(canonical_model)
    thinking_norm = thinking.lower().strip()

    if mode == "thinking" and thinking_norm != "on":
        raise ValueError(
            "Invalid --thinking for this Qwen3-VL checkpoint.\n"
            f"Model: {canonical_model}\n"
            "This is a Thinking checkpoint, so you must use: --thinking on"
        )

    if mode == "instruct" and thinking_norm != "off":
        raise ValueError(
            "Invalid --thinking for this Qwen3-VL checkpoint.\n"
            f"Model: {canonical_model}\n"
            "This is an Instruct checkpoint (no-think), so you must use: --thinking off"
        )


# ----------------------------
# Spatial-MLLM thinking validation
# ----------------------------
def _enforce_spatial_mllm_thinking(model: str, thinking: str) -> None:
    if thinking.lower().strip() != "on":
        raise ValueError(
            "Invalid --thinking for Spatial-MLLM.\n"
            f"Model: {model}\n"
            "Spatial-MLLM is currently only supported in thinking mode. Please use: --thinking on"
        )


# ----------------------------
# Video-R1 thinking validation
# ----------------------------
def _enforce_videor1_thinking(model: str, thinking: str) -> None:
    if thinking.lower().strip() != "on":
        raise ValueError(
            "Invalid --thinking for Video-R1.\n"
            f"Model: {model}\n"
            "Video-R1 is currently only supported in thinking mode in this repo. Please use: --thinking on"
        )


# ----------------------------
# Local checkpoint helpers (Scheme 1)
# ----------------------------
def _validate_local_checkpoint(model_path: str) -> str:
    p = Path(model_path).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(
            "Invalid --model_path.\n"
            f"Received: {model_path}\n"
            "Expected an existing local checkpoint directory."
        )
    return str(p.resolve())


def _route_local(
    backend: str,
    model_path: str,
    num_frames: int,
    thinking: str,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Local checkpoint routing: requires explicit --backend to avoid fuzzy inference.
    - No downloads should be triggered if the underlying adapters call from_pretrained(local_path).
    """
    backend_norm = (backend or "").lower().strip()
    if backend_norm not in BACKENDS:
        raise ValueError(
            "When using --model_path, you must specify --backend.\n"
            f"Allowed: {', '.join(BACKENDS)}\n"
            f"Received: {backend}"
        )

    # Enforce thinking constraints even for local usage
    if backend_norm == "glm":
        if thinking != "on":
            raise ValueError(
                'Invalid --thinking for GLM-4.1V-9B-Thinking.\n'
                'This model has no "off" mode. Please use: --thinking on'
            )
        run_glm4v_thinking(
            user_model=model_path,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if backend_norm == "internvl":
        run_internvl3_5(
            user_model=model_path,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if backend_norm == "mimo":
        run_mimo_vl(
            user_model=model_path,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if backend_norm == "qwen3":
        # Adapter may enforce Thinking/Instruct via checkpoint naming or internal config.
        run_qwen3_vl(
            user_model=model_path,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if backend_norm == "spatial_mllm":
        _enforce_spatial_mllm_thinking(model_path, thinking)
        run_spatial_mllm(
            user_model=model_path,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if backend_norm == "videor1":
        _enforce_videor1_thinking(model_path, thinking)
        run_videor1(
            user_model=model_path,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    raise RuntimeError(f"Internal local routing error: backend={backend_norm}")


# ----------------------------
# Routing (HF ids)
# ----------------------------
def _route_hf(
    model: str,
    num_frames: int,
    thinking: str,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    # 1) Strict validation (HF ids only)
    if (
        (model not in ALLOWED_MODELS_EXACT)
        and (not _is_internvl3_5(model))
        and (not _is_mimo_vl(model))
        and (not _is_qwen3_vl(model))
        and (not _is_spatial_mllm(model))
        and (not _is_videor1(model))
    ):
        allowed_glm = "\n  - ".join(sorted(ALLOWED_MODELS_EXACT))
        allowed_spatial = "\n  - ".join(sorted(ALLOWED_SPATIAL_MLLM_EXACT))
        allowed_videor1 = "\n  - ".join(sorted(ALLOWED_VIDEOR1_EXACT))
        raise ValueError(
            "Unsupported --model value.\n"
            f"Received: {model}\n\n"
            "Allowed GLM values (exact match):\n"
            f"  - {allowed_glm}\n\n"
            "Allowed InternVL3_5 format (regex strict):\n"
            "  - InternVL3_5-<CHECKPOINT>\n"
            "  - OpenGVLab/InternVL3_5-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n\n'
            "Allowed MiMo-VL format (regex strict):\n"
            "  - MiMo-VL-<CHECKPOINT>\n"
            "  - XiaomiMiMo/MiMo-VL-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n\n'
            "Allowed Qwen3-VL format (regex strict):\n"
            "  - Qwen3-VL-<CHECKPOINT>\n"
            "  - Qwen/Qwen3-VL-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n'
            "    and the checkpoint name must include 'Thinking' or 'Instruct'.\n\n"
            "Allowed Spatial-MLLM values (exact match):\n"
            f"  - {allowed_spatial}\n\n"
            "Allowed Video-R1 values (exact match):\n"
            f"  - {allowed_videor1}\n"
        )

    canonical = canonicalize_model(model)

    # 2) Routing
    if canonical == "GLM-4.1V-9B-Thinking":
        if thinking != "on":
            raise ValueError(
                'Invalid --thinking for GLM-4.1V-9B-Thinking.\n'
                'This model has no "off" mode. Please use: --thinking on'
            )
        run_glm4v_thinking(
            user_model=canonical,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if canonical.startswith("OpenGVLab/InternVL3_5-"):
        run_internvl3_5(
            user_model=canonical,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if canonical.startswith("XiaomiMiMo/MiMo-VL-"):
        run_mimo_vl(
            user_model=canonical,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if canonical.startswith("Qwen/Qwen3-VL-"):
        _enforce_qwen3_vl_thinking_hf(canonical, thinking)
        run_qwen3_vl(
            user_model=canonical,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if canonical in ALLOWED_SPATIAL_MLLM_EXACT:
        _enforce_spatial_mllm_thinking(canonical, thinking)
        run_spatial_mllm(
            user_model=canonical,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if canonical in ALLOWED_VIDEOR1_EXACT:
        _enforce_videor1_thinking(canonical, thinking)
        run_videor1(
            user_model=canonical,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    raise RuntimeError(f"Internal routing error for model: {model} (canonical={canonical})")


def route_and_run(
    model: Optional[str],
    num_frames: int,
    thinking: str,
    model_path: Optional[str] = None,
    backend: Optional[str] = None,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Scheme 1:
      - Default: route by HF --model (strict allowlists/patterns)
      - Optional: --model_path + --backend for local checkpoints (no downloads)

    scenes_allowlist:
      - None: run all scenes (default behavior)
      - List[str]: run only these scenes (used by run_sample.py)
    """
    if model_path:
        local = _validate_local_checkpoint(model_path)
        _route_local(
            backend=backend or "",
            model_path=local,
            num_frames=num_frames,
            thinking=thinking,
            scenes_allowlist=scenes_allowlist,
        )
        return

    if not model:
        raise ValueError("When not using --model_path, you must provide --model (Hugging Face id).")

    _route_hf(
        model=model,
        num_frames=num_frames,
        thinking=thinking,
        scenes_allowlist=scenes_allowlist,
    )


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    """Shared by run.py and run_sample.py so the two CLIs cannot drift."""
    parser = argparse.ArgumentParser(
        description="CapNav runner (open-source models, strict argument enforcement)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Hugging Face model id (strict). Required unless --model_path is provided.\n"
            "Either one of the allowed GLM ids, or a checkpoint id matching:\n"
            "  - InternVL3_5-...  (or OpenGVLab/InternVL3_5-...)\n"
            "  - MiMo-VL-...      (or XiaomiMiMo/MiMo-VL-...)\n"
            "  - Qwen3-VL-...     (or Qwen/Qwen3-VL-...; must contain Thinking or Instruct)\n"
            "  - Spatial-MLLM     (exact allowlist; see error message)\n"
            "  - Video-R1         (exact allowlist; see error message)\n\n"
            "If --model_path is provided, --model is optional and only used for logging."
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Optional local checkpoint directory. If set, the runner loads weights from disk "
            "and does NOT trigger Hugging Face downloads. Requires --backend."
        ),
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=BACKENDS,
        help=(
            "Required when using --model_path. Select which adapter to use for the local checkpoint: "
            + ", ".join(BACKENDS)
        ),
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        choices=[16, 32, 64],
        help="Must be one of {16, 32, 64}.",
    )

    parser.add_argument(
        "--thinking",
        type=str,
        required=True,
        choices=["on", "off"],
        help='Must be explicitly provided: "on" or "off".',
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Enforce Scheme 1 constraints
    if args.model_path:
        if not args.backend:
            raise ValueError("When using --model_path, you must also specify --backend.")
    else:
        if not args.model:
            raise ValueError("When not using --model_path, you must provide --model (Hugging Face id).")

    route_and_run(
        model=args.model,
        num_frames=args.num_frames,
        thinking=args.thinking,
        model_path=args.model_path,
        backend=args.backend,
        scenes_allowlist=None,
    )


if __name__ == "__main__":
    main()
