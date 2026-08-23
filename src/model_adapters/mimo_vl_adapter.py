import os
import re
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from src.utils.scene_select import resolve_scenes
from src.utils.output_parsing import extract_records


# ============================================================
# Paths (read from env; fallback to repo-relative defaults)
# ============================================================

PROMPT_ROOT = os.environ.get("CAPNAV_PROMPT_ROOT", "generated_prompts")
GRAPH_DIR = os.environ.get("CAPNAV_GRAPH_DIR", "dataset/ground_truth/graphs")
VIDEO_ROOT  = os.environ.get("CAPNAV_VIDEO_ROOT", "videos_64frames_1fps")  
RESULT_ROOT = os.environ.get("CAPNAV_RESULT_ROOT", "results")

DEFAULT_ORG = "XiaomiMiMo"

# MiMo-VL uses /no_think for disabling reasoning
NO_THINK_SUFFIX = "/no_think"

# Video control: the source videos are fixed at 64 frames, 1 FPS
BASE_FPS = 1.0
TOTAL_FRAMES = 64

# Keep internal (do not expose to CLI)
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_DELAY = 0.0


# ============================================================
# HF cache awareness (user-managed; we do NOT set or override)
# ============================================================

def _print_hf_cache_env_if_debug() -> None:
    """
    Debug/visibility only. Does not change behavior.
    Users control HF cache via HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE, etc.
    Enable by setting CAPNAV_DEBUG_ENV=1.
    """
    if os.environ.get("CAPNAV_DEBUG_ENV") != "1":
        return

    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_ENDPOINT",
        "HF_TOKEN",
    ]
    print("[CapNav] HF cache / hub env (user-managed):")
    found_any = False
    for k in keys:
        v = os.environ.get(k)
        if v:
            if k == "HF_TOKEN":
                print(f"  {k}=<set>")
            else:
                print(f"  {k}={v}")
            found_any = True
    if not found_any:
        print("  (none set) -> will use Hugging Face default cache location")


# ============================================================
# Model id helpers
# ============================================================

def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


def normalize_hf_model_id(user_model: str) -> str:
    """
    Allows:
      - "MiMo-VL-7B-RL-2508" -> "XiaomiMiMo/MiMo-VL-7B-RL-2508"
      - "XiaomiMiMo/MiMo-VL-7B-RL-2508" -> keep as-is
    """
    if "/" in user_model:
        return user_model
    return f"{DEFAULT_ORG}/{user_model}"


# ============================================================
# Scene/prompt/video helpers
# ============================================================


def load_prompts(prompt_root: str, scene: str) -> List[Tuple[str, str]]:
    scene_dir = os.path.join(prompt_root, scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Prompt folder not found: {scene_dir}")

    files = sorted([f for f in os.listdir(scene_dir) if f.endswith(".txt")])
    out: List[Tuple[str, str]] = []
    for fname in files:
        with open(os.path.join(scene_dir, fname), "r", encoding="utf-8") as fp:
            out.append((fname, fp.read().strip()))
    return out


def get_video_path(video_root: str, scene: str) -> str:
    v = os.path.join(video_root, f"{scene}.mp4")
    if not os.path.exists(v):
        raise FileNotFoundError(f"Video not found: {v}")
    return v


# ============================================================
# Output parsing helpers (aligned with your previous behavior)
# ============================================================

def log_failure(fail_path: str, prompt_name: str, error_message: str, elapsed: float) -> None:
    record = {
        "prompt": prompt_name,
        "error": error_message,
        "time_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# MiMo init + single run
# ============================================================

def init_mimo(hf_model_id: str):
    """
    MiMo-VL uses Qwen2.5-VL architecture in transformers.

    IMPORTANT:
    - Passing a HF model id will auto-download weights if missing.
    - Cache location is fully user-managed via HF_HOME / HF_HUB_CACHE / etc.
    """
    _print_hf_cache_env_if_debug()
    print(f"[MODEL] loading from HF: {hf_model_id} (auto-download; cache is user-managed)")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_id,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(hf_model_id)
    return model, processor


def num_frames_to_fps(num_frames: int) -> float:
    """
    Videos are stored as 64 frames @ 1 FPS.
    To sample fewer frames from the same video, reduce fps proportionally:
      64 -> 1.0
      32 -> 0.5
      16 -> 0.25
    """
    if num_frames not in (16, 32, 64):
        raise ValueError("--num_frames must be one of {16, 32, 64}.")
    return float(BASE_FPS) * (float(num_frames) / float(TOTAL_FRAMES))


def maybe_append_no_think(prompt_text: str, thinking: str) -> str:
    """
    Strict equivalence to your old code:
      - thinking=off => enforce trailing ' /no_think'
      - thinking=on  => do not modify prompt
    """
    thinking_norm = thinking.lower().strip()
    if thinking_norm == "off":
        p = (prompt_text or "").strip()
        if not p.endswith(NO_THINK_SUFFIX):
            p = p + " " + NO_THINK_SUFFIX
        return p
    if thinking_norm == "on":
        return (prompt_text or "").strip()
    raise ValueError("--thinking must be one of {on, off}.")


def run_single_prompt(
    model,
    processor,
    video_path: str,
    prompt_text: str,
    fps: float,
    max_new_tokens: int,
) -> str:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "path": os.path.abspath(video_path)},
            {"type": "text", "text": prompt_text},
        ],
    }]

    inputs = processor.apply_chat_template(
        conversation,
        fps=fps,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=20,
            temperature=0.3,
            repetition_penalty=1.0,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text


# ============================================================
# Public entry point (adapter API)
# ============================================================

def run_mimo_vl(
    user_model: str,
    num_frames: int,
    thinking: str,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Public entry point:
      user_model: "MiMo-VL-7B-RL-2508" or "XiaomiMiMo/MiMo-VL-7B-RL-2508"
      num_frames: 16 / 32 / 64  (controls fps sampling from the SAME video)
      thinking:   on / off      (on: strip <think>; off: append /no_think)
    """
    hf_model_id = normalize_hf_model_id(user_model)
    model_name = model_basename(hf_model_id)

    # CapNav paths (env-configurable)
    prompt_root = PROMPT_ROOT
    graph_dir   = GRAPH_DIR
    video_root  = VIDEO_ROOT
    result_root = RESULT_ROOT

    # Required paths
    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    # init model (HF auto-download; user-managed cache)
    model, processor = init_mimo(hf_model_id)

    fps = num_frames_to_fps(num_frames)
    print(f"[VIDEO] total_frames={TOTAL_FRAMES} base_fps={BASE_FPS} num_frames={num_frames} -> fps={fps}")

    scenes = resolve_scenes(
        graph_dir,
        scenes_allowlist=scenes_allowlist,
        strict=True,
    )
    print(f"[SCENES] running {len(scenes)} scenes" + (" (allowlisted)" if scenes_allowlist else ""))

    thinking_tag = thinking.lower().strip()
    if thinking_tag not in ("on", "off"):
        raise ValueError("--thinking must be one of {on, off}.")

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, f"{model_name}_{num_frames}frames_{thinking_tag}", scene)
        os.makedirs(out_dir, exist_ok=True)
        fail_path = os.path.join(out_dir, "failed_prompts.jsonl")

        print(f"\n[SCENE] {scene} | prompts={len(prompts)}")

        for fname, ptext in prompts:
            prompt_stem = os.path.splitext(fname)[0]
            out_file = os.path.join(out_dir, f"{prompt_stem}.json")
            if os.path.exists(out_file):
                continue

            t0 = time.time()
            entry: Dict[str, Any] = {
                "scene": scene,
                "prompt_file": fname,
                "model": model_name,
                "hf_model_id": hf_model_id,
                "num_frames": num_frames,
                "fps": fps,
                "thinking": thinking_tag,
            }

            try:
                prompt_used = maybe_append_no_think(ptext, thinking=thinking)
                raw_text = run_single_prompt(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    prompt_text=prompt_used,
                    fps=fps,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                )

                entry["raw_text"] = raw_text
                entry["time_sec"] = round(time.time() - t0, 2)

                records, cleaned, parse_error = extract_records(raw_text, ptext)
                if parse_error is None:
                    entry["result"] = records
                else:
                    entry["result"] = None
                    entry["parse_error"] = parse_error
                    entry["json_str"] = cleaned[:20000]
                    log_failure(fail_path, prompt_stem, parse_error, time.time() - t0)

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)

            except Exception as e:
                entry["time_sec"] = round(time.time() - t0, 2)
                entry["error"] = repr(e)
                entry["result"] = None
                log_failure(fail_path, prompt_stem, repr(e), time.time() - t0)

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)

                torch.cuda.empty_cache()

            if DEFAULT_DELAY > 0:
                time.sleep(DEFAULT_DELAY)
