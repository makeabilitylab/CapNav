import os
import re
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration

from src.utils.scene_select import resolve_scenes
from src.utils.output_parsing import extract_records


DEFAULT_ORG = "zai-org"

# ============================================================
# Paths (read from env; fallback to repo-relative defaults)
# ============================================================

PROMPT_ROOT = os.environ.get("CAPNAV_PROMPT_ROOT", "generated_prompts")
GRAPH_DIR = os.environ.get("CAPNAV_GRAPH_DIR", "dataset/ground_truth/graphs")
VIDEO_ROOT  = os.environ.get("CAPNAV_VIDEO_ROOT", "videos_64frames_1fps")
RESULT_ROOT = os.environ.get("CAPNAV_RESULT_ROOT", "results")


# ============================================================
# HF cache awareness (user-managed; we do NOT set or override)
# ============================================================

def _print_hf_cache_env_if_debug() -> None:
    """
    Debug only. Users control HF cache via:
      - HF_HOME
      - HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE
      - TRANSFORMERS_CACHE
      - HF_ENDPOINT, HF_TOKEN, etc.
    Enable by setting CAPNAV_DEBUG_ENV=1 in .env.
    """
    if os.environ.get("CAPNAV_DEBUG_ENV") != "1":
        return

    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
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

def normalize_hf_model_id(user_model: str) -> str:
    """
    Allow users to pass either:
      - "GLM-4.1V-9B-Thinking"  -> zai-org/GLM-4.1V-9B-Thinking
      - "zai-org/GLM-4.1V-9B-Thinking" -> keep as-is
    """
    if "/" in user_model:
        return user_model
    return f"{DEFAULT_ORG}/{user_model}"


def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


# ============================================================
# Scene/prompt/video helpers
# ============================================================


def load_prompts(prompt_root: str, scene: str) -> List[Tuple[str, str]]:
    scene_dir = os.path.join(prompt_root, scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(
            f"Prompt folder not found for scene '{scene}': {scene_dir}"
        )

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
# Model init
# ============================================================

def init_model(hf_model_id: str, base_fps: int, total_frames: int, num_frames: int):
    """
    IMPORTANT:
    - Passing a HF model id will auto-download weights if missing.
    - Cache location is fully user-managed via HF_HOME / HF_HUB_CACHE / etc.
    """
    _print_hf_cache_env_if_debug()
    print(f"[MODEL] loading from HF: {hf_model_id} (auto-download; cache is user-managed)")

    model = Glm4vForConditionalGeneration.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(hf_model_id, use_fast=True)

    fps = (
        float(base_fps)
        if num_frames >= total_frames
        else float(base_fps) * (float(total_frames) / float(num_frames))
    )

    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = fps
        processor.video_processor.num_frames = num_frames

    print(
        f"[VIDEO] base_fps={base_fps} total_frames={total_frames} "
        f"num_frames={num_frames} -> effective_fps={fps:.2f}"
    )
    return model, processor, fps


# ============================================================
# Single run
# ============================================================

def run_one(model, processor, video_path: str, prompt_text: str, fps: float, max_new_tokens: int) -> Dict[str, Any]:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "video_url": os.path.abspath(video_path)},
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
            top_p=0.9,
            temperature=0.3,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    raw_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # Decoded with special tokens kept so the <think>/<answer> structure
    # survives into raw_text; extract_records() unwraps it at parse time.
    return {"raw_text": raw_text}


# ============================================================
# Public entry point
# ============================================================

def run_glm4v_thinking(
    user_model: str,
    num_frames: int,
    thinking: str = "on",
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    GLM-4.1V-9B-Thinking runner.

    thinking:
      - "on": supported (default)
      - "off": NOT supported, will raise an error
    """
    thinking_norm = thinking.lower().strip()
    if thinking_norm not in ("on", "off"):
        raise ValueError("--thinking must be one of {on, off}.")

    if thinking_norm == "off":
        raise ValueError(
            "GLM-4.1V-9B-Thinking does not support thinking=off "
            "(no-think mode is unavailable for this model)."
        )

    hf_model_id = normalize_hf_model_id(user_model)
    model_name = model_basename(hf_model_id)

    prompt_root = PROMPT_ROOT
    graph_dir   = GRAPH_DIR
    video_root  = VIDEO_ROOT
    result_root = RESULT_ROOT

    base_fps = 1
    total_frames = 64
    max_new_tokens = 8192
    delay = 0.0

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    model, processor, fps = init_model(
        hf_model_id,
        base_fps=base_fps,
        total_frames=total_frames,
        num_frames=num_frames
    )

    scenes = resolve_scenes(
        graph_dir,
        scenes_allowlist=scenes_allowlist,
        strict=True,
    )
    print(f"[SCENES] running {len(scenes)} scenes" + (" (allowlisted)" if scenes_allowlist else ""))

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, f"{model_name}_{num_frames}frames", scene)
        os.makedirs(out_dir, exist_ok=True)
        fail_path = os.path.join(out_dir, "failed_prompts.jsonl")

        print(f"\n[SCENE] {scene} | prompts={len(prompts)}")

        for fname, prompt_text in prompts:
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
                "thinking": thinking_norm,
            }

            try:
                out = run_one(
                    model,
                    processor,
                    video_path,
                    prompt_text,
                    fps=fps,
                    max_new_tokens=max_new_tokens,
                )
                entry["raw_text"] = out["raw_text"]
                entry["time_sec"] = round(time.time() - t0, 2)

                records, cleaned, parse_error = extract_records(out["raw_text"], prompt_text)
                if parse_error is None:
                    entry["result"] = records
                else:
                    entry["result"] = None
                    entry["parse_error"] = parse_error
                    entry["json_str"] = cleaned[:20000]
                    log_failure(fail_path, prompt_stem, parse_error, time.time() - t0)

            except Exception as e:
                # One bad prompt (OOM, decode error) must not abort a 45-scene run.
                entry["time_sec"] = round(time.time() - t0, 2)
                entry["error"] = repr(e)
                entry["result"] = None
                log_failure(fail_path, prompt_stem, repr(e), time.time() - t0)
                torch.cuda.empty_cache()

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            if delay > 0:
                time.sleep(delay)
