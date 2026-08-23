# src/model_adapters/spatial_mllm_adapter.py

import os
import re
import sys
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch

from src.utils.scene_select import resolve_scenes
from src.utils.output_parsing import extract_records


# ============================================================
# Paths (read from env; fallback to repo-relative defaults)
# ============================================================

PROMPT_ROOT = os.environ.get("CAPNAV_PROMPT_ROOT", "generated_prompts")
GRAPH_DIR = os.environ.get("CAPNAV_GRAPH_DIR", "dataset/ground_truth/graphs")
VIDEO_ROOT  = os.environ.get("CAPNAV_VIDEO_ROOT", "videos_64frames_1fps")  # fixed for open-source videos
RESULT_ROOT = os.environ.get("CAPNAV_RESULT_ROOT", "results")

# Spatial-MLLM repo location (code repo, NOT weights)
# Users should set SPATIAL_MLLM_ROOT in .env if they want to use this adapter.
SPATIAL_MLLM_ROOT = os.environ.get("SPATIAL_MLLM_ROOT", "")


# ============================================================
# Optional debug: print HF cache env (user-managed; do NOT set)
# ============================================================

def _print_hf_cache_env_if_debug() -> None:
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
# Scene/prompt/video helpers
# ============================================================

def load_prompts(prompt_root: str, scene: str) -> List[Tuple[str, str]]:
    scene_dir = os.path.join(prompt_root, scene)
    if not os.path.isdir(scene_dir):
        print(f"[WARN] Prompt folder not found: {scene_dir}")
        return []

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
# Spatial-MLLM import bootstrap
# ============================================================

def _ensure_spatial_mllm_src_on_path(spatial_mllm_root: str) -> str:
    """
    Spatial-MLLM's code lives under <Spatial-MLLM repo>/src.
    Users are expected to clone it and set SPATIAL_MLLM_ROOT accordingly.
    """
    if not spatial_mllm_root:
        raise FileNotFoundError(
            "SPATIAL_MLLM_ROOT is not set.\n"
            "To use this adapter, please clone Spatial-MLLM and set SPATIAL_MLLM_ROOT in your .env.\n"
            "Example:\n"
            "  SPATIAL_MLLM_ROOT=/absolute/path/to/Spatial-MLLM"
        )

    src_dir = os.path.join(spatial_mllm_root, "src")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            "Spatial-MLLM repo not found.\n"
            f"Expected: {src_dir}\n"
            "Please verify SPATIAL_MLLM_ROOT points to a cloned Spatial-MLLM repo."
        )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        print(f"[PATH] Added Spatial-MLLM src to sys.path: {src_dir}")

    return src_dir


# ============================================================
# Failure logging
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
# Model init + single prompt runner
# ============================================================

def init_spatial_mllm(user_model: str, device: str = "cuda"):
    """
    user_model can be:
      - HF id (e.g., Diankun/Spatial-MLLM-subset-sft)
      - local dir path
    We allow auto-download via from_pretrained; cache path is user-managed via HF env vars.
    """
    _ensure_spatial_mllm_src_on_path(SPATIAL_MLLM_ROOT)

    # These imports rely on Spatial-MLLM's repo code under <root>/src
    from models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore

    _print_hf_cache_env_if_debug()
    print(f"[MODEL] Loading Spatial-MLLM from: {user_model} (auto-download if HF id; cache is user-managed)")

    model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
        user_model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)

    processor = Qwen2_5_VLProcessor.from_pretrained(user_model)
    return model, processor, process_vision_info


def run_single_prompt(
    model,
    processor,
    process_vision_info_fn,
    video_path: str,
    text: str,
    num_frames: int,
    device: str = "cuda",
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": int(num_frames)},
                {"type": "text", "text": text},
            ],
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info_fn(messages)

    inputs = processor(text=[text_input], videos=video_inputs, padding=True, return_tensors="pt")
    inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
    inputs = inputs.to(device)

    torch.cuda.empty_cache()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8196,
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
            use_cache=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return (output_texts[0] or "").strip()


# ============================================================
# Public entry point
# ============================================================

def run_spatial_mllm(
    user_model: str,
    num_frames: int,
    thinking: str = "on",
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Adapter API (matches scripts/run.py routing signature):
      - user_model: HF id or local dir
      - num_frames: 16/32/64 (controls Spatial-MLLM message nframes)
      - thinking: must be "on" (Spatial-MLLM only supported in thinking mode here)
    """
    thinking_norm = (thinking or "").lower().strip()
    if thinking_norm != "on":
        raise ValueError(
            "Invalid --thinking for Spatial-MLLM.\n"
            "Spatial-MLLM is currently only supported in thinking mode. Please use: --thinking on"
        )

    prompt_root = PROMPT_ROOT
    graph_dir   = GRAPH_DIR
    video_root  = VIDEO_ROOT
    result_root = RESULT_ROOT

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    if not torch.cuda.is_available():
        raise RuntimeError("Spatial-MLLM adapter requires a CUDA-capable GPU environment.")

    device = "cuda"

    model, processor, process_vision_info_fn = init_spatial_mllm(user_model, device=device)

    scenes = resolve_scenes(
        graph_dir,
        scenes_allowlist=scenes_allowlist,
        strict=True,
    )
    print(f"[SCENES] running {len(scenes)} scenes" + (" (allowlisted)" if scenes_allowlist else ""))

    model_tag = f"spatial_mllm_{num_frames}frames_thinking_on"
    base_out = os.path.join(result_root, "spatial_mllm", model_tag)

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        if not prompts:
            continue

        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(base_out, scene)
        os.makedirs(out_dir, exist_ok=True)
        fail_path = os.path.join(out_dir, "failed_prompts.jsonl")
        print(f"\n[SCENE] {scene} | prompts={len(prompts)} | out={out_dir}")

        for i, (fname, ptext) in enumerate(prompts, 1):
            stem = os.path.splitext(fname)[0]
            out_file = os.path.join(out_dir, f"{stem}.json")

            if os.path.exists(out_file):
                continue

            t0 = time.time()
            entry: Dict[str, Any] = {
                "scene": scene,
                "prompt_file": fname,
                "user_model": user_model,
                "num_frames": num_frames,
                "thinking": "on",
            }

            try:
                raw_text = run_single_prompt(
                    model=model,
                    processor=processor,
                    process_vision_info_fn=process_vision_info_fn,
                    video_path=video_path,
                    text=ptext,
                    num_frames=num_frames,
                    device=device,
                )
                entry["raw_text"] = raw_text
                entry["time_sec"] = round(time.time() - t0, 2)

                records, cleaned, parse_error = extract_records(raw_text, ptext)
                if parse_error is None:
                    entry["result"] = records
                else:
                    # Keep the cleaned text so the failure can be re-parsed
                    # offline rather than re-running the model.
                    entry["result"] = None
                    entry["parse_error"] = parse_error
                    entry["json_str"] = cleaned[:20000]
                    log_failure(fail_path, stem, parse_error, time.time() - t0)

            except Exception as e:
                entry["raw_text"] = ""
                entry["result"] = None
                entry["error"] = repr(e)
                entry["time_sec"] = round(time.time() - t0, 2)
                log_failure(fail_path, stem, repr(e), time.time() - t0)
                torch.cuda.empty_cache()

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

        # Per-prompt files are the single source of truth: capnav_score.py
        # scans every *.json under results/, so a per-scene roll-up would be
        # re-counted as an unparsable record.
        print(f"[DONE] Scene completed -> {out_dir}")
