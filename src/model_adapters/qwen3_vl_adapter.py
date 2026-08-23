import os
import re
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from src.utils.scene_select import resolve_scenes
from src.utils.output_parsing import extract_records


# ============================================================
# Paths (read from env; fallback to repo-relative defaults)
# ============================================================

PROMPT_ROOT = os.environ.get("CAPNAV_PROMPT_ROOT", "generated_prompts")
GRAPH_DIR = os.environ.get("CAPNAV_GRAPH_DIR", "dataset/ground_truth/graphs")
VIDEO_ROOT  = os.environ.get("CAPNAV_VIDEO_ROOT", "videos_64frames_1fps") 
RESULT_ROOT = os.environ.get("CAPNAV_RESULT_ROOT", "results")

DEFAULT_ORG = "Qwen"

# Video folder semantics in this repo:
BASE_FPS = 1.0
TOTAL_FRAMES = 64  # videos_64frames_1fps

# Strict token detection (segment match, not suffix match)
_THINKING_TOKEN = re.compile(r"(?:^|-)Thinking(?:-|$)")
_INSTRUCT_TOKEN = re.compile(r"(?:^|-)Instruct(?:-|$)")


# ============================================================
# HF cache awareness (user-managed; we do NOT set or override)
# ============================================================

def _print_hf_cache_env_if_debug() -> None:
    """
    For transparency/debugging only.
    Does not change behavior. Users control cache via HF_HOME / HF_HUB_CACHE / etc.
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
            # avoid printing token
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

def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


def normalize_hf_model_id(user_model: str) -> str:
    """
    Allows:
      - "Qwen3-VL-30B-A3B-Thinking-FP8" -> "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
      - "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8" -> keep as-is
    """
    if "/" in user_model:
        return user_model
    return f"{DEFAULT_ORG}/{user_model}"


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
# Qwen3-VL think/no-think capability inference (STRICT)
# ============================================================

def infer_required_thinking_from_checkpoint(hf_model_id: str) -> str:
    """
    Returns:
      - "on"  if checkpoint name contains '-Thinking-' segment
      - "off" if checkpoint name contains '-Instruct-' segment
    Raises if neither / both found.
    """
    name = model_basename(hf_model_id)

    has_thinking = _THINKING_TOKEN.search(name) is not None
    has_instruct = _INSTRUCT_TOKEN.search(name) is not None

    if has_thinking and has_instruct:
        raise ValueError(
            f"Ambiguous Qwen3-VL checkpoint name (contains both Thinking and Instruct): {hf_model_id}"
        )
    if has_thinking:
        return "on"
    if has_instruct:
        return "off"

    raise ValueError(
        "Qwen3-VL checkpoint name must contain a '-Thinking-' or '-Instruct-' segment "
        f"to determine think mode. Got: {hf_model_id}"
    )


# ============================================================
# Output parsing helpers (keep consistent with other adapters)
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
# Model init + single run
# ============================================================

def init_qwen3_vl(hf_model_id: str):
    """
    IMPORTANT:
    - Passing a Hugging Face model id will automatically download weights if missing.
    - Cache location is fully user-managed via HF_HOME / HF_HUB_CACHE / etc.
    """
    _print_hf_cache_env_if_debug()
    print(f"[MODEL] loading from HF: {hf_model_id} (auto-download; cache is user-managed)")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_model_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
    return model, processor


def generation_params_for_mode(required_thinking: str) -> Dict[str, Any]:
    """
    You required:
      - no-think (Instruct): top_p=0.8, top_k=20, temperature=0.7
      - think (Thinking):   top_p=0.95, top_k=20, temperature=1.0
    And unify max_new_tokens=8196.
    """
    if required_thinking == "off":
        return {"top_p": 0.8, "top_k": 20, "temperature": 0.7}
    return {"top_p": 0.95, "top_k": 20, "temperature": 1.0}


def run_one(
    model,
    processor,
    video_path: str,
    prompt_text: str,
    fps: float,
    max_new_tokens: int,
    required_thinking: str,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": os.path.abspath(video_path),
                    "fps": fps,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 256 * 32 * 32,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kw = generation_params_for_mode(required_thinking)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=gen_kw["top_p"],
            top_k=gen_kw["top_k"],
            temperature=gen_kw["temperature"],
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    raw_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return {"raw_text": raw_text}


# ============================================================
# Public entry point
# ============================================================

def run_qwen3_vl(
    user_model: str,
    num_frames: int,
    thinking: str,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Public entry point:
      user_model: "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8" OR "Qwen3-VL-30B-A3B-Thinking-FP8"
      num_frames: 16 / 32 / 64  (sampling count from the SAME video under videos_64frames_1fps)
      thinking:   on / off  (MUST match checkpoint type)
    """
    thinking_norm = thinking.lower().strip()
    if thinking_norm not in {"on", "off"}:
        raise ValueError('--thinking must be exactly "on" or "off".')

    hf_model_id = normalize_hf_model_id(user_model)
    model_name = model_basename(hf_model_id)

    # Determine required mode from checkpoint naming
    required_thinking = infer_required_thinking_from_checkpoint(hf_model_id)

    # Enforce user-provided thinking matches checkpoint
    if thinking_norm != required_thinking:
        if required_thinking == "on":
            raise ValueError(
                f"Invalid --thinking for {hf_model_id}.\n"
                "This is a Thinking checkpoint, it only supports: --thinking on"
            )
        raise ValueError(
            f"Invalid --thinking for {hf_model_id}.\n"
            "This is an Instruct checkpoint, it only supports: --thinking off"
        )

    # CapNav paths (env-configurable)
    prompt_root = PROMPT_ROOT
    graph_dir   = GRAPH_DIR
    video_root  = VIDEO_ROOT
    result_root = RESULT_ROOT

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    # init model (HF auto-download; user-managed cache)
    model, processor = init_qwen3_vl(hf_model_id)

    # IMPORTANT: videos are fixed 64 frames @ 1fps in repo
    # We emulate fps-based behavior:
    #   frames = 64 * fps  => fps = num_frames / 64
    fps = float(BASE_FPS) * (float(num_frames) / float(TOTAL_FRAMES))
    print(
        f"[VIDEO] base_fps={BASE_FPS} total_frames={TOTAL_FRAMES} "
        f"num_frames={num_frames} -> effective_fps={fps:.4f}"
    )

    max_new_tokens = 8196

    scenes = resolve_scenes(
        graph_dir,
        scenes_allowlist=scenes_allowlist,
        strict=True,
    )
    print(f"[SCENES] running {len(scenes)} scenes from {graph_dir}" + (" (allowlisted)" if scenes_allowlist else ""))

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, f"{model_name}_{num_frames}frames_{thinking_norm}", scene)
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
                "thinking": thinking_norm,
                "effective_fps": fps,
            }

            try:
                out = run_one(
                    model,
                    processor,
                    video_path=video_path,
                    prompt_text=ptext,
                    fps=fps,
                    max_new_tokens=max_new_tokens,
                    required_thinking=required_thinking,
                )

                entry["raw_text"] = out["raw_text"]
                entry["time_sec"] = round(time.time() - t0, 2)

                records, cleaned, parse_error = extract_records(out["raw_text"], ptext)
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
