import os
import re
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from src.utils.scene_select import resolve_scenes
from src.utils.output_parsing import extract_records


# ============================================================
# Paths (read from env; fallback to repo-relative defaults)
# ============================================================

PROMPT_ROOT = os.environ.get("CAPNAV_PROMPT_ROOT", "generated_prompts")
GRAPH_DIR = os.environ.get("CAPNAV_GRAPH_DIR", "dataset/ground_truth/graphs")
VIDEO_ROOT  = os.environ.get("CAPNAV_VIDEO_ROOT", "videos_64frames_1fps")
RESULT_ROOT = os.environ.get("CAPNAV_RESULT_ROOT", "results")


# ============================================================
# IO helpers
# ============================================================


def load_prompts(prompt_root: str, scene: str) -> List[Tuple[str, str]]:
    scene_dir = os.path.join(prompt_root, scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Prompt folder not found: {scene_dir}")

    files = sorted([f for f in os.listdir(scene_dir) if f.endswith(".txt")])
    out: List[Tuple[str, str]] = []
    for fname in files:
        fpath = os.path.join(scene_dir, fname)
        with open(fpath, "r", encoding="utf-8") as fp:
            out.append((fname, fp.read().strip()))
    return out


def get_video_path(video_root: str, scene: str) -> str:
    v = os.path.join(video_root, f"{scene}.mp4")
    if not os.path.exists(v):
        raise FileNotFoundError(f"Video not found: {v}")
    return v


def log_failure(
    fail_path: str,
    prompt_name: str,
    error_message: str,
    elapsed: float,
) -> None:
    record = {
        "prompt": prompt_name,
        "error": error_message,
        "time_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Failure logging
# ============================================================

# ============================================================
# Prompt template
# ============================================================

def build_prompt(custom_prompt: str) -> str:
    return (
        custom_prompt
        + "\n\n"
        + "Please think about this question as if you were a human pondering deeply. "
          "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc. "
          "Include self-reflection or verification within <think> and </think> tags. "
          "Then give your final JSON answer inside <answer></answer>."
    )


# ============================================================
# Video-R1 init
# ============================================================

def init_videor1_llm(model_name: str):
    """
    NOTE: assumes the user has already installed the official Video-R1 environment
    (vllm, pinned transformers zip, etc.). No auto setup here.
    """
    print(f"[Video-R1] Loading model: {model_name}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=81920,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1, "image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=8196,  
    )

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    # Reduce repeated rescale warnings 
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_rescale = False
    if hasattr(processor, "video_processor"):
        processor.video_processor.do_rescale = False

    return llm, processor, sampling_params


# ============================================================
# Public entry point
# ============================================================

def run_videor1(
    user_model: str,
    num_frames: int,
    thinking: str,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Public entry point for scripts/run.py

    user_model: e.g. "Video-R1/Video-R1-7B"
    num_frames: 16 / 32 / 64  (sampling count from videos_64frames_1fps/<scene>.mp4)
    thinking:   MUST be "on" for this adapter
    """
    thinking_norm = (thinking or "").lower().strip()
    if thinking_norm != "on":
        raise ValueError(
            "Invalid --thinking for Video-R1 adapter.\n"
            "This adapter is designed for Thinking mode only.\n"
            "Please use: --thinking on"
        )

    prompt_root = PROMPT_ROOT
    graph_dir   = GRAPH_DIR
    video_root  = VIDEO_ROOT
    result_root = RESULT_ROOT

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    # init model
    llm, processor, sampling_params = init_videor1_llm(user_model)

    scenes = resolve_scenes(
        graph_dir,
        scenes_allowlist=scenes_allowlist,
        strict=True,
    )
    print(f"[Video-R1] running {len(scenes)} scenes from {graph_dir}" + (" (allowlisted)" if scenes_allowlist else ""))

    model_name = user_model.split("/", 1)[-1]
    model_tag = f"{model_name}_{num_frames}frames_{thinking_norm}"

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, "thinking", model_tag, scene)
        os.makedirs(out_dir, exist_ok=True)
        fail_path = os.path.join(out_dir, "failed_prompts.jsonl")

        print(f"\n[SCENE] {scene} | prompts={len(prompts)} | video={os.path.basename(video_path)}")

        for i, (fname, ptext) in enumerate(prompts, 1):
            prompt_stem = os.path.splitext(fname)[0]
            out_file = os.path.join(out_dir, f"{prompt_stem}.json")

            # resume
            if os.path.exists(out_file):
                print(f"[SKIP] {fname} (already exists).")
                continue

            print(f"🚀 [{i}/{len(prompts)}] {fname}")
            t0 = time.time()

            entry: Dict[str, Any] = {
                "scene": scene,
                "prompt_file": fname,
                "model": model_name,
                "hf_model_id": user_model,
                "num_frames": num_frames,
                "thinking": thinking_norm,
            }

            try:
                final_prompt = build_prompt(ptext)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": 102400,
                                "nframes": num_frames,
                            },
                            {"type": "text", "text": final_prompt},
                        ],
                    }
                ]

                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                _image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )

                llm_inputs = [{
                    "prompt": prompt,
                    "multi_modal_data": {"video": video_inputs[0]},
                    "mm_processor_kwargs": {k: v[0] for k, v in video_kwargs.items()},
                }]

                outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
                raw_text = outputs[0].outputs[0].text.strip()

                entry["raw_text"] = raw_text
                entry["time_sec"] = round(time.time() - t0, 2)

                records, cleaned, parse_error = extract_records(raw_text, ptext)
                if parse_error is None:
                    entry["result"] = records
                else:
                    # `result: null` is how every adapter reports an unparsable
                    # answer, so failure counts stay comparable across models.
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
                # fail log
                log_failure(fail_path, prompt_stem, repr(e), time.time() - t0)

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)

        # Per-prompt files are the single source of truth: capnav_score.py
        # scans every *.json under results/, so a per-scene roll-up would be
        # re-counted as an unparsable record.
        print(f"[SCENE DONE] {scene} -> {out_dir}")
