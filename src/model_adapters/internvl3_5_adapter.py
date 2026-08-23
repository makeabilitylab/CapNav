import os
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from src.utils.scene_select import resolve_scenes
from src.utils.output_parsing import extract_records


# ============================================================
# Paths (read from env; fallback to repo-relative defaults)
# ============================================================

PROMPT_ROOT = os.environ.get("CAPNAV_PROMPT_ROOT", "generated_prompts")
GRAPH_DIR = os.environ.get("CAPNAV_GRAPH_DIR", "dataset/ground_truth/graphs")
VIDEO_ROOT  = os.environ.get("CAPNAV_VIDEO_ROOT", "videos_64frames_1fps")  
RESULT_ROOT = os.environ.get("CAPNAV_RESULT_ROOT", "results")

DEFAULT_ORG = "OpenGVLab"

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


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
# Model id helpers (keep your semantics)
# ============================================================

def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


def normalize_hf_model_id(user_model: str) -> str:
    """
    Allows:
      - "InternVL3_5-8B" -> "OpenGVLab/InternVL3_5-8B"
      - "OpenGVLab/InternVL3_5-8B" -> keep as-is
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
# InternVL video preprocessing (ported from your code)
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed


def load_video_frames(
    video_path: str,
    input_size: int = 448,
    max_num_tiles: int = 1,
    num_segments: int = 64,
) -> Tuple[torch.Tensor, List[int]]:
    """
    IMPORTANT:
      - video is ALWAYS read from videos_64frames_1fps/<scene>.mp4
      - num_segments controls how many frames to sample (16/32/64)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    frame_indices = np.linspace(0, max_frame, num_segments, dtype=int)

    transform = build_transform(input_size)
    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []

    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num_tiles)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


# ============================================================
# Output parsing helpers
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
# InternVL init + runner
# ============================================================

def init_internvl(hf_model_id: str, thinking: str):
    """
    IMPORTANT:
    - Passing a HF model id will auto-download weights if missing.
    - Cache location is fully user-managed via HF_HOME / HF_HUB_CACHE / etc.
    """
    _print_hf_cache_env_if_debug()
    print(f"[MODEL] loading from HF: {hf_model_id} (auto-download; cache is user-managed)")

    model = AutoModel.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_config = dict(
        max_new_tokens=8196,
        do_sample=True,
        temperature=0.6,
    )

    thinking_norm = thinking.lower().strip()
    if thinking_norm == "on":
        model.system_message = R1_SYSTEM_PROMPT
    elif thinking_norm == "off":
        pass
    else:
        raise ValueError("--thinking must be one of {on, off}.")
    return model, tokenizer, generation_config


def run_internvl3_5(
    user_model: str,
    num_frames: int,
    thinking: str,
    scenes_allowlist: Optional[List[str]] = None,
) -> None:
    """
    Public entry point:
      user_model: "OpenGVLab/InternVL3_5-8B" or "InternVL3_5-8B" (normalized)
      num_frames: 16 / 32 / 64  (sampling count from the SAME video)
      thinking:   on / off
    """
    hf_model_id = normalize_hf_model_id(user_model)
    model_name = model_basename(hf_model_id)

    prompt_root = PROMPT_ROOT
    graph_dir   = GRAPH_DIR
    video_root  = VIDEO_ROOT
    result_root = RESULT_ROOT

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    model, tokenizer, generation_config = init_internvl(hf_model_id, thinking=thinking)

    scenes = resolve_scenes(
        graph_dir,
        scenes_allowlist=scenes_allowlist,
        strict=True,
    )
    print(f"[SCENES] running {len(scenes)} scenes" + (" (allowlisted)" if scenes_allowlist else ""))

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, f"{model_name}_{num_frames}frames_{thinking.lower()}", scene)
        os.makedirs(out_dir, exist_ok=True)
        fail_path = os.path.join(out_dir, "failed_prompts.jsonl")

        print(f"\n[SCENE] {scene} | prompts={len(prompts)}")

        # Load sampled frames ONCE per scene
        pixel_values, num_patches_list = load_video_frames(video_path, num_segments=num_frames, max_num_tiles=1)

        # Avoid device mismatch
        first_device = next(model.parameters()).device
        pixel_values = pixel_values.to(torch.bfloat16).to(first_device)

        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])

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
                "thinking": thinking.lower().strip(),
            }

            try:
                question = video_prefix + "\n" + ptext
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )

                raw_text = (response or "").strip()
                entry["raw_text"] = raw_text
                entry["time_sec"] = round(time.time() - t0, 2)

                # Strict equivalence:
                # - thinking=on: strip <think>...</think>
                # - thinking=off: do NOT strip
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
