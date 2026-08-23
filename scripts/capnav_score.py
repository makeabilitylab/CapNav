#!/usr/bin/env python3
import json
import re
import string
import os
import time
from pathlib import Path
from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


# =========================================================
# CONFIG (NO CLI ARGS)
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# run.py default output root
RESULT_ROOT = Path(os.environ.get("CAPNAV_RESULT_ROOT", str(REPO_ROOT / "results")))

# open-source GT layout
GT_ROOT = Path(os.environ.get("CAPNAV_GT_ROOT", str(REPO_ROOT / "dataset" / "ground_truth")))

# output under RESULT_ROOT
OUT_PER_RECORD = RESULT_ROOT / "scored_per_record.jsonl"
OUT_SUMMARY = RESULT_ROOT / "scored_summary.json"

# RV LLM model
RV_LLM_MODEL = os.environ.get("CAPNAV_RV_MODEL", "gpt-5.2").strip() or "gpt-5.2"

_SCENE_CACHE: Dict[str, Tuple[Dict, Dict]] = {}


# =========================================================
# IO: discover record json files under results/
# =========================================================
def discover_record_files(result_root: Path) -> List[Path]:
    """
    Recursively find per-prompt record files under results/.

    A record file is results/<model_setting>/<scene>/<prompt_stem>.json. Score
    outputs and any aggregate roll-ups are skipped: their top level is a JSON
    list, not a record, so counting them would add phantom structural failures
    to every rate reported below.
    """
    if not result_root.exists():
        raise FileNotFoundError(f"RESULT_ROOT not found: {result_root}. Please run run.py first.")

    exclude = {OUT_SUMMARY.name, OUT_PER_RECORD.name, "scored_summary.json", "scored_per_record.jsonl"}
    skip_suffixes = ("_results.json", "_failed.json", "_summary.json")

    files: List[Path] = []
    for p in result_root.rglob("*.json"):
        if p.name in exclude:
            continue
        if p.name.startswith("_") or p.name.endswith(skip_suffixes):
            continue
        files.append(p)
    return sorted(files)


def safe_json_load(path: Path) -> Optional[Dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def infer_model_setting_dir(record_path: Path) -> Optional[str]:
    """
    Expected layout:
      results/<model_setting>/<scene>/<prompt_stem>.json
    model_setting = parent of scene dir
    """
    try:
        return record_path.parent.parent.name
    except Exception:
        return None


# =========================================================
# Text normalization
# =========================================================
def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


# =========================================================
# Parse VLM output: Answer + Path (+ optional Fail/Reason)
# (kept from your score.py)
# =========================================================
def parse_vlm_output_text(output: str) -> Dict:
    if not isinstance(output, str) or not output.strip():
        return {"ok": False, "error": "empty_output", "raw": output}

    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    # Accept: Answer: (A|B) True|False | Path: ...
    m = re.search(
        r"Answer:\s*\((A|B)\)\s*(True|False)\s*\|\s*Path:\s*(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not m:
        return {"ok": False, "error": "cannot_parse_answer_path", "raw": cleaned}

    answer = "yes" if m.group(2).lower() == "true" else "no"
    path_str = m.group(3).strip()

    # Cut off Fail/Reason tail if present
    path_str = re.split(r"\s*\|\s*(?:Fail|Reason)\s*:", path_str, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    path_names = [p.strip() for p in re.split(r"\s*(?:->|→)\s*", path_str) if p.strip()]

    if answer == "yes" and len(path_names) < 2:
        return {"ok": False, "error": "yes_without_path", "raw": cleaned}

    return {"ok": True, "answer": answer, "path_names": path_names, "raw": cleaned}


# =========================================================
# Reverse match src/tgt from question via graph node names
# (kept from your score.py)
# =========================================================
def find_from_to_indices(qn: str) -> Optional[Tuple[int, int]]:
    m_from = re.search(r"\bfrom\b", qn)
    if not m_from:
        return None
    m_to = re.search(r"\bto\b", qn[m_from.end():])
    if not m_to:
        return None
    to_start = m_from.end() + m_to.start()
    return (m_from.end(), to_start)


def all_node_hits_in_question(qn: str, node_names: List[str]) -> List[Dict]:
    hits = []
    for name in node_names:
        nn = norm_text(name)
        if not nn:
            continue
        for m in re.finditer(rf"\b{re.escape(nn)}\b", qn):
            hits.append(
                {
                    "name": name,
                    "start": m.start(),
                    "end": m.end(),
                    "length": len(nn),
                    "center": (m.start() + m.end()) / 2.0,
                }
            )
    return hits


def derive_src_tgt_by_reverse_match(question: str, name_to_id: Dict[str, str]) -> Dict:
    if not isinstance(question, str) or not question.strip():
        return {"ok": False, "error": "empty_question"}

    qn = norm_text(question)
    node_names = list(name_to_id.keys())

    hits = all_node_hits_in_question(qn, node_names)
    if not hits:
        return {"ok": False, "error": "no_node_name_hit_in_question"}

    span_idx = find_from_to_indices(qn)
    if not span_idx:
        hits_sorted = sorted(hits, key=lambda h: (h["start"], -h["length"]))
        start_hit = hits_sorted[0]
        end_hit = hits_sorted[-1]
        if start_hit["name"] == end_hit["name"] and len(hits_sorted) >= 2:
            end_hit = hits_sorted[-2]
        return {
            "ok": True,
            "src_name": start_hit["name"],
            "tgt_name": end_hit["name"],
            "src_id": name_to_id[start_hit["name"]],
            "tgt_id": name_to_id[end_hit["name"]],
            "method": "fallback_first_last",
        }

    from_end, to_start = span_idx
    start_cands = [h for h in hits if from_end <= h["center"] <= to_start]
    end_cands = [h for h in hits if h["center"] >= to_start]

    def score_start(h):
        return (-abs(h["center"] - from_end) + 0.08 * h["length"])

    def score_end(h):
        return (-abs(h["center"] - to_start) + 0.08 * h["length"])

    if start_cands and end_cands:
        start_hit = max(start_cands, key=score_start)
        end_hit = max(end_cands, key=score_end)
        if start_hit["name"] == end_hit["name"] and len(end_cands) > 1:
            end_sorted = sorted(end_cands, key=score_end, reverse=True)
            for h in end_sorted:
                if h["name"] != start_hit["name"]:
                    end_hit = h
                    break
        return {
            "ok": True,
            "src_name": start_hit["name"],
            "tgt_name": end_hit["name"],
            "src_id": name_to_id[start_hit["name"]],
            "tgt_id": name_to_id[end_hit["name"]],
            "method": "from_to_partition",
        }

    after_from = [h for h in hits if h["center"] >= from_end]
    if not after_from:
        return {"ok": False, "error": "no_hits_after_from"}

    start_hit = min(after_from, key=lambda h: abs(h["center"] - from_end) - 0.05 * h["length"])
    end_hit = min(after_from, key=lambda h: abs(h["center"] - to_start) - 0.05 * h["length"])
    if start_hit["name"] == end_hit["name"]:
        alt = sorted(after_from, key=lambda h: abs(h["center"] - to_start) - 0.05 * h["length"])
        for h in alt:
            if h["name"] != start_hit["name"]:
                end_hit = h
                break

    return {
        "ok": True,
        "src_name": start_hit["name"],
        "tgt_name": end_hit["name"],
        "src_id": name_to_id[start_hit["name"]],
        "tgt_id": name_to_id[end_hit["name"]],
        "method": "degraded_closest",
    }


# =========================================================
# Load graph + traverse
# =========================================================
def load_graph_traverse(scene: str) -> Tuple[Dict, Dict]:
    if scene in _SCENE_CACHE:
        return _SCENE_CACHE[scene]

    graph_path = GT_ROOT / "graphs" / f"{scene}-graph.json"
    traverse_path = GT_ROOT / "traverse" / f"{scene}-traverse.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")
    if not traverse_path.exists():
        raise FileNotFoundError(f"Missing traverse: {traverse_path}")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    traverse = json.loads(traverse_path.read_text(encoding="utf-8"))
    _SCENE_CACHE[scene] = (graph, traverse)
    return graph, traverse


def build_graph_index(graph: Dict) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    name_to_id = {n["name"]: n["id"] for n in graph["nodes"]}
    adj = {n["id"]: [] for n in graph["nodes"]}
    edge_set = set()
    for e in graph["edges"]:
        u, v = e["from"], e["to"]
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))  # undirected
    return name_to_id, adj, edge_set


# =========================================================
# Traversability (RTA)
# =========================================================
def agent_key(traverse: Dict, agent: str) -> Optional[str]:
    a = (agent or "").upper()
    if a in traverse:
        return a
    if a == "HUMANOID" and "ROBOT" in traverse:
        return "ROBOT"
    return None


def edge_traversable(traverse: Dict, agent: str, u: str, v: str) -> bool:
    k = agent_key(traverse, agent)
    if not k:
        return False
    ad = traverse[k]
    return bool(ad.get(f"{u}|{v}", ad.get(f"{v}|{u}", {})).get("traversable", False))


def traversability_score(traverse: Dict, agent: str, path_ids: List[str]) -> Optional[float]:
    if not path_ids or len(path_ids) < 2:
        return None
    total = len(path_ids) - 1
    ok = 0
    for i in range(total):
        if edge_traversable(traverse, agent, path_ids[i], path_ids[i + 1]):
            ok += 1
    return ok / total


# =========================================================
# PV validity
# =========================================================
def pv_strict(path_ids: List[Optional[str]], src_id: str, tgt_id: str, edge_set: set) -> Tuple[bool, List[str]]:
    errors = []
    if not path_ids:
        return False, ["empty_path"]
    if any(x is None for x in path_ids):
        errors.append("unresolved_node_in_path")
    if path_ids[0] != src_id:
        errors.append("path_start_not_src")
    if path_ids[-1] != tgt_id:
        errors.append("path_end_not_tgt")

    clean = [x for x in path_ids if x is not None]
    if len(clean) != len(set(clean)):
        errors.append("not_simple_path_repeated_nodes")

    for i in range(len(path_ids) - 1):
        u, v = path_ids[i], path_ids[i + 1]
        if u is None or v is None:
            continue
        if (u, v) not in edge_set:
            errors.append(f"missing_edge:{u}->{v}")

    return (len(errors) == 0), errors


# =========================================================
# GT doable: exists traversable path (agent-conditioned BFS)
# =========================================================
def exists_traversable_path(adj: Dict[str, List[str]], traverse: Dict, agent: str, src: str, tgt: str) -> bool:
    if src == tgt:
        return True
    dq = deque([src])
    seen = {src}
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v in seen:
                continue
            if not edge_traversable(traverse, agent, u, v):
                continue
            if v == tgt:
                return True
            seen.add(v)
            dq.append(v)
    return False


# =========================================================
# Map predicted path node names -> node ids (legacy fallback)
# =========================================================
def best_match_pred_name(pred_name: str, node_names: List[str]) -> Optional[str]:
    if not pred_name:
        return None
    pn = norm_text(pred_name)
    if not pn:
        return None

    for n in node_names:
        if norm_text(n) == pn:
            return n

    contain = []
    for n in node_names:
        nn = norm_text(n)
        if nn and (nn in pn or pn in nn):
            contain.append((len(nn), n))
    if contain:
        contain.sort(reverse=True)
        return contain[0][1]

    best = (0.0, None)
    for n in node_names:
        nn = norm_text(n)
        r = SequenceMatcher(None, pn, nn).ratio()
        if r > best[0]:
            best = (r, n)
    if best[0] >= 0.78:
        return best[1]
    return None


# =========================================================
# RV: LLM-as-judge (infeasible predictions only)
# =========================================================
def extract_vlm_reason_from_raw(raw_output: str) -> str:
    """
    Extract rationale from " | Fail: ..." or " | Reason: ..."
    If missing, return "".
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ""
    m = re.search(r"\|\s*(fail|reason)\s*:\s*(.+)$", raw_output, flags=re.IGNORECASE)
    if not m:
        return ""
    return m.group(2).strip()


def collect_failure_notes_along_path(traverse: Dict, agent: str, path_ids: List[str]) -> List[str]:
    """
    Collect GT notes for edges on proposed route P_hat that are NOT traversable.
    If note missing, fall back to a generic edge message.
    Dedup while preserving order.
    """
    notes: List[str] = []
    k = agent_key(traverse, agent)
    if not k or not path_ids or len(path_ids) < 2:
        return notes

    ad = traverse[k]
    for i in range(len(path_ids) - 1):
        u, v = path_ids[i], path_ids[i + 1]
        info = ad.get(f"{u}|{v}") or ad.get(f"{v}|{u}") or {}
        traversable = bool(info.get("traversable", False))
        if traversable:
            continue
        note = info.get("note")
        if isinstance(note, str) and note.strip():
            notes.append(note.strip())
        else:
            notes.append(f"Edge {u}->{v} not traversable for {k}")

    seen = set()
    out: List[str] = []
    for n in notes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def evaluate_reasoning_with_openai(vlm_reason: str, failure_notes: List[str]) -> Dict:
    """
    LLM-as-judge. Returns:
      { ok: bool, correct: bool, explanation: str, llm_response: str }
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "correct": False,
            "explanation": "OPENAI_API_KEY not set; cannot compute RV",
            "llm_response": "",
        }

    try:
        from openai import OpenAI, RateLimitError
    except Exception as e:
        return {
            "ok": False,
            "correct": False,
            "explanation": f"OpenAI SDK not available: {e}",
            "llm_response": "",
        }

    # Missing rationale => RV = 0 (counted)
    if not vlm_reason:
        return {
            "ok": True,
            "correct": False,
            "explanation": "No reasoning provided by VLM",
            "llm_response": "",
        }

    notes_text = "\n".join([f"- {n}" for n in failure_notes if n])
    if not notes_text:
        notes_text = "- (No edge-level failure notes available.)"

    prompt = f"""You are evaluating whether a navigation model's failure reasoning is correct.

Model's rationale (why it believes the route is infeasible):
"{vlm_reason}"

Ground-truth failure notes for the proposed route (edge-level constraints for this agent):
{notes_text}

Decide whether the model's rationale semantically matches the ground-truth failure notes.
- It does NOT need to match wording exactly.
- It SHOULD capture the main constraint(s).
- If the rationale is vague or unrelated, mark incorrect.

Respond with JSON only:
{{
  "correct": true/false,
  "explanation": "brief explanation"
}}
"""

    client = OpenAI(api_key=api_key)

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=RV_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=200,
            )
            txt = (resp.choices[0].message.content or "").strip()
            if not txt:
                return {
                    "ok": True,
                    "correct": False,
                    "explanation": f"Empty response from {RV_LLM_MODEL}",
                    "llm_response": "",
                }
            try:
                obj = json.loads(txt)
                return {
                    "ok": True,
                    "correct": bool(obj.get("correct", False)),
                    "explanation": str(obj.get("explanation", "")),
                    "llm_response": txt,
                }
            except json.JSONDecodeError:
                return {
                    "ok": True,
                    "correct": False,
                    "explanation": "Non-JSON response",
                    "llm_response": txt,
                }

        except RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return {
                "ok": True,
                "correct": False,
                "explanation": f"RateLimitError: {e}",
                "llm_response": "",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return {
                "ok": True,
                "correct": False,
                "explanation": f"API error: {e}",
                "llm_response": "",
            }

    return {
        "ok": True,
        "correct": False,
        "explanation": "Max retries exceeded",
        "llm_response": "",
    }


# =========================================================
# Helpers: synthesize legacy output text from structured {answer,path}
# (so your existing parser + downstream logic stays the same)
# =========================================================
def synthesize_output_text(answer: str, path_ids: List[str]) -> str:
    ans = (answer or "").strip().lower()
    is_true = ans in {"yes", "true", "a", "(a)"}
    tf = "True" if is_true else "False"
    letter = "A" if is_true else "B"
    path_str = " -> ".join([str(x) for x in (path_ids or [])])
    return f"Answer: ({letter}) {tf} | Path: {path_str}"


def looks_like_node_id(s: str) -> bool:
    return isinstance(s, str) and bool(re.fullmatch(r"node_\d+", s.strip()))


# =========================================================
# Convert one run output JSON file -> list[rec] for scorer
# expected format:
# {
#   "scene": "...",
#   "prompt_file": "...",
#   "result": [ { "question":..., "agent":..., "result": {"answer":..., "path":[...]} } ],
#   "time_sec": ...
# }
# =========================================================
def records_from_output_file(path: Path) -> Tuple[List[Dict], Optional[str]]:
    outer = safe_json_load(path)
    if outer is None:
        return [], "invalid_json"

    scene = outer.get("scene")
    prompt_file = outer.get("prompt_file")
    time_sec = outer.get("time_sec")
    items = outer.get("result")

    if not scene or not isinstance(scene, str):
        return [], "missing_scene"
    if items is None:
        return [], "missing_result"
    if not isinstance(items, list):
        return [], "result_not_list"

    model_setting = infer_model_setting_dir(path)

    recs: List[Dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        agent = item.get("agent")
        inner = item.get("result") or {}
        if not isinstance(inner, dict):
            inner = {}

        ans = inner.get("answer")
        pth = inner.get("path") or []
        if not isinstance(pth, list):
            pth = []
        pth = [str(x) for x in pth]

        out_text = synthesize_output_text(str(ans or ""), pth)

        qid = f"{path.stem}:{i}"
        recs.append(
            {
                "qid": qid,
                "scene": scene,
                "agent": agent,
                "question": q,
                "output": out_text,
                "entry_idx": i,
                "status": None,
                # provenance
                "source_file": str(path),
                "model_setting": model_setting,
                "prompt_file": prompt_file,
                "time_sec": time_sec,
            }
        )

    if not recs:
        return [], "empty_result_items_or_missing_fields"
    return recs, None


# =========================================================
# Score one record (F1/PV/RTA unchanged; RV now real)
# =========================================================
def score_record(rec: Dict) -> Dict:
    qid = rec.get("qid")
    scene = rec.get("scene")
    agent = rec.get("agent")
    question = rec.get("question")
    output = rec.get("output", "")
    entry_idx = rec.get("entry_idx")
    status = rec.get("status")

    source_file = rec.get("source_file")
    model_setting = rec.get("model_setting")
    prompt_file = rec.get("prompt_file")
    time_sec = rec.get("time_sec")

    base = {
        "qid": qid,
        "scene": scene,
        "agent": agent,
        "entry_idx": entry_idx,
        "status": status,
        "question": question,
        # provenance
        "source_file": source_file,
        "model_setting": model_setting,
        "prompt_file": prompt_file,
        "time_sec": time_sec,
    }

    if not scene or not agent or not question:
        return {**base, "ok": False, "error": "missing_scene_agent_or_question"}

    graph, traverse = load_graph_traverse(scene)
    name_to_id, adj, edge_set = build_graph_index(graph)
    node_names = list(name_to_id.keys())
    node_id_set = set(adj.keys())

    st = derive_src_tgt_by_reverse_match(question, name_to_id)
    if not st["ok"]:
        return {**base, "ok": False, "error": st["error"], "derive_method": st.get("method")}

    src_id, tgt_id = st["src_id"], st["tgt_id"]
    gt_doable = exists_traversable_path(adj, traverse, agent, src_id, tgt_id)

    parsed = parse_vlm_output_text(output)
    if not parsed["ok"]:
        binary_bucket = "FN" if gt_doable else "FP"
        return {
            **base,
            "ok": True,
            "format_error": True,
            "format_error_type": parsed["error"],
            "derive_method": st.get("method"),
            "src_name": st.get("src_name"),
            "tgt_name": st.get("tgt_name"),
            "src_id": src_id,
            "tgt_id": tgt_id,
            "vlm_answer": "fail",
            "vlm_path_names_raw": [],
            "vlm_path_names_mapped": [],
            "vlm_path_ids": [],
            "pv_valid": None,
            "pv_errors": [],
            "rta_applicable": False,
            "traversability_score": None,
            "rv_applicable": False,
            "rv_score": None,
            "rv_judge_explanation": None,
            "rv_llm_model": RV_LLM_MODEL,
            "ground_truth_doable": gt_doable,
            "correct_answer": False,
            "binary_bucket": binary_bucket,
        }

    # If path tokens are node IDs that exist in this scene graph, use them directly.
    use_ids_directly = False
    if parsed["path_names"] and all(looks_like_node_id(x) for x in parsed["path_names"]):
        if all(x in node_id_set for x in parsed["path_names"]):
            use_ids_directly = True

    if use_ids_directly:
        pred_ids: List[Optional[str]] = [x for x in parsed["path_names"]]
        pred_mapped_names: List[Optional[str]] = [None for _ in pred_ids]
    else:
        pred_ids = []
        pred_mapped_names = []
        for pn in parsed["path_names"]:
            bn = best_match_pred_name(pn, node_names)
            pred_mapped_names.append(bn)
            pred_ids.append(name_to_id[bn] if bn else None)

    pv_valid, pv_errors = pv_strict(pred_ids, src_id, tgt_id, edge_set)

    # RTA applicable: y_hat=1 and PV valid
    rta_applicable = (parsed["answer"] == "yes" and pv_valid)
    trav = None
    if rta_applicable:
        trav = traversability_score(traverse, agent, [x for x in pred_ids if x is not None])

    # RV applicable: y_hat=0; must return proposed route P_hat and short rationale
    rv_applicable = (parsed["answer"] == "no")
    rv_score = None
    rv_expl = None

    if rv_applicable:
        # proposed route P_hat: require at least 2 nodes and all resolvable
        clean_ids = [x for x in pred_ids if x is not None]
        if len(clean_ids) >= 2:
            vlm_reason = extract_vlm_reason_from_raw(parsed["raw"])
            failure_notes = collect_failure_notes_along_path(traverse, agent, clean_ids)
            ev = evaluate_reasoning_with_openai(vlm_reason, failure_notes)

            # RV is required for your composite (/4). If API unavailable -> structural fail.
            if not ev.get("ok", False):
                return {**base, "ok": False, "error": ev.get("explanation", "rv_eval_failed")}

            rv_score = 1.0 if ev.get("correct", False) else 0.0
            rv_expl = ev.get("explanation", "")
        else:
            # y_hat=0 but no usable route => RV=0 (counted)
            rv_score = 0.0
            rv_expl = "No valid proposed route for RV"

    correct_answer = ((parsed["answer"] == "yes" and gt_doable) or (parsed["answer"] == "no" and not gt_doable))

    return {
        **base,
        "ok": True,
        "format_error": False,
        "format_error_type": None,
        "binary_bucket": None,
        "derive_method": st.get("method"),
        "src_name": st.get("src_name"),
        "tgt_name": st.get("tgt_name"),
        "src_id": src_id,
        "tgt_id": tgt_id,
        "vlm_answer": parsed["answer"],
        "vlm_path_names_raw": parsed["path_names"],
        "vlm_path_names_mapped": pred_mapped_names,
        "vlm_path_ids": pred_ids,
        "pv_valid": pv_valid,
        "pv_errors": pv_errors,
        "rta_applicable": rta_applicable,
        "traversability_score": trav,
        "rv_applicable": rv_applicable,
        "rv_score": rv_score,
        "rv_judge_explanation": rv_expl,
        "rv_llm_model": RV_LLM_MODEL,
        "ground_truth_doable": gt_doable,
        "correct_answer": correct_answer,
    }


# =========================================================
# Main
# =========================================================
def main():
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(f"RESULT_ROOT not found: {RESULT_ROOT}. Please run run.py first.")
    if not GT_ROOT.exists():
        raise FileNotFoundError(f"GT_ROOT not found: {GT_ROOT}. Expected dataset/ground_truth with graphs/ and traverse/.")

    # RV requires OpenAI key (your requirement: composite uses RV)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. RV is required for capnav_score (/4).")

    record_files = discover_record_files(RESULT_ROOT)
    if not record_files:
        raise FileNotFoundError(f"No *.json record files found under: {RESULT_ROOT}")

    print("Scoring record JSON files under:")
    print(f"  RESULT_ROOT = {RESULT_ROOT}")
    print(f"  GT_ROOT     = {GT_ROOT}")
    print(f"  RV model    = {RV_LLM_MODEL}")
    print(f"Found {len(record_files)} files.")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if OUT_PER_RECORD.exists():
        OUT_PER_RECORD.unlink()

    agg = {
        "n_records": 0,
        "n_ok": 0,
        "n_fail": 0,      # structural failure only
        "n_correct": 0,

        # PV: only parse-success subset
        "pv_total": 0,
        "pv_valid": 0,

        # RTA: only y_hat=1 and PV valid subset
        "rta_count": 0,
        "rta_sum": 0.0,

        # RV: only y_hat=0 subset (now real)
        "rv_count": 0,
        "rv_sum": 0.0,

        "format_errors": 0,
    }
    fail_reason_count = defaultdict(int)
    derive_method_count = defaultdict(int)

    # confusion matrix for binary feasibility (includes parse failures)
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    format_error_details = defaultdict(int)

    with OUT_PER_RECORD.open("w", encoding="utf-8") as fout:
        for fpath in record_files:
            recs, err = records_from_output_file(fpath)
            if err is not None:
                agg["n_records"] += 1
                agg["n_fail"] += 1
                fail_reason_count[err] += 1
                continue

            for rec in recs:
                agg["n_records"] += 1
                scored = score_record(rec)
                fout.write(json.dumps(scored, ensure_ascii=False) + "\n")

                if not scored.get("ok", False):
                    agg["n_fail"] += 1
                    fail_reason_count[scored.get("error", "unknown")] += 1
                    continue

                agg["n_ok"] += 1
                derive_method_count[scored.get("derive_method", "unknown")] += 1

                if scored.get("correct_answer", False):
                    agg["n_correct"] += 1

                if scored.get("format_error", False):
                    agg["format_errors"] += 1
                    format_error_details[scored.get("format_error_type", "unknown")] += 1

                # update confusion matrix (including parse failures)
                bb = scored.get("binary_bucket")
                if bb in cm:
                    cm[bb] += 1
                else:
                    pred_yes = (scored.get("vlm_answer") == "yes")
                    gt = bool(scored.get("ground_truth_doable"))
                    if gt and pred_yes:
                        cm["TP"] += 1
                    elif (not gt) and pred_yes:
                        cm["FP"] += 1
                    elif gt and (not pred_yes):
                        cm["FN"] += 1
                    else:
                        cm["TN"] += 1

                # PV statistics: ONLY when parse succeeded
                if scored.get("format_error", False) is False:
                    agg["pv_total"] += 1
                    if scored.get("pv_valid", False):
                        agg["pv_valid"] += 1

                # RTA statistics: ONLY when applicable
                if scored.get("traversability_score") is not None:
                    agg["rta_count"] += 1
                    agg["rta_sum"] += float(scored["traversability_score"])

                # RV statistics: ONLY when applicable and computed
                if scored.get("rv_score") is not None:
                    agg["rv_count"] += 1
                    agg["rv_sum"] += float(scored["rv_score"])

    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    bin_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    pv_rate = agg["pv_valid"] / agg["pv_total"] if agg["pv_total"] else 0.0
    rta_mean = agg["rta_sum"] / agg["rta_count"] if agg["rta_count"] else 0.0
    rv_mean = agg["rv_sum"] / agg["rv_count"] if agg["rv_count"] else 0.0

    # Composite CapNav score: average of F1, PV, RTA, RV
    capnav_score = (f1 + pv_rate + rta_mean + rv_mean) / 4.0

    summary = {
        **agg,
        "accuracy": agg["n_correct"] / agg["n_ok"] if agg["n_ok"] else 0.0,

        "pv_validity_rate": pv_rate,
        "avg_traversability_score": rta_mean,
        "avg_rv_score": rv_mean,

        "capnav_score": capnav_score,
        "capnav_components": {
            "f1": f1,
            "pv": pv_rate,
            "rta": rta_mean,
            "rv": rv_mean,
        },

        "coverage": {
            "pv_applicable_frac_over_ok": agg["pv_total"] / agg["n_ok"] if agg["n_ok"] else 0.0,
            "rta_applicable_frac_over_ok": agg["rta_count"] / agg["n_ok"] if agg["n_ok"] else 0.0,
            "rv_applicable_frac_over_ok": agg["rv_count"] / agg["n_ok"] if agg["n_ok"] else 0.0,
        },

        "fail_reason_count": dict(fail_reason_count),
        "derive_method_count": dict(derive_method_count),
        "format_error_details": dict(format_error_details),

        "binary_classification": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": bin_acc,
        },

        "result_root": str(RESULT_ROOT),
        "gt_root": str(GT_ROOT),
        "rv_llm_model": RV_LLM_MODEL,
        "n_input_files_scanned": len(record_files),
        "per_record_output": str(OUT_PER_RECORD),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote:\n  {OUT_PER_RECORD}\n  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()