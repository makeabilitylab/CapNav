"""Shared parsing of model answers into the CapNav record contract.

Every adapter routes its model output through :func:`extract_records`, so a
model is scored on what it answered rather than on how leniently its adapter
happens to parse. This matters on a benchmark: `scripts/capnav_score.py` counts
an unparsable answer as a structural failure, i.e. the same as a wrong answer.

The contract the scorer expects for a record file's ``result`` field is a list::

    [{"question": ..., "agent": ..., "result": {"answer": "yes",
                                                "path": ["node_1", "node_2"]}}]

Models do not reliably emit exactly that, so parsing proceeds in three steps:

1. **Unwrap.** Remove ``<think>`` reasoning traces and take the payload out of
   ``<answer>``, ``<json>`` or a Markdown code fence.
2. **Locate.** Slice the first complete JSON value with a balanced scan that
   respects strings and escapes. Prose on either side of the JSON is ignored,
   and an opening bracket that is never closed is skipped rather than fatal —
   a truncated ``<json>[ { ... } </json>`` still yields its inner object.
3. **Normalise.** Promote a bare answer object to the record contract, filling
   in the question and agent recovered from the prompt.

Anything that survives all three steps is scorable; anything that does not comes
back with an explicit error so the caller can store the text and re-parse it
offline instead of re-running the model.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "strip_reasoning",
    "strip_code_fences",
    "strip_lead_ins",
    "clean_output",
    "first_balanced",
    "extract_json",
    "parse_prompt_metadata",
    "to_record_list",
    "extract_records",
]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_JSON_TAG_BLOCK = re.compile(r"<json>(.*?)</json>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)

_LEAD_INS = (
    "final answer:", "answer:", "final json:", "json:", "output:",
    "here is the json:", "here is my answer:",
)

# The prompt template in scripts/generate_prompts.py.
_QUESTION_RE = re.compile(r"\*\*Question:\*\*\s*(.*?)\s*\*\*Agent profile:\*\*", re.DOTALL)
_AGENT_RE = re.compile(r"Agent name:\s*(\S+)")


# ---------------------------------------------------------------- text cleanup
def strip_reasoning(text: str) -> str:
    """Drop chain-of-thought wrappers, keeping the answer span.

    Handles ``<think>``/``<answer>`` (GLM, Video-R1, InternVL, MiMo) and the
    ``<json>`` wrapper Spatial-MLLM emits.
    """
    out = _THINK_BLOCK.sub("", text or "")
    # An unterminated <think> means the model ran out of budget mid-thought.
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[1]
    elif "<think>" in out:
        out = out.split("<think>", 1)[0]

    for pattern in (_ANSWER_BLOCK, _JSON_TAG_BLOCK):
        match = pattern.search(out)
        if match:
            out = match.group(1)
            break
    return out.strip()


def strip_code_fences(text: str) -> str:
    """Return the first fenced block's contents, else the text unchanged."""
    match = _FENCE.search(text or "")
    if match:
        return match.group(2).strip()

    stripped = (text or "").strip()
    if stripped.startswith("```"):  # unterminated fence
        body = stripped[3:]
        body = re.sub(r"^[A-Za-z0-9_+-]*\r?\n", "", body, count=1)
        return body.rstrip("`").strip()
    return stripped


def strip_lead_ins(text: str) -> str:
    out = (text or "").strip()
    changed = True
    while changed:
        changed = False
        for lead in _LEAD_INS:
            if out[: len(lead)].lower() == lead:
                out = out[len(lead):].strip()
                changed = True
    return out


def clean_output(text: str) -> str:
    """Full normalisation pipeline, no parsing."""
    return strip_lead_ins(strip_code_fences(strip_reasoning(text)))


def first_balanced(text: str) -> Optional[str]:
    """Slice the first complete JSON array/object, respecting strings/escapes."""
    openers = {"[": "]", "{": "}"}
    for i, ch in enumerate(text or ""):
        if ch not in openers:
            continue
        closing = openers[ch]
        depth = 0
        in_string = False
        escaped = False
        for j in range(i, len(text)):
            c = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == ch:
                depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
        # Opened but never closed: keep looking from the next candidate.
    return None


# ------------------------------------------------------------------- json load
def extract_json(text: str) -> Tuple[Optional[Any], str, Optional[str]]:
    """Parse a model answer.

    Returns ``(parsed, cleaned_text, error)``. ``parsed`` is ``None`` exactly
    when ``error`` is set.
    """
    cleaned = clean_output(text)
    if not cleaned:
        return None, cleaned, "empty output"

    for candidate in (cleaned, first_balanced(cleaned)):
        if not candidate:
            continue
        try:
            return json.loads(candidate), cleaned, None
        except json.JSONDecodeError:
            continue

    if first_balanced(cleaned) is None:
        return None, cleaned, "no JSON object or array found"
    return None, cleaned, "JSONDecodeError"


# ------------------------------------------------------- record normalisation
def parse_prompt_metadata(prompt_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Recover ``(question, agent)`` from a generated prompt.

    Used to promote a bare answer object into the scorer's record contract.
    """
    question = agent = None
    match = _QUESTION_RE.search(prompt_text or "")
    if match:
        question = match.group(1).strip()
    match = _AGENT_RE.search(prompt_text or "")
    if match:
        agent = match.group(1).strip()
    return question, agent


def _looks_like_answer(obj: Any) -> bool:
    return isinstance(obj, dict) and "answer" in obj


def to_record_list(
    parsed: Any,
    question: Optional[str] = None,
    agent: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Coerce parsed JSON into the list-of-records contract, or return None.

    Accepted shapes:

    * ``[{"question", "agent", "result": {...}}, ...]`` - already correct.
    * ``[{"answer", "path"}, ...]`` - answers without the record envelope.
    * ``{"question", "agent", "result": {...}}`` - a single record.
    * ``{"answer", "path"}`` - a bare answer with no envelope.
    """
    def envelope(answer_obj: Dict[str, Any]) -> Dict[str, Any]:
        return {"question": question, "agent": agent, "result": answer_obj}

    if isinstance(parsed, list):
        if not parsed:
            return None
        records: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                return None
            if isinstance(item.get("result"), dict):
                # Fill in provenance the model omitted, never overwrite it.
                if item.get("question") is None and question is not None:
                    item = {**item, "question": question}
                if item.get("agent") is None and agent is not None:
                    item = {**item, "agent": agent}
                records.append(item)
            elif _looks_like_answer(item):
                records.append(envelope(item))
            else:
                return None
        return records

    if isinstance(parsed, dict):
        if isinstance(parsed.get("result"), dict):
            item = dict(parsed)
            item.setdefault("question", question)
            item.setdefault("agent", agent)
            return [item]
        if _looks_like_answer(parsed):
            return [envelope(parsed)]

    return None


def extract_records(
    raw_text: str,
    prompt_text: str = "",
) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
    """Turn raw model text into scorer-ready records.

    Returns ``(records, cleaned_text, error)``. ``records`` is ``None`` exactly
    when ``error`` is set; callers should then store ``cleaned_text`` so the
    failure can be re-parsed offline instead of costing another GPU-hour.
    """
    parsed, cleaned, error = extract_json(raw_text)
    if error is not None:
        return None, cleaned, error

    question, agent = parse_prompt_metadata(prompt_text)
    records = to_record_list(parsed, question=question, agent=agent)
    if records is None:
        return None, cleaned, "parsed JSON does not match the CapNav record schema"
    return records, cleaned, None
