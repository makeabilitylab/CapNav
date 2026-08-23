#!/usr/bin/env python3
"""Tests for the shared answer parser.

Each case is an output shape one of the evaluated models actually produces. The
point is that every adapter is scored on the same terms: whatever any model
emits, the parser either returns the record contract `capnav_score.py` expects
or reports an explicit error.

    python tests/test_output_parsing.py        # or: python -m pytest tests/ -q

Needs no GPU, no model weights and no dataset.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.output_parsing import clean_output, extract_records  # noqa: E402

PROMPT = """### Inputs
**Question:**
Can [Agent] move from the Lobby to the Kitchen?

**Agent profile:**
Agent name: WHEELCHAIR
Body shape: box
"""

RECORD = (
    '[{"question": "Can [Agent] move from the Lobby to the Kitchen?", '
    '"agent": "WHEELCHAIR", '
    '"result": {"answer": "yes", "path": ["node_1", "node_2"]}}]'
)

PARSABLE = {
    "plain record list": RECORD,
    "fenced json": f"```json\n{RECORD}\n```",
    "fenced, no language": f"```\n{RECORD}\n```",
    "think block": f"<think>The door is 0.8 m wide.</think>\n{RECORD}",
    "think plus answer tags": f"<think>reasoning</think>\n<answer>{RECORD}</answer>",
    "answer tags only": f"<answer>{RECORD}</answer>",
    "lead-in": f"Final Answer:\n{RECORD}",
    "prose before json": f"Here is my analysis.\n\n{RECORD}",
    "prose after json": f"{RECORD}\n\nI hope this helps! [end]",
    "unterminated think": f"<think>still reasoning</think>{RECORD}",
    "several think blocks": f"<think>a</think><think>b</think>{RECORD}",
    "backtick inside a value": (
        '```json\n[{"question": "Can [Agent] pass the `narrow` hall?", '
        '"agent": "WHEELCHAIR", "result": {"answer": "no", "path": []}}]\n```'
    ),
    "brackets inside a string": (
        '[{"agent": "WHEELCHAIR", "result": {"answer": "no", "path": []}, '
        '"note": "see [fig 2] and {x}"}]'
    ),
    # Answers that omit the record envelope, or the whole array.
    "answers without envelope": '[{"answer": "no", "path": [], "reason": "stairs"}]',
    "bare answer object": '{"answer": "yes", "path": ["node_12", "node_14"]}',
    # Spatial-MLLM wraps its answer in <json> and often never closes the array.
    "json tags, unclosed array": (
        '<json>[\n  {\n    "answer": "yes",\n'
        '    "path": ["node_12", "node_14", "node_15"]\n  }\n</json>'
    ),
}

REJECTED = {
    "empty": "",
    "prose only": "I cannot determine this from the video.",
    "unterminated think, no answer": "<think>I am still thinking and never finish",
    "truncated json": '[{"question": "Can [Agent] move", "agent": "WHEE',
}


def run() -> int:
    failures = []

    for name, text in PARSABLE.items():
        records, _cleaned, error = extract_records(text, PROMPT)
        if error is not None:
            failures.append(f"{name}: expected a parse, got error={error!r}")
            continue
        if not isinstance(records, list) or not records:
            failures.append(f"{name}: expected a non-empty list, got {records!r}")
            continue
        first = records[0]
        if not isinstance(first.get("result"), dict):
            failures.append(f"{name}: record has no result object: {first!r}")
            continue
        # Provenance is filled in from the prompt when the model omits it,
        # because the scorer needs the question to resolve start/goal nodes.
        if first.get("question") is None or first.get("agent") is None:
            failures.append(f"{name}: missing question/agent: {first!r}")

    for name, text in REJECTED.items():
        records, _cleaned, error = extract_records(text, PROMPT)
        if error is None:
            failures.append(f"{name}: expected rejection, parsed {records!r}")

    # Normalisation must never raise, whatever the model emitted.
    for text in list(PARSABLE.values()) + list(REJECTED.values()):
        clean_output(text)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for line in failures:
            print("  -", line)
        return 1

    print(f"ok - {len(PARSABLE)} shapes parsed, {len(REJECTED)} correctly rejected")
    return 0


def test_output_parsing():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
