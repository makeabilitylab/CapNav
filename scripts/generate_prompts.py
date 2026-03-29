import os
from typing import Dict, List, Any, Optional

from datasets import load_dataset

# ============================================================
# Path configuration (ALL RELATIVE PATHS FOR OPEN-SOURCE USE)
# ============================================================

DATASET_PATH = "dataset/capnav_v0_with_answer.parquet"
AGENT_PROFILES_PATH = "dataset/agent_profiles.parquet"

# Output directory for generated prompt files
OUTPUT_DIR = "generated_prompts"

# Default behavior: export ALL prompts
EXPORT_ALL = True

# Optional overrides (only used when EXPORT_ALL = False)
ROW_INDEX = 0
START_INDEX = 0
N_EXPORT: Optional[int] = None
TARGET_QID: Optional[str] = None

# Whether to group outputs by scene_id subfolders
GROUP_BY_SCENE = True

# ============================================================
# Prompt template (kept aligned with the previous version)
# ============================================================

PROMPT_TEMPLATE = """You are an expert visual reasoning agent for indoor navigation tasks. 
You will receive:
1. A video showing the indoor environment.
2. A single navigation question asking whether an agent can move from one area (node) to another.
3. The agent's physical and capability profile.
4. The list of all nodes in this environment, with their textual descriptions (e.g., room type, furniture, width of passages).

Your goal is to determine whether the agent can successfully reach the destination area from the starting area,
based on both the video and the textual descriptions. You must reason step by step about spatial constraints,
obstacles, connectivity, and the agent’s mobility limitations.

---

### Inputs
**Question:**  
{question_text}

**Agent profile:**  
Agent name: {agent_name}  
Body shape: {body_shape}  
Height (m): {body_height_m}  
Width (m): {body_width_m}  
Depth (m): {body_depth_m}  
Max vertical cross height (m): {max_vertical_cross_height_m}  
Can go up or down stairs: {can_go_up_or_down_stairs}  
Can operate elevator: {can_operate_elevator}  
Can open the door: {can_open_the_door}  
Description: {description}

**Scene graph nodes:**  
{node_list_text}

**Video:** (You can observe the video for spatial layout and obstacles.)

---

### Task
Your goal is to determine whether the agent can **navigate** from the start area to the goal area.  
Focus exclusively on **movement feasibility**, considering physical dimensions, obstacle heights, and connection constraints.
You must **not stop after a single failed route attempt**. If one possible route is blocked (e.g., by stairs or narrow spaces),
you must **actively consider all other possible paths** between the start and goal nodes in the scene graph.

Follow these principles:
1. Explore **all possible routes** through the scene graph before deciding the task is impossible. That means, try **multiple alternative routes** using all visible connections in the scene graph until you are confident that **no feasible route** exists.
2. Account for the agent’s capabilities (e.g., door opening, elevator operation, stair traversal) when evaluating possible paths.
3. When multiple feasible paths exist, select the **most direct and realistic one** given the agent's capabilities.
4. If no route works, specify which **edge or physical barrier** prevents traversal and explain why.
4. If a feasible route exists, specify the **sequence of nodes** representing the navigable path.

Important: If you initially find the route impossible, **re-examine the scene graph** and attempt at least two distinct alternative paths before concluding "no".
Your reasoning should reflect persistent exploration: do not assume failure after one obstacle; explore until all logical alternatives are ruled out.
You do not need to consider unrelated interactions (e.g., turning on lights, using computers, or touching furniture).

---

Please decide for each given question whether the agent can complete the navigation task.  
If yes, provide a **feasible path** through the relevant nodes.  
If no, specify the **edge (two connected nodes)** that blocks traversal and give a concise **reason** (e.g., too narrow passage, stairs, or closed door).  
Your reasoning should always consider the agent’s physical capabilities mentioned in the Agent profile(e.g., wheelchair cannot climb stairs, sweeper robot cannot open doors).

Return your answer in the required structured JSON format below.


---

### Output format (JSON only)

Example output for a list of questions:
[
{example_outputs}
]

Return only the JSON array, no explanation, commentary, or markdown formatting.
"""

# === Example entries showing both YES and NO cases ===
EXAMPLE_ENTRIES = """    {{
        "question": "{example_question_1}",
        "agent": "{agent_name}",
        "result": {{
            "answer": "yes",
            "path": ["node_12", "node_14", "node_15"]
        }}
    }},
    {{
        "question": "{example_question_2}",
        "agent": "{agent_name}",
        "result": {{
            "answer": "no",
            "path": ["node_12", "node_14", "node_15"],
            "reason": "Too narrow passage between the sofa and wall"
        }}
    }}"""

# ============================================================
# Helper functions
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_agent_profiles_parquet(path: str) -> Dict[str, Dict[str, Any]]:
    agent_ds = load_dataset("parquet", data_files=path)["train"]
    return {r["agent_name"]: dict(r) for r in agent_ds if r.get("agent_name")}

def format_scene_nodes(scene_nodes: List[Dict[str, str]]) -> str:
    return "\n".join(f'{n["node_id"]} — {n["name"]}' for n in scene_nodes)

def pick_example_question_same_scene(dataset, scene_id: str, fallback_question: str) -> str:
    subset = dataset.filter(lambda x: x["scene_id"] == scene_id)
    if len(subset) == 0:
        return fallback_question
    return subset[0]["question"]

def build_prompt_from_row(row: Dict[str, Any], dataset, agent_profiles: Dict[str, Dict[str, Any]]) -> str:
    agent_name = row["agent_name"]
    agent = agent_profiles.get(agent_name)
    if agent is None:
        raise KeyError(f"Agent '{agent_name}' not found in agent_profiles.parquet")

    node_list_text = format_scene_nodes(row["scene_nodes"])

    example_q1 = pick_example_question_same_scene(dataset, row["scene_id"], fallback_question=row["question"])
    example_q2 = row["question"]

    example_outputs = EXAMPLE_ENTRIES.format(
        example_question_1=example_q1,
        example_question_2=example_q2,
        agent_name=agent_name
    )

    return PROMPT_TEMPLATE.format(
        question_text=row["question"],
        agent_name=agent.get("agent_name"),
        body_shape=agent.get("body_shape"),
        body_height_m=agent.get("body_height_m"),
        body_width_m=agent.get("body_width_m"),
        body_depth_m=agent.get("body_depth_m"),
        max_vertical_cross_height_m=agent.get("max_vertical_cross_height_m"),
        can_go_up_or_down_stairs=agent.get("can_go_up_or_down_stairs"),
        can_operate_elevator=agent.get("can_operate_elevator"),
        can_open_the_door=agent.get("can_open_the_door"),
        description=agent.get("description"),
        node_list_text=node_list_text,
        example_outputs=example_outputs
    )

def safe_filename(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)

def prompt_output_path(base_dir: str, row: Dict[str, Any]) -> str:
    qid = safe_filename(row["question_id"])
    agent = safe_filename(row["agent_name"])
    scene = safe_filename(row["scene_id"])

    filename = f"{qid}__{agent}.txt"

    out_dir = os.path.join(base_dir, scene) if GROUP_BY_SCENE else base_dir
    ensure_dir(out_dir)
    return os.path.join(out_dir, filename)

def iter_rows(dataset):
    if EXPORT_ALL:
        for i in range(len(dataset)):
            yield dataset[i]
        return

    if TARGET_QID is not None:
        subset = dataset.filter(lambda x: x["question_id"] == TARGET_QID)
        for i in range(len(subset)):
            yield subset[i]
        return

    if N_EXPORT is not None:
        end = min(len(dataset), START_INDEX + N_EXPORT)
        for i in range(START_INDEX, end):
            yield dataset[i]
        return

    yield dataset[ROW_INDEX]

# ============================================================
# Main entry
# ============================================================

def main():
    dataset = load_dataset("parquet", data_files=DATASET_PATH)["train"]
    agent_profiles = load_agent_profiles_parquet(AGENT_PROFILES_PATH)

    ensure_dir(OUTPUT_DIR)

    written = 0
    for row in iter_rows(dataset):
        prompt = build_prompt_from_row(row, dataset, agent_profiles)
        out_path = prompt_output_path(OUTPUT_DIR, row)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        written += 1

    print(f"[DONE] wrote_prompts={written} to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
