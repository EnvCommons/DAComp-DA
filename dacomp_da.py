"""DAComp DA — OpenReward sandbox environment for open-ended data analysis.

Agents analyze SQLite databases and produce markdown reports with visualizations.
Scoring uses LLM judges for rubric (60%) and GSB pairwise comparison (40%).

Paper: https://arxiv.org/abs/2512.04324
Dataset: https://huggingface.co/DAComp
"""

import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI
from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel

from evaluate_da import evaluate_da_report

logger = logging.getLogger(__name__)

# --- Module-level client cache (avoid creating one per task) ---
_openai_clients: dict[str, AsyncOpenAI] = {}


def _get_openai_client(api_key: str) -> AsyncOpenAI:
    if api_key not in _openai_clients:
        _openai_clients[api_key] = AsyncOpenAI(api_key=api_key)
    return _openai_clients[api_key]


# --- Module-level data loading ---

if os.path.exists("/orwd_data"):
    _DATA_DIR = Path("/orwd_data")
else:
    _DATA_DIR = Path(__file__).parent

_all_records: dict[str, dict] = {}
_test_tasks: list[JSONObject] = []

_json_path = _DATA_DIR / "tasks_da.json"
if _json_path.exists():
    with open(_json_path) as _f:
        _records = json.load(_f)
    for _record in _records:
        _instance_id = _record["instance_id"]
        _all_records[_instance_id] = _record
        # Task spec: agent-visible fields only
        _test_tasks.append({
            "instance_id": _instance_id,
        })
else:
    logger.warning(f"DA data file not found: {_json_path}")

# Load evaluation metadata (rubrics + scoring maximums)
_eval_meta: dict[str, dict] = {}
_eval_data_dir = _DATA_DIR / "eval_data"
if _eval_data_dir.exists():
    for _task_dir in sorted(_eval_data_dir.iterdir()):
        if not _task_dir.is_dir():
            continue
        _rubric_path = _task_dir / "rubric.txt"
        _meta_path = _task_dir / "metadata.json"
        if _rubric_path.exists() and _meta_path.exists():
            _eval_meta[_task_dir.name] = {
                "rubric": _rubric_path.read_text(encoding="utf-8"),
                "metadata": json.loads(_meta_path.read_text(encoding="utf-8")),
            }


# --- Pydantic parameter models ---

class BashParams(BaseModel, extra="forbid"):
    command: str


class SubmitParams(BaseModel, extra="forbid"):
    """Submit a data analysis report for grading."""
    report: str
    image_paths: list[str] = []


# --- Environment class ---

class DACompDA(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        instance_id = str(task_spec["instance_id"])
        if instance_id not in _all_records:
            raise ValueError(f"Unknown DA task: {instance_id}")

        record = _all_records[instance_id]
        self.instance_id = instance_id
        self.instruction: str = record["instruction"]

        # OpenReward API key for sandbox
        or_api_key = (
            secrets.get("OPENREWARD_API_KEY")
            or secrets.get("api_key")
            or os.environ.get("OPENREWARD_API_KEY", "").strip('"')
        )
        if not or_api_key:
            raise ValueError("OpenReward API key required (pass as OPENREWARD_API_KEY)")

        # OpenAI API key for LLM evaluation
        openai_api_key = (
            secrets.get("OPENAI_API_KEY")
            or secrets.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY", "").strip('"')
        )
        if not openai_api_key:
            raise ValueError("OpenAI API key required for evaluation (pass as OPENAI_API_KEY)")

        self.openai_client = _get_openai_client(openai_api_key)

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/DAComp-DA",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="2:4",
            block_network=True,
            bucket_config=SandboxBucketConfig(
                mount_path="/data",
                read_only=True,
                only_dir=f"{self.instance_id}",
            ),
        )

        or_client = AsyncOpenReward(api_key=or_api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        self.submitted = False

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "test":
            return _test_tasks
        return []

    async def get_prompt(self) -> list[TextBlock]:
        prompt = f"""You are a data analyst tasked with producing a comprehensive analysis report.

## Task

{self.instruction}

## Data

A SQLite database is available at `/data/{self.instance_id}.sqlite`. You can explore it using Python:

```python
import sqlite3
conn = sqlite3.connect('/data/{self.instance_id}.sqlite')
# List tables
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(tables)
```

You also have access to pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, and other common data science libraries.

## Environment

You have access to a Linux sandbox with Python 3.12. Use the `bash` tool to run Python scripts, explore the data, perform analysis, and create visualizations.

## Instructions

1. Explore the database to understand the schema and data.
2. Perform thorough analysis addressing the task requirements.
3. Create visualizations (charts/plots) that support your findings. Save them as PNG files.
4. When ready, use the `submit` tool with:
   - **report**: Your complete analysis report in Markdown format.
   - **image_paths**: List of paths to any visualization PNG files you created in the sandbox.

Your report will be evaluated on:
- **Completeness** (covering all required aspects)
- **Accuracy** (correct analysis and conclusions)
- **Conclusiveness** (actionable insights and clear recommendations)
- **Readability** (clear structure and writing)
- **Visualization quality** (professional, accurate, and insightful charts)

You get one submission attempt."""

        return [TextBlock(text=prompt)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command in the sandbox environment."""
        result = await self.sandbox.run(params.command.strip())
        output, code = result

        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit a data analysis report for evaluation.

        Your report will be graded against a detailed rubric covering completeness,
        accuracy, and conclusiveness, plus compared against reference reports for
        readability and visualization quality. This is a terminal action — you get
        one submission attempt.
        """
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed.")],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )

        self.submitted = True

        # Download agent images from sandbox
        agent_images: list[bytes] = []
        for img_path in params.image_paths:
            try:
                img_data = await self.sandbox.download(img_path.strip())
                if img_data:
                    agent_images.append(img_data)
            except Exception as e:
                logger.warning(f"Failed to download image {img_path}: {e}")

        # Get evaluation data for this task
        eval_meta = _eval_meta.get(self.instance_id, {})
        rubric = eval_meta.get("rubric", "")
        metadata = eval_meta.get("metadata", {"Total": 1})
        eval_dir = _eval_data_dir / self.instance_id

        if not rubric:
            logger.warning(f"No rubric found for {self.instance_id}")
            return ToolOutput(
                blocks=[TextBlock(text="Evaluation error: no rubric found for this task.")],
                metadata={"error": "no_rubric"},
                reward=0.0,
                finished=True,
            )

        try:
            eval_result = await evaluate_da_report(
                client=self.openai_client,
                instance_id=self.instance_id,
                instruction=self.instruction,
                report=params.report,
                agent_images=agent_images,
                rubric=rubric,
                metadata=metadata,
                eval_dir=eval_dir,
            )
        except Exception as e:
            logger.exception("DA evaluation error")
            return ToolOutput(
                blocks=[TextBlock(text=f"Evaluation error: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=True,
            )

        total = eval_result["total"]
        reward = total / 100.0

        result_text = f"""Submission Results:
- Total Score: {total:.1f}/100
- Rubric Score: {eval_result['rubric_pct']:.1f}/100 (weight: 60%)
- GSB Readability: {eval_result['gsb_readability']:.1f}/100 (weight: 10%)
- GSB Professionalism: {eval_result['gsb_professionalism']:.1f}/100 (weight: 10%)
- GSB Visualization: {eval_result['gsb_visualization']:.1f}/100 (weight: 20%)"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "total": total,
                "rubric_pct": eval_result["rubric_pct"],
                "rubric_raw_score": eval_result["rubric_raw_score"],
                "gsb_readability": eval_result["gsb_readability"],
                "gsb_professionalism": eval_result["gsb_professionalism"],
                "gsb_visualization": eval_result["gsb_visualization"],
                "n_images": len(agent_images),
            },
            reward=reward,
            finished=True,
        )
