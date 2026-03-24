"""DAComp DA evaluation — rubric + GSB scoring via LLM judge.

Ported from ByteDance-Seed/DAComp evaluation suite. Evaluates a DA report
across three scoring channels:

    total = 0.60 * rubric_pct + 0.10 * gsb_readability + 0.10 * gsb_professionalism + 0.20 * gsb_visualization

Where:
    - rubric_pct: [0, 100] — LLM-judged rubric score normalized by max possible
    - gsb_readability: [0, 100] — pairwise comparison against 5 references (text)
    - gsb_professionalism: [0, 100] — pairwise comparison against 5 references (text)
    - gsb_visualization: [0, 100] — pairwise comparison against 5 references (visual)
"""

import asyncio
import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EVAL_MODEL = "gpt-5-mini"

# ---------------------------------------------------------------------------
# Prompts (from DAComp evaluation suite, English versions)
# ---------------------------------------------------------------------------

RUBRIC_PROMPT = """
# Task Instruction
You are a senior data-analytics evaluator. Given a user question, the assistant's analysis history, and a rubric, evaluate how well the assistant addressed each requirement.

The rubric provides:
- A total possible score.
- Multiple requirements, each containing one or more scoring criteria.
  - Deterministic criteria have fixed expectations and can be scored directly.
  - Path-dependent criteria describe several alternative solution paths. First identify the best-matching path according to the assistant's response, then score the sub-criteria under that path. If no path fits, reason independently based on your own domain knowledge; such fallback scores cannot exceed the highest path score.

Criterion types fall into completeness, accuracy, or conclusion quality.

Scoring logic:
- Final score = sum of requirement scores.
- Requirement score = sum of its criteria scores.
- Criterion score = direct score, best-path score, or fallback score.
- Best-path score = sum of the matched path's sub-criteria scores.

Review every requirement sequentially; missing coverage should reduce the score accordingly. If the prompt mentions "[Answer image attachments...]", treat the referenced images as part of the assistant's evidence and evaluate them alongside the text.

【User Question】
{user_query}
【End User Question】

【Assistant Analysis History】
{assistant_response}
【End Assistant Analysis】

【Rubric】
{rubric}
【End Rubric】

Follow the rubric meticulously and provide a detailed breakdown. Return strictly in the JSON format below:
```json
{{
    "Requirement 1": {{
        "Criterion 1.1": {{
            "analysis": "Explain how the assistant handled this criterion and assign a score.",
            "criterion_type": "",
            "score": int
        }},
        "Criterion 1.2": {{
            "analysis": "Explain why a specific path is the best match and analyze its sub-criteria.",
            "best_path_analysis": {{
                "Criterion 1.2.x.1": {{
                    "analysis": "Explain performance on sub-criterion 1.2.x.1 and score it.",
                    "criterion_type": "",
                    "score": int
                }},
                "Criterion 1.2.x.2": {{
                    "analysis": "",
                    "criterion_type": "",
                    "score": int
                }}
            }},
            "score": int
        }},
        "total_score": int
    }},
    "Requirement 2": {{
        "Criterion 2.1": {{
            "analysis": "If no path matches, explain why and reason independently (e.g., refer to Criterion 2.1.notfound.1).",
            "best_path_analysis": {{
                "Criterion 2.1.notfound.1": {{
                    "analysis": "",
                    "criterion_type": "",
                    "score": int
                }},
                "Criterion 2.1.notfound.2": {{
                    "analysis": "",
                    "criterion_type": "",
                    "score": int
                }}
            }},
            "score": int
        }},
        "total_score": int
    }},
    "total_score": int
}}
```
"""

GSB_TEXT_PROMPT = """
You are an expert reviewer of data-analysis reports. Compare the following candidate report with the baseline report.
Evaluate two aspects in detail:
1. Readability and clarity.
2. Analytical professionalism and depth.

For each aspect, assign a score in the range -10 to 10.
Guidelines:
- All analysis is comparative: judge the candidate relative to the baseline.
- -10: the candidate is dramatically worse on this aspect.
- 0: the candidate matches the baseline on this aspect.
- +10: the candidate is dramatically better on this aspect.
- The total score equals the sum of sub-dimension scores (sub-dimension ranges add up to -10~10).
- If both reports are uniformly weak or uniformly strong on a given aspect, explain this and keep the score near 0 instead of using extreme positive or negative values.
- Do not double-count the same issue in multiple sub-dimensions; reflect each issue mainly in the most relevant sub-dimension.

Readability (text-only) can be judged via:
- Conveying complex information concisely (e.g., clean and elegant format, bold/italic highlights) — score range -4~4.
- Logical flow and paragraph organization (headings, lists, progressive storytelling) — score range -3~3.
- Emphasizing key takeaways or action items in the narrative — score range -2~2.
- Concise wording without redundancy; do not reward verbosity. Longer text that does not improve clarity or understanding should not increase the score — score range -1~1.

Analytical professionalism can be judged via:
- Multi-angle analysis that considers different segments/scenarios — score range -4~4.
- Professional reasoning with clear conclusions and sufficient evidence — score range -3~3.
- Practical and grounded insights that are consistent with the provided data and business context, and that support decisions; avoid fabricated numbers or unjustified claims — score range -2~2.
- Anticipation of potential impact, with a clear explanation of likely direction and rationale rather than vague statements — score range -1~1.

Output JSON format:
```json
{{
    "Readability": {{
        "analysis": "Discuss each sub-dimension, compare both reports, and justify the score.",
        "summary": "Overall readability assessment.",
        "score": int
    }},
    "Analytical Depth": {{
        "analysis": "Discuss sub-dimensions, compare both reports, and justify the score.",
        "summary": "Overall analytical depth assessment.",
        "score": int
    }}
}}
```
【Candidate Report】
{content1}
【End Candidate Report】

【Baseline Report】
{content2}
【End Baseline Report】
"""

GSB_VIS_PROMPT = """
You are a senior data visualization and analytics report reviewer. You must compare the candidate report against the baseline **only on the "Insight Presentation & Visualization" dimension**. Ignore all other aspects such as reasoning correctness, business impact, or modeling choices.

Evaluate whether the visualizations are professional and visually appealing, using the following three sub-dimensions (final score = sum of them, range -10 ~ +10):
1. Visual clarity & professionalism (-4 ~ +4): Judge whether the visuals look professional, clean, and aesthetically pleasing while remaining easy to read. Consider whether titles, axes, legends, and labels are complete and clear; font sizes and line weights are appropriate; color choices and contrast are harmonious and not overly flashy; layout, alignment, and spacing are well organized; and unnecessary decorations or chartjunk are avoided.
2. Chart appropriateness & accuracy (-3 ~ +3): Check whether chart types and encodings match the data and analytical task (trend, comparison, composition, distribution, correlation, etc.); axis scales, sorting, and grouping are sensible; there is no misleading design (e.g., truncated axes, exaggerated 3D effects, or arbitrary dual axes); and plotted values and proportions are accurate and consistent with the written conclusions.
3. Text–visual synergy & insightfulness (-3 ~ +3): Assess whether the narrative explicitly references and explains the charts (e.g., "as shown by the blue bar in Figure 2"), highlights key trends, outliers, and segment differences rather than merely restating numbers, and whether the visuals help readers quickly grasp and trust the main insights instead of being decorative or disconnected from the text.

All scores are relative (candidate vs baseline). Negative = clearly worse, zero = comparable, positive = better. When either report contains images, you must use both the visuals and the surrounding text to form your judgment. If both reports have very weak visualization (almost no charts or only very rough visuals), state this in your analysis and keep scores near 0 instead of assigning extreme values.

Return JSON with a single top-level dimension:
```json
{{
    "Insight Presentation & Visualization": {{
        "analysis": "Discuss each sub-dimension, compare both reports, explain the score.",
        "score": int
    }}
}}
```
【Candidate Report】
{content1}
【End Candidate Report】

【Baseline Report】
{content2}
【End Baseline Report】
"""

# ---------------------------------------------------------------------------
# Score extraction helpers (ported from DAComp rubric_scoring.py / tasks.py)
# ---------------------------------------------------------------------------

TARGET_KEYS = {"总得分", "总分", "score", "total_score"}


def _strip_json_block(text: str) -> str:
    """Strip markdown code fence from JSON block."""
    trimmed = text.strip()
    if not trimmed:
        return ""
    if trimmed.startswith("```"):
        parts = trimmed.split("\n", 1)
        body = parts[1] if len(parts) > 1 else ""
        end_split = body.rsplit("```", 1)
        body = end_split[0] if len(end_split) > 1 else body
        return body.strip()
    match = re.search(r"```json\s*(\{.*?\})\s*```", trimmed, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return trimmed


def _parse_json_response(raw: Optional[str]) -> Optional[Any]:
    """Parse JSON from an LLM response, handling common formatting issues."""
    if not raw:
        return None
    text = _strip_json_block(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # Clean up common JSON issues: +N scores, unescaped backslashes
        cleaned = re.sub(r":\s*\+(\d+(?:\.\d+)?)", r": \1", text)
        cleaned = re.sub(r"\\(?![\"\\/bfnrtu0-9])", r"\\\\", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def _to_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _collect_scores(obj: Any, acc: list[Optional[float]]) -> None:
    """Recursively collect all total_score/得分 values from JSON."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in TARGET_KEYS:
                acc.append(_to_float(value))
            _collect_scores(value, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_scores(item, acc)


def extract_total_score(result: Optional[str]) -> Optional[float]:
    """Extract the total score from a rubric evaluation response.

    Takes the LAST occurrence of a total_score key (the outermost/final one).
    """
    if not result:
        return None
    data = _parse_json_response(result)
    if data is None:
        # Fallback: regex scan for score keys
        pattern = re.compile(
            r"[\"']?(?:总得分|总分|score|total_score)[\"']?\s*[:=：]\s*(-?\d+(?:\.\d+)?)"
        )
        matches = pattern.findall(result)
        if matches:
            for match in reversed(matches):
                try:
                    return float(match)
                except ValueError:
                    continue
        return None

    scores: list[Optional[float]] = []
    _collect_scores(data, scores)
    for score in reversed(scores):
        if score is not None:
            return score
    return None


# GSB score extraction keys
GSB_READABILITY_KEYS = (
    "Readability", "Readability Score", "Structure & Readability",
    "可读性", "可读性评分", "报告结构与可读性",
)
GSB_PROFESSIONALISM_KEYS = (
    "Analytical Depth", "Analytical Professionalism", "Analysis Professionalism",
    "Analysis Depth", "分析专业深度", "分析专业有深度", "分析专业性", "分析专业",
)
GSB_VISUALIZATION_KEYS = (
    "Insight Presentation & Visualization", "Insight Presentation",
    "Visualization", "Visualizations",
    "洞察呈现与可视化", "洞察呈现", "可视化表现",
)
GSB_SCORE_FIELDS = ("score", "得分")


def extract_gsb_scores(result: str) -> dict[str, Optional[float]]:
    """Extract readability, professionalism, and visualization scores from GSB response."""
    data = _parse_json_response(result)

    def lookup_score(keys: tuple[str, ...]) -> Optional[float]:
        if data and isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, dict):
                    for field_key in GSB_SCORE_FIELDS:
                        if field_key in value:
                            score = _to_float(value.get(field_key))
                            if score is not None:
                                return score
                else:
                    score = _to_float(value)
                    if score is not None:
                        return score
        # Fallback: regex scan
        if result:
            for key in keys:
                pattern = re.compile(
                    rf"{re.escape(key)}[^\d-]*(-?\d+(?:\.\d+)?)", flags=re.IGNORECASE
                )
                matches = pattern.findall(result)
                if matches:
                    try:
                        return float(matches[-1])
                    except ValueError:
                        continue
        return None

    return {
        "readability": lookup_score(GSB_READABILITY_KEYS),
        "professionalism": lookup_score(GSB_PROFESSIONALISM_KEYS),
        "visualization": lookup_score(GSB_VISUALIZATION_KEYS),
    }


def trans_gsb_score(score_list: list[Optional[float]]) -> float:
    """Transform raw GSB scores via threshold mapping and average.

    Raw scores [-10, 10] are mapped to {-1, 0, +1}, then averaged and clamped to [0, 1].
    """
    POS_THRESHOLD = 3.0

    def score_map(raw: float) -> float:
        if raw < -POS_THRESHOLD:
            return -1.0
        if raw <= POS_THRESHOLD:
            return 0.0
        return 1.0

    mapped = []
    for value in score_list:
        if value is None:
            continue
        mapped.append(score_map(float(value)))

    if not mapped:
        return 0.0
    avg = sum(mapped) / len(mapped)
    return max(0.0, avg)


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def encode_image(image_path: Path) -> Optional[dict]:
    """Encode an image file as a base64 data URL for OpenAI vision API."""
    try:
        data = image_path.read_bytes()
    except Exception:
        return None
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def encode_images_b64(image_data_list: list[bytes]) -> list[dict]:
    """Encode raw image bytes as base64 data URLs."""
    segments = []
    for data in image_data_list:
        encoded = base64.b64encode(data).decode("ascii")
        segments.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        })
    return segments


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

async def _call_openai_json(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str = EVAL_MODEL,
    max_retries: int = 3,
) -> Optional[str]:
    """Call OpenAI API and return raw content string with retries."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
    return None


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

async def _eval_rubric(
    client: AsyncOpenAI,
    instruction: str,
    report: str,
    rubric: str,
) -> Optional[str]:
    """Run rubric evaluation (1 LLM call)."""
    prompt_text = RUBRIC_PROMPT.format(
        user_query=instruction,
        assistant_response=report,
        rubric=rubric,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return await _call_openai_json(client, messages)


async def _eval_gsb_text(
    client: AsyncOpenAI,
    report: str,
    reference_report: str,
) -> Optional[str]:
    """Run GSB text evaluation (readability + professionalism) against one reference."""
    prompt_text = GSB_TEXT_PROMPT.format(
        content1=report,
        content2=reference_report,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return await _call_openai_json(client, messages)


async def _eval_gsb_vis(
    client: AsyncOpenAI,
    report: str,
    agent_images: list[bytes],
    reference_report: str,
    reference_images: list[Path],
) -> Optional[str]:
    """Run GSB visual evaluation against one reference (requires vision model)."""
    prompt_text = GSB_VIS_PROMPT.format(
        content1=report,
        content2=reference_report,
    )

    content_segments: list[dict] = [{"type": "text", "text": prompt_text}]

    # Add agent images
    if agent_images:
        content_segments.append({"type": "text", "text": "\n[Candidate report image attachments]"})
        content_segments.extend(encode_images_b64(agent_images))

    # Add reference images
    ref_image_segments = []
    for img_path in reference_images:
        seg = encode_image(img_path)
        if seg:
            ref_image_segments.append(seg)
    if ref_image_segments:
        content_segments.append({"type": "text", "text": "\n[Baseline report image attachments]"})
        content_segments.extend(ref_image_segments)

    messages = [{"role": "user", "content": content_segments}]
    return await _call_openai_json(client, messages)


def _load_reference_report(ref_dir: Path) -> tuple[str, list[Path]]:
    """Load a GSB reference report (markdown text + image paths)."""
    md_files = list(ref_dir.glob("*.md"))
    report_text = md_files[0].read_text(encoding="utf-8") if md_files else ""
    image_paths = sorted(ref_dir.glob("*.png"))
    return report_text, image_paths


async def evaluate_da_report(
    client: AsyncOpenAI,
    instance_id: str,
    instruction: str,
    report: str,
    agent_images: list[bytes],
    rubric: str,
    metadata: dict[str, int],
    eval_dir: Path,
) -> dict[str, Any]:
    """Run the full DA evaluation pipeline (11 LLM calls in parallel).

    Args:
        client: AsyncOpenAI client for LLM calls.
        instance_id: Task instance ID (e.g., "dacomp-001").
        instruction: The original task instruction/query.
        report: Agent's markdown report text.
        agent_images: Agent's visualization images as raw bytes.
        rubric: Task rubric text.
        metadata: Scoring metadata (Total, Completeness, Accuracy, Conclusiveness).
        eval_dir: Path to eval_data/{instance_id}/ with GSB reference dirs.

    Returns:
        Dict with total score [0, 100] and detailed breakdown.
    """
    # Load GSB reference reports
    references: list[tuple[str, str, list[Path]]] = []
    for i in range(5):
        ref_dir = eval_dir / f"gsb_ref_{i}"
        if ref_dir.exists():
            ref_text, ref_images = _load_reference_report(ref_dir)
            references.append((f"gsb_ref_{i}", ref_text, ref_images))

    # Launch all LLM calls in parallel
    tasks = []

    # 1. Rubric scoring (1 call)
    tasks.append(asyncio.create_task(_eval_rubric(client, instruction, report, rubric)))

    # 2. GSB text scoring (5 calls)
    for _name, ref_text, _ref_images in references:
        tasks.append(asyncio.create_task(_eval_gsb_text(client, report, ref_text)))

    # 3. GSB visual scoring (5 calls)
    for _name, ref_text, ref_images in references:
        tasks.append(asyncio.create_task(
            _eval_gsb_vis(client, report, agent_images, ref_text, ref_images)
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Parse results
    rubric_result = results[0] if not isinstance(results[0], Exception) else None
    n_refs = len(references)
    gsb_text_results = results[1:1 + n_refs]
    gsb_vis_results = results[1 + n_refs:1 + 2 * n_refs]

    # --- Rubric score ---
    rubric_total_score = extract_total_score(rubric_result) if rubric_result else None
    max_score = metadata.get("Total", 1)
    if rubric_total_score is not None and max_score > 0:
        rubric_pct = min(100.0, max(0.0, (rubric_total_score / max_score) * 100.0))
    else:
        rubric_pct = 0.0

    # --- GSB text scores ---
    readability_scores: list[Optional[float]] = []
    professionalism_scores: list[Optional[float]] = []
    for r in gsb_text_results:
        if isinstance(r, Exception) or r is None:
            readability_scores.append(None)
            professionalism_scores.append(None)
            continue
        scores = extract_gsb_scores(r)
        readability_scores.append(scores.get("readability"))
        professionalism_scores.append(scores.get("professionalism"))

    gsb_readability = trans_gsb_score(readability_scores) * 100.0
    gsb_professionalism = trans_gsb_score(professionalism_scores) * 100.0

    # --- GSB visual scores ---
    visualization_scores: list[Optional[float]] = []
    if agent_images:
        for r in gsb_vis_results:
            if isinstance(r, Exception) or r is None:
                visualization_scores.append(None)
                continue
            scores = extract_gsb_scores(r)
            visualization_scores.append(scores.get("visualization"))
        gsb_visualization = trans_gsb_score(visualization_scores) * 100.0
    else:
        # No agent images → visualization score is 0
        gsb_visualization = 0.0

    # --- Weighted total ---
    total = (
        0.60 * rubric_pct
        + 0.10 * gsb_readability
        + 0.10 * gsb_professionalism
        + 0.20 * gsb_visualization
    )

    return {
        "total": total,
        "rubric_pct": rubric_pct,
        "rubric_raw_score": rubric_total_score,
        "rubric_max_score": max_score,
        "gsb_readability": gsb_readability,
        "gsb_professionalism": gsb_professionalism,
        "gsb_visualization": gsb_visualization,
        "readability_raw_scores": readability_scores,
        "professionalism_raw_scores": professionalism_scores,
        "visualization_raw_scores": visualization_scores,
        "n_references": n_refs,
    }
