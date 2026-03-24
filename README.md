# DAComp-DA

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DAComp-DA)

## Description

**DAComp-DA** (Data Agent Competition — Data Analysis) is an environment for evaluating AI agents on open-ended data analysis tasks. Agents analyze SQLite databases and produce markdown reports with visualizations, evaluated by LLM judges on rubric fidelity, readability, analytical depth, and visualization quality.

## Capabilities

- SQL querying and database exploration
- Python-based data analysis (pandas, numpy, scipy, scikit-learn)
- Data visualization (matplotlib, seaborn)
- Report writing (Markdown)

## Compute Requirements

- Sandbox: 2 CPU / 4GB memory per session
- LLM evaluation: OpenAI API access (gpt-5-mini) for rubric/GSB scoring

## License

[MIT License](https://github.com/ByteDance-Seed/DAComp/blob/main/LICENCE)

## Tasks

| Split | Count | Description |
|-------|-------|-------------|
| test | 100 | Open-ended data analysis with SQLite databases |

## Reward Structure

LLM-judged, 0–100 scale:

- **Rubric scoring (60%)**: LLM evaluates the report against task-specific rubrics with dimensions: Completeness, Accuracy, Conclusiveness.
- **GSB readability (10%)**: Pairwise comparison against 5 reference reports on readability.
- **GSB professionalism (10%)**: Pairwise comparison on analytical depth and professionalism.
- **GSB visualization (20%)**: Pairwise comparison on visualization quality (requires images).

GSB raw scores are threshold-mapped (< -3 → -1, [-3,3] → 0, >3 → +1), averaged, and clamped to [0, 1].

## Data

- **Source**: [HuggingFace](https://huggingface.co/DAComp) (dacomp-da)
- 100 SQLite databases (~6GB total), 100 evaluation rubrics with 5 reference reports each

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute bash commands in the sandbox (Python, SQL, file I/O) |
| `submit` | Submit a markdown report with optional image paths for grading |

## Time Horizon

Multi-turn. Tasks typically require 10–30 tool calls (exploration → analysis → visualization → report).

## Environment Difficulty

Even state-of-the-art agents achieve average scores below 40/100.

## Other Environment Requirements

- OpenAI API key for LLM evaluation (rubric/GSB scoring)

## Safety

Tasks involve analysis of synthetic/public business data. No sensitive personal data. Sandboxes are network-isolated.

## Citations

```bibtex
@misc{lei2025dacomp,
      title={DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle},
      author={Fangyu Lei and Jinxiang Meng and Yiming Huang and Junjie Zhao and Yitong Zhang and Jianwen Luo and Xin Zou and Ruiyi Yang and Wenbo Shi and Yan Gao and Shizhu He and Zuo Wang and Qian Liu and Yang Wang and Ke Wang and Jun Zhao and Kang Liu},
      year={2025},
      eprint={2512.04324},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.04324},
}
```
