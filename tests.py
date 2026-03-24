"""Unit tests for DAComp-DA environment — scoring logic and data integrity.

Run with: uv run pytest tests.py -v
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# DA evaluation scoring tests
# ---------------------------------------------------------------------------
from evaluate_da import (
    extract_gsb_scores,
    extract_total_score,
    trans_gsb_score,
)


class TestExtractTotalScore:
    """Tests for rubric total score extraction."""

    def test_simple_json(self):
        result = '```json\n{"Requirement 1": {"total_score": 10}, "total_score": 25}\n```'
        assert extract_total_score(result) == 25

    def test_nested_total(self):
        result = json.dumps({
            "Req1": {"Criterion 1.1": {"score": 3}, "total_score": 8},
            "Req2": {"Criterion 2.1": {"score": 5}, "total_score": 12},
            "total_score": 20,
        })
        assert extract_total_score(result) == 20

    def test_chinese_key(self):
        result = json.dumps({"需求1": {"得分": 5}, "总得分": 15})
        assert extract_total_score(result) == 15

    def test_fallback_regex(self):
        result = 'Some text... total_score: 42 more text'
        assert extract_total_score(result) == 42

    def test_none_input(self):
        assert extract_total_score(None) is None

    def test_empty_string(self):
        assert extract_total_score("") is None

    def test_no_score(self):
        result = '{"analysis": "some analysis", "details": "no score here"}'
        assert extract_total_score(result) is None

    def test_last_occurrence(self):
        """Should return the last (outermost) total_score."""
        result = json.dumps({
            "Req1": {"total_score": 10},
            "Req2": {"total_score": 15},
            "total_score": 25,
        })
        assert extract_total_score(result) == 25


class TestExtractGsbScores:
    """Tests for GSB score extraction."""

    def test_english_keys(self):
        result = json.dumps({
            "Readability": {"analysis": "...", "score": 3},
            "Analytical Depth": {"analysis": "...", "score": -2},
        })
        scores = extract_gsb_scores(result)
        assert scores["readability"] == 3
        assert scores["professionalism"] == -2

    def test_visualization_key(self):
        result = json.dumps({
            "Insight Presentation & Visualization": {"analysis": "...", "score": 5},
        })
        scores = extract_gsb_scores(result)
        assert scores["visualization"] == 5

    def test_missing_keys(self):
        result = json.dumps({"something_else": {"score": 10}})
        scores = extract_gsb_scores(result)
        assert scores["readability"] is None

    def test_code_fence(self):
        result = '```json\n{"Readability": {"score": 4}, "Analytical Depth": {"score": -1}}\n```'
        scores = extract_gsb_scores(result)
        assert scores["readability"] == 4


class TestTransGsbScore:
    """Tests for GSB score threshold mapping and aggregation."""

    def test_all_positive(self):
        scores = [5.0, 7.0, 4.0, 6.0, 8.0]
        assert trans_gsb_score(scores) == 1.0

    def test_all_negative(self):
        scores = [-5.0, -7.0, -4.0, -6.0, -8.0]
        assert trans_gsb_score(scores) == 0.0

    def test_all_neutral(self):
        scores = [0.0, 1.0, -1.0, 2.0, -2.0]
        assert trans_gsb_score(scores) == 0.0

    def test_mixed(self):
        scores = [5.0, 4.0, 1.0, 0.0, -5.0]
        result = trans_gsb_score(scores)
        assert abs(result - 0.2) < 1e-6

    def test_empty(self):
        assert trans_gsb_score([]) == 0.0

    def test_all_none(self):
        assert trans_gsb_score([None, None]) == 0.0

    def test_threshold_boundary(self):
        scores = [-3.0, 3.0]
        assert trans_gsb_score(scores) == 0.0

    def test_just_above_threshold(self):
        scores = [3.1]
        assert trans_gsb_score(scores) == 1.0

    def test_just_below_neg_threshold(self):
        scores = [-3.1]
        assert trans_gsb_score(scores) == 0.0


class TestWeightedTotal:
    """Tests for the DA weighted total formula."""

    def test_perfect_scores(self):
        total = 0.60 * 100.0 + 0.10 * 100.0 + 0.10 * 100.0 + 0.20 * 100.0
        assert total == 100.0

    def test_zero_scores(self):
        total = 0.60 * 0 + 0.10 * 0 + 0.10 * 0 + 0.20 * 0
        assert total == 0.0

    def test_rubric_only(self):
        total = 0.60 * 80.0 + 0.10 * 0 + 0.10 * 0 + 0.20 * 0
        assert total == 48.0


# ---------------------------------------------------------------------------
# Data integrity tests (only run if data is downloaded)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
DA_TASKS_PATH = DATA_DIR / "tasks_da.json"


@pytest.fixture
def da_tasks():
    if not DA_TASKS_PATH.exists():
        pytest.skip("DA task data not downloaded (run prepare_data.py first)")
    with open(DA_TASKS_PATH) as f:
        return json.load(f)


class TestDADataIntegrity:
    def test_task_count(self, da_tasks):
        assert len(da_tasks) == 100

    def test_required_fields(self, da_tasks):
        for task in da_tasks:
            assert "instance_id" in task
            assert "instruction" in task
            assert task["instance_id"]  # Not empty
            assert task["instruction"]  # Not empty

    def test_unique_ids(self, da_tasks):
        ids = [t["instance_id"] for t in da_tasks]
        assert len(ids) == len(set(ids))

    def test_stable_ordering(self, da_tasks):
        ids = [t["instance_id"] for t in da_tasks]
        assert ids == sorted(ids)

    def test_no_gold_data_in_specs(self, da_tasks):
        """Ensure no evaluation data leaks into agent-visible task specs."""
        for task in da_tasks:
            assert "rubric" not in task
            assert "gold" not in task
            assert "answer" not in task
