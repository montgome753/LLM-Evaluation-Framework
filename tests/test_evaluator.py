"""
Tests for the core LLM Evaluation engine and metrics.

Uses mock LiteLLM responses — no real API keys required.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval.core.evaluator import EvaluationConfig, EvaluationResult, LLMEvaluator, SampleResult
from llm_eval.metrics.accuracy import AccuracyMetric
from llm_eval.metrics.cost import CostMetric
from llm_eval.metrics.hallucination import HallucinationMetric
from llm_eval.metrics.latency import LatencyMetric, LatencyStats
from llm_eval.benchmarks.custom import CustomBenchmark
from llm_eval.database.models import Database


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_eval.db")


@pytest.fixture
def evaluator(tmp_db):
    return LLMEvaluator(db_path=tmp_db)


@pytest.fixture
def sample_result():
    return SampleResult(
        sample_id=0,
        prompt="What is 2+2?",
        expected="4",
        response="The answer is 4.",
        is_correct=True,
        latency_ms=350.0,
        input_tokens=12,
        output_tokens=8,
        cost_usd=0.000002,
        hallucination_score=0.0,
        reasoning_score=5.0,
        error=None,
    )


@pytest.fixture
def eval_result(sample_result):
    return EvaluationResult(
        run_id="test1234",
        model="gpt-4o-mini",
        benchmark="mmlu",
        num_samples=1,
        accuracy=1.0,
        avg_latency_ms=350.0,
        p50_latency_ms=350.0,
        p95_latency_ms=350.0,
        p99_latency_ms=350.0,
        total_cost_usd=0.000002,
        cost_per_1k_tokens=0.001,
        hallucination_rate=0.0,
        avg_reasoning_score=5.0,
        samples=[sample_result],
        created_at=datetime.utcnow(),
        config=None,
    )


# ── AccuracyMetric Tests ───────────────────────────────────────────────────

class TestAccuracyMetric:
    def setup_method(self):
        self.metric = AccuracyMetric()

    def test_exact_match(self):
        assert self.metric.score("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert self.metric.score("paris", "Paris") is True

    def test_normalized_match(self):
        assert self.metric.score("The answer is Paris.", "Paris") is True

    def test_mc_correct(self):
        assert self.metric.score("The answer is A", "A") is True
        assert self.metric.score("I think B is correct", "B") is True

    def test_mc_incorrect(self):
        assert self.metric.score("A", "B") is False

    def test_fuzzy_match_close(self):
        assert self.metric.score("mitochondria", "mitochondrion") is True

    def test_empty_prediction(self):
        assert self.metric.score("", "Paris") is False

    def test_empty_reference(self):
        assert self.metric.score("Paris", "") is False

    def test_batch_score(self):
        preds = ["A", "B", "A", "D"]
        refs = ["A", "B", "C", "D"]
        results, acc = self.metric.batch_score(preds, refs)
        assert results == [True, True, False, True]
        assert acc == 0.75


# ── HallucinationMetric Tests ──────────────────────────────────────────────

class TestHallucinationMetric:
    def setup_method(self):
        self.metric = HallucinationMetric()

    def test_grounded_response_low_score(self):
        response = (
            "Based on the research data, specifically the 2023 study, "
            "the evidence shows that X is the case. For example, in fact..."
        )
        score = self.metric.score("test prompt", response)
        assert score < 0.5

    def test_empty_response_max_score(self):
        assert self.metric.score("prompt", "") == 1.0

    def test_hallucination_signals_increase_score(self):
        response = "I believe it was supposedly around 5 million, I could be wrong."
        score = self.metric.score("prompt", response)
        assert score > 0.0

    def test_reasoning_quality_empty(self):
        assert self.metric.reasoning_quality("") == 1.0

    def test_reasoning_quality_rich_response(self):
        response = (
            "First, we need to consider X. Based on the evidence, therefore "
            "we can conclude that Y. Since Z is also true, this means that "
            "the final answer follows logically. For example, specifically..."
        )
        score = self.metric.reasoning_quality(response)
        assert score > 4.0

    def test_reasoning_quality_short_response(self):
        score = self.metric.reasoning_quality("Yes.")
        assert score <= 3.0


# ── CostMetric Tests ───────────────────────────────────────────────────────

class TestCostMetric:
    def setup_method(self):
        self.metric = CostMetric()

    def test_known_model_cost(self):
        cost = self.metric.calculate("gpt-4o-mini", 1000, 500)
        assert cost > 0
        assert cost < 0.01  # should be < 1 cent for 1500 tokens

    def test_zero_tokens_zero_cost(self):
        assert self.metric.calculate("gpt-4o", 0, 0) == 0.0

    def test_expensive_model_higher_cost(self):
        cheap = self.metric.calculate("gpt-4o-mini", 1000, 1000)
        expensive = self.metric.calculate("gpt-4", 1000, 1000)
        assert expensive > cheap

    def test_partial_model_name_match(self):
        cost = self.metric.calculate("gpt-4o-mini-2024-07-18", 100, 100)
        assert cost > 0

    def test_unknown_model_uses_default(self):
        cost = self.metric.calculate("unknown-model-xyz", 100, 100)
        assert cost > 0

    def test_run_cost_estimate(self):
        estimate = self.metric.estimate_run_cost("gpt-4o-mini", 100)
        assert estimate > 0
        assert estimate < 1.0  # $1 for 100 samples on mini


# ── LatencyMetric Tests ────────────────────────────────────────────────────

class TestLatencyMetric:
    def setup_method(self):
        self.metric = LatencyMetric()

    def test_empty_returns_zeros(self):
        stats = self.metric.compute_stats([])
        assert stats.count == 0
        assert stats.mean_ms == 0

    def test_percentiles(self):
        latencies = list(range(100, 1100, 100))  # 100–1000
        stats = self.metric.compute_stats(latencies)
        assert stats.p50_ms == latencies[5]
        assert stats.p95_ms >= latencies[8]
        assert stats.min_ms == 100
        assert stats.max_ms == 1000

    def test_sla_violation_rate(self):
        latencies = [100, 200, 6000, 7000, 8000]
        rate = self.metric.sla_violation_rate(latencies, threshold_ms=5000)
        assert rate == 0.6

    def test_classify(self):
        assert self.metric.classify(200) == "excellent"
        assert self.metric.classify(1000) == "good"
        assert self.metric.classify(2000) == "acceptable"
        assert self.metric.classify(4000) == "slow"
        assert self.metric.classify(9000) == "very_slow"


# ── CustomBenchmark Tests ──────────────────────────────────────────────────

class TestCustomBenchmark:
    def test_load_csv_string(self):
        csv_content = "prompt,expected\n\"What is 2+2?\",4\n\"Capital of France?\",Paris"
        bench = CustomBenchmark.from_string(csv_content, format="csv")
        samples = bench.load(10)
        assert len(samples) == 2
        assert samples[0]["prompt"] == "What is 2+2?"
        assert samples[0]["expected"] == "4"

    def test_load_json_string(self):
        data = [{"prompt": "Q1", "expected": "A1"}, {"prompt": "Q2", "expected": "A2"}]
        import json
        bench = CustomBenchmark.from_string(json.dumps(data), format="json")
        samples = bench.load(10)
        assert len(samples) == 2

    def test_missing_prompt_raises(self):
        csv = "question,answer\nQ1,A1"
        with pytest.raises(ValueError, match="prompt"):
            CustomBenchmark.from_string(csv, format="csv")

    def test_num_samples_limit(self):
        data = [{"prompt": f"Q{i}", "expected": f"A{i}"} for i in range(50)]
        import json
        bench = CustomBenchmark.from_string(json.dumps(data), format="json")
        samples = bench.load(10)
        assert len(samples) == 10

    def test_from_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CustomBenchmark.from_file(tmp_path / "nonexistent.csv")

    def test_unsupported_format(self):
        with pytest.raises(ValueError):
            CustomBenchmark.from_string("data", format="xml")

    def test_len(self):
        data = [{"prompt": f"Q{i}"} for i in range(5)]
        import json
        bench = CustomBenchmark.from_string(json.dumps(data), format="json")
        assert len(bench) == 5


# ── Database Tests ─────────────────────────────────────────────────────────

class TestDatabase:
    def test_save_and_retrieve(self, tmp_db, eval_result):
        db = Database(tmp_db)
        db.save_result(eval_result)
        record = db.get_result("test1234")
        assert record is not None
        assert record.model == "gpt-4o-mini"
        assert abs(record.accuracy - 1.0) < 1e-6

    def test_list_results(self, tmp_db, eval_result):
        db = Database(tmp_db)
        db.save_result(eval_result)
        records = db.list_results()
        assert len(records) == 1

    def test_filter_by_model(self, tmp_db, eval_result):
        db = Database(tmp_db)
        db.save_result(eval_result)
        records = db.list_results(model="gpt-4o-mini")
        assert len(records) == 1
        records_miss = db.list_results(model="nonexistent")
        assert len(records_miss) == 0

    def test_delete_result(self, tmp_db, eval_result):
        db = Database(tmp_db)
        db.save_result(eval_result)
        assert db.delete_result("test1234") is True
        assert db.get_result("test1234") is None

    def test_delete_nonexistent(self, tmp_db):
        db = Database(tmp_db)
        assert db.delete_result("no_such_run") is False

    def test_export_csv(self, tmp_db, eval_result, tmp_path):
        db = Database(tmp_db)
        db.save_result(eval_result)
        out = str(tmp_path / "out.csv")
        path = db.export_csv(out)
        content = Path(path).read_text()
        assert "gpt-4o-mini" in content
        assert "accuracy" in content

    def test_export_json(self, tmp_db, eval_result, tmp_path):
        db = Database(tmp_db)
        db.save_result(eval_result)
        out = str(tmp_path / "out.json")
        path = db.export_json(out)
        data = json.loads(Path(path).read_text())
        assert isinstance(data, list)
        assert data[0]["model"] == "gpt-4o-mini"

    def test_export_empty_raises(self, tmp_db, tmp_path):
        db = Database(tmp_db)
        with pytest.raises(ValueError, match="No results"):
            db.export_csv(str(tmp_path / "empty.csv"))


# ── Core Evaluator Integration Tests ──────────────────────────────────────

class TestLLMEvaluator:
    """Integration tests with mocked LiteLLM calls."""

    def _mock_response(self, content: str = "A", input_tokens: int = 10, output_tokens: int = 5):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = input_tokens
        resp.usage.completion_tokens = output_tokens
        return resp

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, tmp_db):
        evaluator = LLMEvaluator(db_path=tmp_db)
        config = EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=3)
        samples = [
            {"prompt": "Q1?\nA) Yes\nB) No\nAnswer:", "expected": "A"},
            {"prompt": "Q2?\nA) Yes\nB) No\nAnswer:", "expected": "B"},
            {"prompt": "Q3?\nA) Yes\nB) No\nAnswer:", "expected": "A"},
        ]

        mock_resp = self._mock_response("A")
        with patch("llm_eval.core.evaluator.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await evaluator.evaluate(config, samples)

        assert result.num_samples == 3
        assert 0.0 <= result.accuracy <= 1.0
        assert result.avg_latency_ms >= 0
        assert result.model == "gpt-4o-mini"
        assert result.run_id == config.run_id

    @pytest.mark.asyncio
    async def test_evaluate_persists_to_db(self, tmp_db):
        evaluator = LLMEvaluator(db_path=tmp_db)
        config = EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=2)
        samples = [{"prompt": "Q?", "expected": "A"}, {"prompt": "Q2?", "expected": "B"}]

        mock_resp = self._mock_response("A")
        with patch("llm_eval.core.evaluator.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await evaluator.evaluate(config, samples)

        db = Database(tmp_db)
        record = db.get_result(result.run_id)
        assert record is not None
        assert record.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_evaluate_handles_timeout(self, tmp_db):
        evaluator = LLMEvaluator(db_path=tmp_db)
        config = EvaluationConfig(
            model="gpt-4o-mini", benchmark="mmlu", num_samples=1, timeout=0.001
        )
        samples = [{"prompt": "Q?", "expected": "A"}]

        async def slow_completion(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("llm_eval.core.evaluator.acompletion", new=AsyncMock(side_effect=slow_completion)):
            result = await evaluator.evaluate(config, samples)

        assert result.samples[0].error == "timeout"
        assert result.accuracy == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_multiple(self, tmp_db):
        evaluator = LLMEvaluator(db_path=tmp_db)
        configs = [
            EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=2),
            EvaluationConfig(model="claude-3-haiku-20240307", benchmark="mmlu", num_samples=2),
        ]
        samples = [{"prompt": "Q?", "expected": "A"}, {"prompt": "Q2?", "expected": "B"}]

        mock_resp = self._mock_response("A")
        with patch("llm_eval.core.evaluator.acompletion", new=AsyncMock(return_value=mock_resp)):
            results = await evaluator.evaluate_multiple(configs, samples)

        assert len(results) == 2
        models = {r.model for r in results}
        assert "gpt-4o-mini" in models
        assert "claude-3-haiku-20240307" in models

    @pytest.mark.asyncio
    async def test_evaluate_progress_callback(self, tmp_db):
        evaluator = LLMEvaluator(db_path=tmp_db)
        config = EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=3)
        samples = [{"prompt": f"Q{i}?", "expected": "A"} for i in range(3)]

        progress_calls = []

        async def cb(done, total):
            progress_calls.append((done, total))

        mock_resp = self._mock_response("A")
        with patch("llm_eval.core.evaluator.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await evaluator.evaluate(config, samples, progress_callback=cb)

        assert len(progress_calls) == 3
        assert progress_calls[-1][0] == 3

    def test_result_to_dict(self, eval_result):
        d = eval_result.to_dict()
        assert "model" in d
        assert "accuracy" in d
        assert "run_id" in d
        assert isinstance(d["accuracy"], float)
