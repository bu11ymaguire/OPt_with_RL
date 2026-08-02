"""재개 가능한 결과 저장소 테스트.

프로세스가 끊겨도 계산한 결과를 잃지 않아야 한다. Stage 2는 수백~수천 run 이라
이 성질이 없으면 실험을 완주할 수 없다.
"""

from __future__ import annotations

import json
import math

import pytest

from rl_newton.benchmark.metrics import RunSummary
from rl_newton.benchmark.store import ResultStore, RunKey, RunRecord


def make_summary(
    *,
    controller: str = "best_static",
    instance: str = "quad_spd_d64_k1e+02_seed0",
    seed: int = 0,
    target: str = "relative_loss<=0.0001",
    reached: bool = True,
    cost: float | None = 123.5,
) -> RunSummary:
    return RunSummary(
        run_id=f"{controller}|{instance}",
        controller=controller,
        task_instance_id=instance,
        seed=seed,
        target=target,
        reached=reached,
        cost_to_target_ge=cost,
        steps_to_target=7 if reached else None,
        hvp_to_target=90 if reached else None,
        initial_loss=1.0,
        final_loss=1.0e-5,
        total_cost_ge=600.0,
        total_hvp=520,
        search_cost_ge=0.0,
        n_steps=30,
        stop_reason="cost_budget",
        rejection_rate=0.0,
        failure_rate=0.0,
        negative_curvature_rate=0.0,
        cg_convergence_rate=0.1,
        median_residual_ratio=0.2,
        median_damping=1.0e-2,
        median_trust_ratio=1.0,
    )


class TestRunKey:
    def test_key_string_is_stable_and_unique(self):
        a = RunKey("fixed", "inst", 0, "t")
        b = RunKey("fixed", "inst", 0, "t")
        c = RunKey("fixed", "inst", 1, "t")

        assert a.as_str() == b.as_str()
        assert a.as_str() != c.as_str()

    def test_key_from_summary_round_trips(self):
        summary = make_summary()
        key = RunKey.from_summary(summary)

        assert key.controller == summary.controller
        assert key.task_instance_id == summary.task_instance_id
        assert key.seed == summary.seed
        assert key.target == summary.target


class TestResume:
    def test_completed_run_is_skipped_on_reopen(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, wall_clock_sec=1.5)

        reopened = ResultStore(path)
        assert reopened.is_completed(RunKey.from_summary(summary))
        assert len(reopened) == 1

    def test_failed_run_is_retried(self, tmp_path):
        """실패는 건너뛰지 않는다. 원인은 보존한다 (README §15)."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        key = RunKey("mpc_H3_narrow", "inst", 0, "t")
        store.record_failure(key, "RuntimeError: boom")

        reopened = ResultStore(path)
        assert not reopened.is_completed(key)
        failures = reopened.failures()
        assert len(failures) == 1
        assert "boom" in str(failures[0].error)

    def test_retry_after_failure_marks_completed(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        key = RunKey.from_summary(summary)

        store.record_failure(key, "transient")
        assert not store.is_completed(key)

        store.record_success(summary, wall_clock_sec=2.0)
        assert store.is_completed(key)
        # 마지막 기록이 유효하다
        assert ResultStore(path).is_completed(key)

    def test_summary_survives_round_trip(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        original = make_summary()
        ResultStore(path).record_success(original, wall_clock_sec=3.0)

        restored = ResultStore(path).summaries()[0]

        assert restored.controller == original.controller
        assert restored.reached is True
        assert restored.cost_to_target_ge == pytest.approx(123.5)
        assert restored.total_cost_ge == pytest.approx(600.0)
        assert restored.n_steps == 30
        assert restored.stop_reason == "cost_budget"

    def test_unreached_run_keeps_none_cost(self, tmp_path):
        """절단 규칙: 미도달은 큰 값으로 대입되지 않는다 (프로토콜 D6)."""
        path = tmp_path / "runs.jsonl"
        summary = make_summary(reached=False, cost=None)
        ResultStore(path).record_success(summary, wall_clock_sec=1.0)

        restored = ResultStore(path).summaries()[0]
        assert restored.reached is False
        assert restored.cost_to_target_ge is None

    def test_non_finite_floats_round_trip_as_nan(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        summary = make_summary()
        summary.median_trust_ratio = float("nan")
        ResultStore(path).record_success(summary, wall_clock_sec=float("nan"))

        restored = ResultStore(path).summaries()[0]
        assert math.isnan(restored.median_trust_ratio)

    def test_output_is_valid_json_per_line(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), wall_clock_sec=1.0)
        store.record_failure(RunKey("x", "i", 1, "t"), "err")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # allow_nan 없이 파싱 가능해야 한다

    def test_corrupt_trailing_line_is_skipped(self, tmp_path):
        """프로세스가 쓰는 중 끊기면 마지막 줄이 깨질 수 있다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), wall_clock_sec=1.0)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"key": {"controller": "broken"')  # 미완성 JSON

        reopened = ResultStore(path)
        assert len(reopened) == 1
        assert reopened.is_completed(RunKey.from_summary(make_summary()))


class TestMetadata:
    def test_provenance_is_attached(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path, git_commit="abc1234", config_hash="deadbeef")
        store.record_success(make_summary(), wall_clock_sec=1.0)

        record = next(iter(ResultStore(path)))
        assert record.git_commit == "abc1234"
        assert record.config_hash == "deadbeef"
        assert record.recorded_at

    def test_action_counts_and_depths_are_preserved(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(
            make_summary(),
            wall_clock_sec=1.0,
            action_counts={"m=3,k=20,a=1": 5},
            chosen_depths={"1": 3, "3": 2},
        )

        record = next(iter(ResultStore(path)))
        assert record.action_counts == {"m=3,k=20,a=1": 5}
        assert record.chosen_depths == {"1": 3, "3": 2}

    def test_wall_clock_is_recorded_separately_from_ge(self, tmp_path):
        """wall-clock 은 GE 와 별개로 기록한다 (프로토콜 D1)."""
        path = tmp_path / "runs.jsonl"
        ResultStore(path).record_success(make_summary(), wall_clock_sec=12.75)

        record = next(iter(ResultStore(path)))
        assert record.wall_clock_sec == pytest.approx(12.75)
        assert record.summary is not None
        assert record.summary.total_cost_ge == pytest.approx(600.0)

    def test_filter_summaries_by_controller(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(controller="a"), wall_clock_sec=1.0)
        store.record_success(make_summary(controller="b"), wall_clock_sec=1.0)

        assert len(store.summaries()) == 2
        assert len(store.summaries(controller="a")) == 1
        assert store.controllers() == ["a", "b"]

    def test_describe_reports_counts(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), wall_clock_sec=1.0)
        store.record_failure(RunKey("x", "i", 1, "t"), "err")

        text = store.describe()
        assert "완료 1" in text
        assert "실패 1" in text


class TestRunRecordSerialization:
    def test_failure_record_has_no_summary(self, tmp_path):
        record = RunRecord(
            key=RunKey("c", "i", 0, "t"), status="failed", error="ValueError: x"
        )
        payload = record.to_json()

        assert payload["summary"] is None
        assert payload["status"] == "failed"

        restored = RunRecord.from_json(payload)
        assert restored.summary is None
        assert restored.error == "ValueError: x"
