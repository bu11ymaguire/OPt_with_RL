"""재개 가능한 결과 저장소 테스트.

프로세스가 끊겨도 계산한 결과를 잃지 않아야 한다. Stage 2는 수백~수천 run 이라
이 성질이 없으면 실험을 완주할 수 없다.

가장 위험한 실패 모드는 재개 기능 자체가 **낡은 결과를 새 결과로 착각**하는
것이다. 설정이 바뀌었는데 같은 조합으로 보고 건너뛰면, 재개가 실험을 오염시킨다.
그래서 실험 정체성 테스트를 가장 중요하게 다룬다.
"""

from __future__ import annotations

import json
import math

import pytest

from rl_newton.benchmark.metrics import RunSummary
from rl_newton.benchmark.store import (
    ResultStore,
    RunKey,
    RunRecord,
    environment_fingerprint,
    experiment_id,
)

EXP = "exp0001"


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


# ---------------------------------------------------------------------------
# 실험 정체성: 재개가 실험을 오염시키지 않도록 하는 핵심 장치
# ---------------------------------------------------------------------------


class TestExperimentIdentity:
    def test_same_payload_gives_same_id(self):
        payload = {"beam": 2, "budget": 600.0, "cg_budgets": [3, 5, 10, 20]}
        assert experiment_id(payload) == experiment_id(dict(payload))

    def test_key_order_does_not_matter(self):
        a = {"beam": 2, "budget": 600.0}
        b = {"budget": 600.0, "beam": 2}
        assert experiment_id(a) == experiment_id(b)

    @pytest.mark.parametrize(
        "change",
        [
            {"beam": 4},
            {"budget": 1200.0},
            {"cg_budgets": [5, 20]},
            {"horizons": [1, 3]},
            {"damping_values": [0.5, 1.0, 2.0]},
            {"max_steps": 400},
            {"protocol_version": "stage2-v2"},
            {"code_dirty": True},
        ],
    )
    def test_any_config_change_gives_new_id(self, change):
        """설정이 하나라도 바뀌면 다른 실험이다.

        ``(controller, task, seed, target)`` 만으로 완료를 판단하면 beam, horizon,
        GE 예산, action space, CG budget 중 무엇이 바뀌어도 낡은 결과를 재사용한다.
        """
        base = {
            "beam": 2,
            "budget": 600.0,
            "cg_budgets": [3, 5, 10, 20],
            "horizons": [1, 3, 5],
            "damping_values": [1 / 3, 1.0, 3.0],
            "max_steps": 200,
            "protocol_version": "stage2-v1",
            "code_dirty": False,
        }
        assert experiment_id(base) != experiment_id(base | change)

    def test_key_includes_experiment_id(self):
        a = RunKey("expA", "fixed", "inst", 0, "t")
        b = RunKey("expB", "fixed", "inst", 0, "t")
        assert a.as_str() != b.as_str()
        assert a.as_str().startswith("expA|")

    def test_completed_run_is_not_skipped_after_config_change(self, tmp_path):
        """가장 위험한 회귀. 설정이 바뀌면 반드시 다시 실행되어야 한다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, "exp_beam2", wall_clock_sec=1.0)

        old_key = RunKey.from_summary(summary, "exp_beam2")
        new_key = RunKey.from_summary(summary, "exp_beam4")

        assert store.is_completed(old_key)
        assert not store.is_completed(new_key)


class TestRunKey:
    def test_key_string_is_stable_and_unique(self):
        a = RunKey(EXP, "fixed", "inst", 0, "t")
        b = RunKey(EXP, "fixed", "inst", 0, "t")
        c = RunKey(EXP, "fixed", "inst", 1, "t")

        assert a.as_str() == b.as_str()
        assert a.as_str() != c.as_str()

    def test_key_from_summary_round_trips(self):
        summary = make_summary()
        key = RunKey.from_summary(summary, EXP)

        assert key.experiment_id == EXP
        assert key.controller == summary.controller
        assert key.task_instance_id == summary.task_instance_id
        assert key.seed == summary.seed
        assert key.target == summary.target


class TestResume:
    def test_completed_run_is_skipped_on_reopen(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, EXP, wall_clock_sec=1.5)

        reopened = ResultStore(path)
        assert reopened.is_completed(RunKey.from_summary(summary, EXP))
        assert len(reopened) == 1

    def test_no_duplicate_records_for_same_key(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, EXP, wall_clock_sec=1.0)
        store.record_success(summary, EXP, wall_clock_sec=2.0)

        reopened = ResultStore(path)
        assert len(reopened) == 1  # 인덱스는 마지막 것만 유지
        assert len(reopened.summaries()) == 1

    def test_failed_run_is_retried(self, tmp_path):
        """실패는 건너뛰지 않는다. 원인은 보존한다 (README §15)."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        key = RunKey(EXP, "mpc_H3_narrow", "inst", 0, "t")
        store.record_failure(key, "RuntimeError: boom")

        reopened = ResultStore(path)
        assert not reopened.is_completed(key)
        failures = reopened.failures()
        assert len(failures) == 1
        assert "boom" in str(failures[0].error)

    def test_failed_run_is_excluded_from_summaries(self, tmp_path):
        """실패가 평균에 섞이면 안 된다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        store.record_failure(RunKey(EXP, "other", "inst", 1, "t"), "err")

        assert len(store.summaries()) == 1
        assert len(ResultStore(path).summaries()) == 1

    def test_retry_after_failure_marks_completed(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        key = RunKey.from_summary(summary, EXP)

        store.record_failure(key, "transient")
        assert not store.is_completed(key)

        store.record_success(summary, EXP, wall_clock_sec=2.0)
        assert store.is_completed(key)
        assert ResultStore(path).is_completed(key)

    def test_summary_survives_round_trip(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        original = make_summary()
        ResultStore(path).record_success(original, EXP, wall_clock_sec=3.0)

        restored = ResultStore(path).summaries()[0]

        assert restored.controller == original.controller
        assert restored.reached is True
        assert restored.cost_to_target_ge == pytest.approx(123.5)
        assert restored.total_cost_ge == pytest.approx(600.0)
        assert restored.n_steps == 30
        assert restored.stop_reason == "cost_budget"

    def test_unreached_run_keeps_none_cost(self, tmp_path):
        """절단 규칙: 미도달은 큰 값으로도, NaN 으로도 대입되지 않는다 (D6).

        ``None`` 은 "목표에 도달하지 못했다", NaN 은 "값을 모른다"다. 둘을
        섞으면 도달률과 cost-to-target 집계가 오염된다.
        """
        path = tmp_path / "runs.jsonl"
        summary = make_summary(reached=False, cost=None)
        ResultStore(path).record_success(summary, EXP, wall_clock_sec=1.0)

        restored = ResultStore(path).summaries()[0]
        assert restored.reached is False
        assert restored.cost_to_target_ge is None
        assert restored.steps_to_target is None

    def test_non_finite_floats_round_trip_as_nan(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        summary = make_summary()
        summary.median_trust_ratio = float("nan")
        ResultStore(path).record_success(summary, EXP, wall_clock_sec=float("nan"))

        restored = ResultStore(path).summaries()[0]
        assert math.isnan(restored.median_trust_ratio)

    def test_output_is_valid_json_per_line(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        store.record_failure(RunKey(EXP, "x", "i", 1, "t"), "err")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # allow_nan 없이 파싱 가능해야 한다

    def test_corrupt_trailing_line_is_skipped(self, tmp_path):
        """프로세스가 쓰는 중 끊기면 마지막 줄이 깨질 수 있다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"key": {"controller": "broken"')  # 미완성 JSON

        reopened = ResultStore(path)
        assert len(reopened) == 1
        assert reopened.is_completed(RunKey.from_summary(make_summary(), EXP))


class TestMetadata:
    def test_provenance_is_attached(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path, git_commit="abc1234", config_hash="deadbeef")
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)

        record = next(iter(ResultStore(path)))
        assert record.git_commit == "abc1234"
        assert record.config_hash == "deadbeef"
        assert record.recorded_at

    def test_action_counts_and_depths_are_preserved(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(
            make_summary(),
            EXP,
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
        ResultStore(path).record_success(make_summary(), EXP, wall_clock_sec=12.75)

        record = next(iter(ResultStore(path)))
        assert record.wall_clock_sec == pytest.approx(12.75)
        assert record.summary is not None
        assert record.summary.total_cost_ge == pytest.approx(600.0)

    def test_filter_summaries_by_controller(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(controller="a"), EXP, wall_clock_sec=1.0)
        store.record_success(make_summary(controller="b"), EXP, wall_clock_sec=1.0)

        assert len(store.summaries()) == 2
        assert len(store.summaries(controller="a")) == 1
        assert store.controllers() == ["a", "b"]

    def test_describe_reports_counts(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        store.record_failure(RunKey(EXP, "x", "i", 1, "t"), "err")

        text = store.describe()
        assert "완료 1" in text
        assert "실패 1" in text


class TestEnvironmentFingerprint:
    def test_pins_threads_and_records_them(self):
        """wall-clock tie-break 가 CPU 경쟁에 흔들리지 않게 고정한다."""
        info = environment_fingerprint(pin_threads=1)

        assert info["torch_num_threads"] == 1
        assert info["pinned"] == 1
        assert info["cpu_count"]
        assert "platform" in info

    def test_can_skip_pinning(self):
        info = environment_fingerprint(pin_threads=None)
        assert info["pinned"] is None


class TestRunRecordSerialization:
    def test_failure_record_has_no_summary(self):
        record = RunRecord(
            key=RunKey(EXP, "c", "i", 0, "t"), status="failed", error="ValueError: x"
        )
        payload = record.to_json()

        assert payload["summary"] is None
        assert payload["status"] == "failed"

        restored = RunRecord.from_json(payload)
        assert restored.summary is None
        assert restored.error == "ValueError: x"
        assert restored.key.experiment_id == EXP
