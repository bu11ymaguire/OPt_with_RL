"""집계 지표 테스트.

여기서 검증하는 것은 **비교의 공정성**이다. 수치 계산이 맞는지보다, 서로 다른
컨트롤러를 같은 조건에서 비교하고 있는지가 결론을 좌우한다.

프로토콜 D11 (Track E 예산 절단)
--------------------------------
optimizer 루프는 ``spent >= budget`` 에서 종료하므로 **마지막 step 이 예산을
초과한다.** 초과량은 컨트롤러가 고른 action 크기에 비례하므로, 고정 예산
비교에서 큰 step 을 고르는 컨트롤러가 공짜로 이득을 본다.

```text
C0 (평균 k=17.9)   150 GE 예산에 실제 171 GE 소모   <- 큰 step 하나가 공짜
Q=4 (평균 k=3.3)   150 GE 예산에 실제 154 GE 소모
```

이 편향은 "쿼터를 키우면 planner 가 싼 action 을 고르고 성능이 나빠진다"는
관측에 그대로 섞여 있었다. 집계에서 예산을 넘지 않는 prefix 로 잘라 제거한다.
"""

from __future__ import annotations

import pytest

from rl_newton.benchmark.metrics import (
    TargetSpec,
    budget_respecting_prefix,
    summarize_run,
)
from rl_newton.optimizers.newton_cg import OptimizationTrace
from rl_newton.types import StepRecord


def make_trace(costs, losses, *, initial_loss: float = 1.0) -> OptimizationTrace:
    trace = OptimizationTrace(run_id="t", controller="c", task_instance_id="i", seed=0)
    trace.initial_loss = initial_loss
    for i, (cost, loss) in enumerate(zip(costs, losses, strict=True)):
        trace.records.append(
            StepRecord(
                run_id="t",
                seed=0,
                optimizer="newton_cg",
                step=i,
                train_loss_before=initial_loss if i == 0 else losses[i - 1],
                train_loss_after=loss,
                cost_ge=cost,
                hvp_count=3,
                cg_budget=3,
            )
        )
    trace.final_loss = losses[-1]
    trace.total_cost_ge = sum(costs)
    # ``n_steps`` 는 records 에서 파생되는 property 다. 직접 설정하지 않는다.
    return trace


class TestBudgetRespectingPrefix:
    def test_overshooting_step_is_excluded(self):
        # 20 + 20 = 40 <= 50. 세 번째 step 을 더하면 60 > 50 이므로 제외한다.
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        loss, cost, steps = budget_respecting_prefix(trace, 50.0)
        assert steps == 2
        assert cost == pytest.approx(40.0)
        assert loss == pytest.approx(0.2)

    def test_cost_never_exceeds_budget(self):
        trace = make_trace([21.3] * 10, [0.5**i for i in range(1, 11)])
        _loss, cost, _steps = budget_respecting_prefix(trace, 150.0)
        assert cost <= 150.0

    def test_big_step_controller_loses_its_free_overshoot(self):
        """절단 전에는 큰 step 컨트롤러가 예산을 초과해 쓰고 있었다."""
        big = make_trace([21.3] * 8, [10.0**-i for i in range(1, 9)])
        assert big.total_cost_ge > 150.0
        _loss, big_cost, _steps = budget_respecting_prefix(big, 150.0)
        assert big_cost <= 150.0

        small = make_trace([4.3] * 40, [10.0 ** -(i * 0.2) for i in range(1, 41)])
        _l, small_cost, _s = budget_respecting_prefix(small, 150.0)
        assert small_cost <= 150.0
        # 두 컨트롤러의 예산 사용량 차이가 한 step 비용 이내로 줄어든다.
        assert abs(big_cost - small_cost) <= 21.3

    def test_single_step_over_budget_yields_initial_loss(self):
        """첫 step 조차 예산을 넘으면 아무 진행도 인정하지 않는다."""
        trace = make_trace([100.0], [0.1])
        loss, cost, steps = budget_respecting_prefix(trace, 50.0)
        assert steps == 0
        assert cost == pytest.approx(0.0)
        assert loss == pytest.approx(trace.initial_loss)

    def test_none_budget_leaves_trace_untouched(self):
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        assert budget_respecting_prefix(trace, None) == (
            trace.final_loss,
            trace.total_cost_ge,
            trace.n_steps,
        )

    def test_nan_step_stops_the_prefix(self):
        trace = make_trace([10.0, 10.0, 10.0], [0.5, float("nan"), 0.01])
        loss, _cost, steps = budget_respecting_prefix(trace, 100.0)
        assert steps == 2
        assert loss != loss  # NaN 은 결과 자체다. 조용히 건너뛰지 않는다.


class TestSummarizeRunFairness:
    def test_summarize_run_applies_the_prefix(self):
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        summary = summarize_run(
            trace, TargetSpec(metric="relative_loss", value=1.0e-6), budget_ge=50.0
        )
        assert summary.total_cost_ge == pytest.approx(40.0)
        assert summary.final_loss == pytest.approx(0.2)
        assert summary.n_steps == 2

    def test_track_t_metrics_are_not_truncated(self):
        """cost-to-target 은 도달 시점으로 정의되므로 예산 절단과 무관하다.

        Track E 를 공정하게 만드는 수정이 Track T 의 정의를 바꾸면 안 된다.
        """
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 1.0e-9])
        summary = summarize_run(
            trace, TargetSpec(metric="relative_loss", value=1.0e-6), budget_ge=50.0
        )
        assert summary.reached
        assert summary.cost_to_target_ge == pytest.approx(60.0)

    def test_without_budget_behaviour_is_unchanged(self):
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        summary = summarize_run(trace, TargetSpec(metric="relative_loss", value=1.0e-6))
        assert summary.final_loss == pytest.approx(0.01)
        assert summary.n_steps == 3
