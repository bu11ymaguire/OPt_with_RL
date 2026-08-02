"""Stage 2: 컨트롤러와 planner 의 구현 불변조건.

**horizon 이 늘면 실현 성능이 좋아진다고 가정하지 않는다.** beam search 는 정확한
planner 가 아니고, MPC 는 매 step 재계획하므로 깊은 탐색이 좋은 branch 를 중간에
잘라낼 수 있다. 따라서 테스트하는 것은 구현 불변조건이다.

  - 같은 시퀀스를 같은 초기 상태에서 실행하면 같은 terminal loss / cost
  - planner 효용이 실제 terminal objective 와 일치한다
  - **incumbent carry-over**: 이전 horizon 의 최선이 최종 후보에 남는다
    (이것 덕분에 planner **효용**의 단조성만은 보장된다)
  - beam pruning 전후 상태 복원이 정확하다
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_newton.optimizers.action_space import (
    ABSOLUTE,
    LATTICE_BASE,
    NARROW,
    WIDE,
    ActionSpace,
)
from rl_newton.optimizers.controllers import (
    FixedController,
    HeuristicController,
    HorizonPlannerController,
    OneStepEfficiencyController,
    OpenLoopController,
    ScheduleSegment,
    efficiency_score,
    horizon_utility,
    make_open_loop_controller,
)
from rl_newton.optimizers.newton_cg import (
    NewtonCGConfig,
    NewtonCGOptimizer,
    StepContext,
    apply_damping_action,
)
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.types import ControllerAction

NARROW_F = NARROW.with_fixed_step_size(1.0)
WIDE_F = WIDE.with_fixed_step_size(1.0)


def make_task(kappa: float = 1.0e3, d: int = 32, seed: int = 0) -> QuadraticTask:
    return QuadraticTask(QuadraticSpec(dimension=d, condition_number=kappa), seed=seed)


def make_optimizer(controller, *, budget: float = 200.0, steps: int = 50):
    task = make_task()
    config = NewtonCGConfig(total_steps=steps, cost_budget_ge=budget, initial_damping=1.0e-2)
    return NewtonCGOptimizer(task, controller, config, run_id="t", seed=0)


# ---------------------------------------------------------------------------
# 행동 공간: 게이트 B의 전제
# ---------------------------------------------------------------------------


class TestActionSpaceResolution:
    def test_narrow_multipliers_are_exact_reciprocals(self):
        """``3 x (1/3) = 1`` 이 정확해야 damping 표류가 없다.

        README 원안 ``0.3`` 은 ``3 x 0.3 = 0.9`` 라서 배수를 번갈아 고르면
        step 당 10% 아래로 밀린다.
        """
        values = NARROW.damping_values
        assert len(values) == 3
        assert values[0] * values[2] == pytest.approx(1.0, rel=1e-12)

    def test_alternating_multipliers_do_not_drift(self):
        """올렸다 되돌리기를 반복해도 damping 이 제자리로 와야 한다."""
        log_damping = -2.0
        up = ControllerAction(damping_multiplier=3.0, cg_budget=5, step_size=1.0)
        down = ControllerAction(damping_multiplier=1.0 / 3.0, cg_budget=5, step_size=1.0)
        for _ in range(20):
            log_damping = apply_damping_action(log_damping, up, min_log10=-8, max_log10=8)
            log_damping = apply_damping_action(log_damping, down, min_log10=-8, max_log10=8)
        assert log_damping == pytest.approx(-2.0, abs=1e-9)

    def test_all_spaces_share_log_resolution(self):
        """게이트 B가 도달성 손실만 재려면 로그 해상도가 같아야 한다.

        ``absolute`` 가 범위만 넓고 해상도가 거칠면 도달성 이득이 해상도
        손실에 잠식되어 두 효과를 분리할 수 없다. 실제로 초기 구성(2 decade
        간격)에서 absolute 가 narrow 보다 나쁜 결과를 냈다.
        """
        expected = math.log10(3.0)
        for space in (NARROW, WIDE, ABSOLUTE):
            logs = sorted(math.log10(v) for v in space.damping_values)
            gaps = [b - a for a, b in zip(logs, logs[1:], strict=False)]
            for gap in gaps:
                assert gap == pytest.approx(expected, rel=1e-9), space.name

    def test_absolute_has_widest_range(self):
        assert ABSOLUTE.log10_span > WIDE.log10_span > NARROW.log10_span

    def test_all_damping_values_lie_on_the_same_power_of_three_lattice(self):
        """세 공간이 같은 격자 위에 있어야 게이트 B가 범위 차이만 잰다.

        ``NARROW`` 와 ``WIDE`` 의 배수 집합은 ``ABSOLUTE`` 의 값 집합과 같은
        ``3^e`` 격자에 놓인다. 해상도가 교란 요인이 되지 않는다.
        """
        for space in (NARROW, WIDE, ABSOLUTE):
            for value in space.damping_values:
                exponent = math.log(value) / math.log(LATTICE_BASE)
                assert exponent == pytest.approx(round(exponent), abs=1e-9), (
                    f"{space.name}: {value}"
                )

    def test_absolute_range_fits_inside_default_damping_bounds(self):
        """경계에서 클립되면 서로 다른 action 이 같은 damping 으로 붕괴한다."""
        config = NewtonCGConfig()
        assert min(ABSOLUTE.damping_values) > config.min_damping
        assert max(ABSOLUTE.damping_values) < config.max_damping

    def test_cg_budgets_are_preserved_across_presets(self):
        """CG budget 은 이 프로젝트의 핵심 inexactness 축이므로 축소하지 않는다."""
        for space in (NARROW, WIDE, ABSOLUTE):
            assert space.cg_budgets == (3, 5, 10, 20), space.name

    def test_fixed_step_size_subset_only(self):
        with pytest.raises(ValueError, match="not in"):
            NARROW.with_fixed_step_size(0.7)

    def test_solve_groups_share_step_sizes(self):
        """step_size 는 CG solve 에 영향이 없으므로 결과를 공유한다."""
        groups = list(NARROW.iter_solve_groups())
        assert len(groups) == NARROW.n_solve_groups
        for representative, step_sizes in groups:
            assert step_sizes == NARROW.step_sizes
            assert representative.step_size == NARROW.step_sizes[0]

    def test_absolute_actions_carry_absolute_damping(self):
        action = ABSOLUTE.action_from_flat(0)
        assert action.is_absolute
        assert action.damping_absolute is not None
        assert not NARROW.action_from_flat(0).is_absolute


# ---------------------------------------------------------------------------
# 효용 함수: terminal objective 가 ratio 합산과 다름을 고정
# ---------------------------------------------------------------------------


class TestHorizonUtility:
    def test_fixed_budget_is_cumulative_not_per_step_sum(self):
        """누적 효율은 step 별 비율의 합과 다르다. 초판 결함의 회귀 테스트.

        두 step 으로 각각 loss 를 절반씩 줄이며 1 GE 를 쓴 경우:
          per-step ratio 합 = ln2/1 + ln2/1 = 1.386
          누적 효용        = ln4/2            = 0.693
        """
        per_step_sum = efficiency_score(1.0, 0.5, 1.0) + efficiency_score(0.5, 0.25, 1.0)
        cumulative = horizon_utility(1.0, 0.25, 2.0, track="fixed_budget")

        assert per_step_sum == pytest.approx(2.0 * math.log(2.0))
        assert cumulative == pytest.approx(math.log(4.0) / 2.0)
        assert cumulative != pytest.approx(per_step_sum)

    def test_fixed_budget_ranks_by_rate_not_by_absolute_gain(self):
        """고정 예산 효용은 **비율**로 순위를 정한다. 절대 감소량이 아니다.

        이것이 파일럿 현상의 메커니즘이다. CG 반복은 수익이 체감하므로
        ``k=3`` 이 절대 감소량은 작아도 비용당 감소량은 클 수 있고, 그러면
        국소 효율 기준에서 이긴다.
        """
        # 절대 감소량은 작지만 비용당으로는 큰 경우
        cheap_high_rate = horizon_utility(1.0, 0.5, 3.0, track="fixed_budget")
        costly_low_rate = horizon_utility(1.0, 0.1, 20.0, track="fixed_budget")
        assert cheap_high_rate == pytest.approx(math.log(2.0) / 3.0)
        assert costly_low_rate == pytest.approx(math.log(10.0) / 20.0)
        assert cheap_high_rate > costly_low_rate

        # 반대로 비용당 감소량이 작으면 싼 행동도 진다
        cheap_low_rate = horizon_utility(1.0, 0.9, 3.0, track="fixed_budget")
        costly_high_rate = horizon_utility(1.0, 0.4, 20.0, track="fixed_budget")
        assert cheap_low_rate < costly_high_rate

    def test_cost_to_target_returns_negative_total_cost_when_reached(self):
        u = horizon_utility(1.0, 1.0e-7, 42.0, track="cost_to_target", target_loss=1.0e-6)
        assert u == pytest.approx(-42.0)

    def test_cost_to_target_estimates_remaining_cost(self):
        """미도달이면 남은 거리를 관측 진행률로 나눠 예상 총비용을 만든다."""
        # 10 GE 로 loss 를 1 -> 0.1 (ln10 nat). 목표는 0.01 (추가로 ln10 필요).
        u = horizon_utility(1.0, 0.1, 10.0, track="cost_to_target", target_loss=0.01)
        assert u == pytest.approx(-20.0, rel=1e-9)

    def test_cost_to_target_prefers_lower_total_cost(self):
        fast = horizon_utility(1.0, 0.1, 10.0, track="cost_to_target", target_loss=0.01)
        slow = horizon_utility(1.0, 0.5, 10.0, track="cost_to_target", target_loss=0.01)
        assert fast > slow

    def test_no_progress_is_rejected(self):
        assert horizon_utility(1.0, 1.0, 5.0, track="fixed_budget") == -math.inf
        assert horizon_utility(1.0, 2.0, 5.0, track="fixed_budget") == -math.inf
        assert horizon_utility(1.0, float("nan"), 5.0, track="fixed_budget") == -math.inf


# ---------------------------------------------------------------------------
# 시뮬레이션 결정론성과 상태 복원
# ---------------------------------------------------------------------------


class TestSimulationInvariants:
    def test_same_sequence_from_same_state_gives_same_outcome(self):
        """planner 가 신뢰할 수 있으려면 시뮬레이션이 결정론적이어야 한다."""
        optimizer = make_optimizer(FixedController(NARROW_F.action_from_flat(0)))
        actions = [NARROW_F.action_from_flat(i) for i in (0, 5, 11)]

        def run_sequence() -> list[tuple[float, float, bool]]:
            out = []
            for action in actions:
                out.append(optimizer.simulate_step(action))
            return out

        root = optimizer.snapshot()
        first = run_sequence()
        optimizer.restore(root)
        second = run_sequence()

        assert first == second

    def test_restore_recovers_parameters_and_damping_exactly(self):
        optimizer = make_optimizer(FixedController(NARROW_F.action_from_flat(0)))
        root_params, root_log = optimizer.snapshot()

        optimizer.simulate_step(NARROW_F.action_from_flat(11))
        assert not torch.allclose(optimizer.flattener.flatten_params(), root_params)

        optimizer.restore((root_params, root_log))
        assert torch.equal(optimizer.flattener.flatten_params(), root_params)
        assert optimizer.damping_log10 == root_log

    def test_simulate_step_cost_is_charged_to_search(self):
        """planner 비용은 본문 비용에 섞이지 않아야 한다 (프로토콜 D5)."""
        planner = HorizonPlannerController(NARROW_F, horizon=2, beam_width=2)
        optimizer = make_optimizer(planner, budget=60.0, steps=3)
        trace = optimizer.run()

        assert trace.search_cost_ge > trace.total_cost_ge
        assert trace.search_hvp > 0


# ---------------------------------------------------------------------------
# incumbent carry-over: 게이트 C 해석의 전제
# ---------------------------------------------------------------------------


class TestIncumbentCarryOver:
    def _utility_at_first_step(self, space: ActionSpace, horizon: int) -> float:
        planner = HorizonPlannerController(space, horizon=horizon, beam_width=2)
        task = make_task()
        config = NewtonCGConfig(total_steps=1, cost_budget_ge=1.0e9, initial_damping=1.0e-2)
        optimizer = NewtonCGOptimizer(task, planner, config, run_id="u", seed=0)
        optimizer.run()
        return planner.last_utility

    @pytest.mark.parametrize("space", [NARROW_F, WIDE_F], ids=["narrow", "wide"])
    def test_planner_utility_is_monotone_in_horizon(self, space: ActionSpace):
        """incumbent carry-over 덕분에 **효용**은 H 가 늘어도 감소하지 않는다.

        실현 성능의 단조성은 보장되지 않는다 (MPC 는 매 step 재계획하고 beam
        search 는 정확하지 않다). 그래서 테스트는 효용에만 적용한다.
        """
        u1 = self._utility_at_first_step(space, 1)
        u3 = self._utility_at_first_step(space, 3)

        assert math.isfinite(u1)
        assert u3 >= u1 - 1e-12

    def test_chosen_depth_is_recorded(self):
        """planner 가 실제로 깊은 계획을 쓰는지 관측 가능해야 한다.

        항상 1이면 horizon 을 늘려도 의미가 없다는 직접적 증거다.
        """
        planner = HorizonPlannerController(NARROW_F, horizon=3, beam_width=2)
        make_optimizer(planner, budget=80.0, steps=4).run()

        assert planner.choices
        assert all(1 <= c.chosen_depth <= 3 for c in planner.choices)


# ---------------------------------------------------------------------------
# H=1 등가성: 게이트 A1/B가 one-step 구현을 쓰는 근거
# ---------------------------------------------------------------------------


class TestOneStepEquivalence:
    def test_one_step_is_far_cheaper_than_h1_planner(self):
        """one-step 은 HVP 그래프를 후보 전체에 공유하므로 훨씬 싸다.

        absolute (34 damping x 4 budget) 를 감당할 수 있는 유일한 경로다.
        """
        onestep_trace = make_optimizer(
            OneStepEfficiencyController(NARROW_F), budget=60.0, steps=3
        ).run()
        planner_trace = make_optimizer(
            HorizonPlannerController(NARROW_F, horizon=1, beam_width=1),
            budget=60.0,
            steps=3,
        ).run()

        assert onestep_trace.search_cost_ge < planner_trace.search_cost_ge

    def test_absolute_sweep_cost_is_bounded(self):
        """absolute 는 비싸지만 one-step 경로로는 감당 가능한 범위여야 한다."""
        absolute_f = ABSOLUTE.with_fixed_step_size(1.0)
        assert len(absolute_f) == len(ABSOLUTE.damping_values) * 4
        # sweep 당 HVP = n_damping x sum(budgets)
        assert absolute_f.hvp_per_sweep == len(ABSOLUTE.damping_values) * 38


# ---------------------------------------------------------------------------
# 기타 컨트롤러
# ---------------------------------------------------------------------------


class TestBaselineControllers:
    def test_fixed_controller_always_returns_same_action(self):
        action = NARROW_F.action_from_flat(3)
        controller = FixedController(action)
        context = StepContext(step=0, total_steps=10, loss=1.0, grad_norm=1.0, damping=1e-2)
        assert controller.select(context, None) is action  # type: ignore[arg-type]

    def test_open_loop_switches_on_progress_only(self):
        early = NARROW_F.action_from_flat(0)
        late = NARROW_F.action_from_flat(11)
        controller = OpenLoopController([ScheduleSegment(0.5, early), ScheduleSegment(1.0, late)])
        assert controller.action_at(0.0) is early
        assert controller.action_at(0.5) is early
        assert controller.action_at(0.51) is late
        assert controller.action_at(1.0) is late

    def test_open_loop_requires_sorted_segments_covering_one(self):
        a = NARROW_F.action_from_flat(0)
        with pytest.raises(ValueError, match="sorted"):
            OpenLoopController([ScheduleSegment(1.0, a), ScheduleSegment(0.5, a)])
        with pytest.raises(ValueError, match="progress 1.0"):
            OpenLoopController([ScheduleSegment(0.5, a)])

    def test_make_open_loop_controller_validates_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            make_open_loop_controller(NARROW_F, [0, 1], [1.0])

    def test_heuristic_rejects_absolute_space(self):
        """absolute 는 분석 전용이다. 상대 배수를 전제하는 규칙에 쓸 수 없다."""
        with pytest.raises(ValueError, match="absolute"):
            HeuristicController(ABSOLUTE)

    def test_heuristic_moves_conservative_on_low_trust(self):
        from rl_newton.types import StepRecord

        controller = HeuristicController(NARROW_F, initial_flat=len(NARROW_F) // 2)
        low_trust = StepRecord(
            run_id="t",
            seed=0,
            optimizer="h",
            step=0,
            train_loss_before=1.0,
            train_loss_after=0.99,
            trust_ratio=0.01,
        )
        context = StepContext(
            step=1,
            total_steps=10,
            loss=0.99,
            grad_norm=1.0,
            damping=1e-2,
            previous=low_trust,
        )
        action = controller.select(context, None)  # type: ignore[arg-type]
        # damping 을 올리고 예산을 늘리는 방향이어야 한다
        assert action.damping_multiplier >= 1.0

    def test_all_controllers_run_without_crashing(self):
        controllers = [
            FixedController(NARROW_F.action_from_flat(5)),
            HeuristicController(NARROW_F),
            OneStepEfficiencyController(NARROW_F),
            HorizonPlannerController(NARROW_F, horizon=2, beam_width=2),
            make_open_loop_controller(NARROW_F, [0, 5, 11], [0.3, 0.7, 1.0]),
        ]
        for controller in controllers:
            trace = make_optimizer(controller, budget=80.0, steps=6).run()
            assert trace.n_steps >= 1
            assert math.isfinite(trace.final_loss)
            assert trace.final_loss <= trace.initial_loss


class TestCostBudgetTermination:
    def test_run_stops_on_cost_budget(self):
        """비교는 GE 예산 기준이어야 한다 (README §4.2)."""
        trace = make_optimizer(
            FixedController(NARROW_F.action_from_flat(11)), budget=100.0, steps=1000
        ).run()

        assert trace.stop_reason == "cost_budget"
        assert trace.total_cost_ge >= 100.0
        assert trace.n_steps < 1000

    def test_expensive_actions_consume_budget_faster(self):
        """k=20 은 k=3 보다 같은 예산에서 적은 step 을 쓴다."""
        cheap = next(a for a in NARROW_F.iter_actions() if a.cg_budget == 3)
        costly = next(a for a in NARROW_F.iter_actions() if a.cg_budget == 20)

        cheap_trace = make_optimizer(FixedController(cheap), budget=200.0, steps=1000).run()
        costly_trace = make_optimizer(FixedController(costly), budget=200.0, steps=1000).run()

        assert cheap_trace.n_steps > costly_trace.n_steps
