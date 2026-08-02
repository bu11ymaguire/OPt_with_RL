"""action 을 고르는 컨트롤러들.

모두 같은 ``Controller`` 프로토콜을 구현하고 같은 ``ActionSpace`` 에서 고른다.
차이는 **무엇을 보고 고르는가** 뿐이다.

```text
FixedController              아무것도 안 본다. 항상 같은 action
OpenLoopController           progress 만 본다 (step / total_steps)
HeuristicController          trust ratio 를 본다
OneStepEfficiencyController  현재 step 결과를 본다 (전수 시도, 국소 효율 최대)
HorizonPlannerController     H step 앞을 본다 (beam search, terminal objective)
```

어느 것도 전역 상한이 아니다
----------------------------
초판은 ``greedy_oracle`` / ``lookahead_oracle`` 이라는 이름을 쓰고 이를 헤드룸의
상한으로 해석했다. **파일럿에서 그 해석이 틀렸음이 확인됐다.** 국소 효율을 매 step
최대화한 컨트롤러가 고정 설정보다 cost-to-target 에서 나빴다 (비율 0.967x).

```text
행동 A:  3 GE 로 loss 10% 감소     -> 순간 효율 높음
행동 B: 20 GE 로 loss 60% 감소     -> 목표까지 총비용은 더 적을 수 있음
```

국소 효율 최대화와 총비용 최소화는 다른 문제다. 그래서 이름과 해석을 정정했다
(프로토콜 D9). ``OneStepEfficiencyController`` 는 상한이 아니라 **비교군의 하나**이고,
``HorizonPlannerController`` 는 유한 horizon 과 beam 폭에 제한된 근사다.

이 계층이 프로토콜 게이트를 구성한다
------------------------------------
```text
게이트 A  absolute MPC planner vs best_static  (Track E, 고정 GE 예산)
          -> 적응 제어의 내재적 여지. 작으면 연구를 접거나 음성 결과로 정리.

게이트 B  absolute vs wide vs narrow planner   (로그 해상도를 맞춘 상태에서)
          -> 도달성/행동범위 손실. 크면 행동 공간을 고친다.

게이트 C  H=1 vs H=3 vs H=5 planner            (같은 terminal objective)
          -> 장기 의사결정의 가치. 개선이 미미하면 contextual bandit 이나
             heuristic 으로 충분하고 PPO 를 시작하지 않는다.

게이트 D  best_static vs planner               (Track T, target 난이도별)
          -> cost-to-target 헤드룸. 게이트 A와 결론이 다를 수 있고
             그 불일치 자체가 결과다.
```

``OpenLoopController`` 는 별도로 결정적이다 (프로토콜 D4). RL 이 fixed 는
이기고 open_loop 는 못 이기면, 학습된 것은 **적응 제어가 아니라 스케줄**이다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rl_newton.optimizers.action_space import ActionSpace
from rl_newton.optimizers.newton_cg import (
    Candidate,
    NewtonCGOptimizer,
    StepContext,
)
from rl_newton.types import ControllerAction

__all__ = [
    "FixedController",
    "OpenLoopController",
    "HeuristicController",
    "OneStepEfficiencyController",
    "HorizonPlannerController",
    "PlannerTrack",
    "ScheduleSegment",
    "PlannerChoice",
    "make_open_loop_controller",
    "efficiency_score",
    "horizon_utility",
]

PlannerTrack = Literal["fixed_budget", "cost_to_target"]
"""planner 가 최적화하는 트랙 (프로토콜 D9).

``fixed_budget``    Track E. 동일 GE 예산에서 terminal loss 최소화
``cost_to_target``  Track T. 목표 loss 도달까지 총 GE 최소화
"""


def efficiency_score(
    loss_before: float, loss_after: float, cost_ge: float, *, loss_floor: float = 1.0e-30
) -> float:
    """``(log L_before - log L_after) / cost_GE``. 국소 효율.

    **주의: 이것을 매 step 최대화하면 총비용이 나빠질 수 있다.** 파일럿에서
    이 목적으로 매 step 최선을 고른 컨트롤러가 고정 설정보다 cost-to-target 에서
    나빴다 (비율 0.967x). 싸고 작은 행동(``k=3``)을 반복하는 유인이 생기고,
    그 행동이 다음 step 의 상태를 나쁘게 만드는 것을 보지 못한다.

    따라서 이 함수는 ``OneStepEfficiencyController`` 전용이며, 그 컨트롤러는
    **상한이 아니라 비교군의 하나**다 (프로토콜 D9). planner 는
    ``horizon_utility`` 를 쓴다.

    **CG 수렴 여부가 아니라 objective 감소를 본다.** 높은 damping 은
    ``(H + lambda I)^{-1} g ~ g / lambda`` 로 CG 를 쉽게 만들지만 실제 감소는
    느려질 수 있다. 실측에서 damping ``1e6`` 은 CG 를 30/30 수렴시키고도 최종
    loss 가 1500배 나빴다.

    개선이 없으면 ``-inf`` 를 반환한다. optimizer 가 어차피 거절할 후보다.
    """
    if not math.isfinite(loss_after) or loss_after >= loss_before:
        return -math.inf
    before = max(loss_before, loss_floor)
    after = max(loss_after, loss_floor)
    return (math.log(before) - math.log(after)) / max(cost_ge, 1e-12)


def horizon_utility(
    loss_start: float,
    loss_terminal: float,
    cumulative_cost: float,
    *,
    track: PlannerTrack,
    target_loss: float | None = None,
    loss_floor: float = 1.0e-30,
) -> float:
    """H-step 시퀀스의 효용. **per-step ratio 의 합이 아니다.**

    이것이 파일럿 실패의 핵심 수정이다. 초판 planner 는 step 별 비율을 더했고,
    그러면 각 step 의 국소 효율을 합산하는 것이므로 근시안성이 그대로 남는다.
    시퀀스는 **terminal loss 와 누적 비용**으로 평가해야 한다.

    Args:
        loss_start: 시퀀스 시작 시점 loss.
        loss_terminal: 시퀀스 종료 시점 loss.
        cumulative_cost: 시퀀스가 소모한 총 GE.
        track: ``fixed_budget`` 또는 ``cost_to_target``.
        target_loss: ``cost_to_target`` 트랙에서 목표 loss. 없으면 ``fixed_budget``
            처럼 동작한다.
        loss_floor: ``log`` 하한.

    Returns:
        클수록 좋은 효용. 진행이 없으면 ``-inf``.

    두 트랙의 효용
    --------------
    ``fixed_budget`` (Track E): 남은 예산을 가장 효율적으로 쓰는 것이 목적이므로
    **누적** 로그 감소를 **누적** 비용으로 나눈다. horizon 이 길어지면 실제
    목적(총 감소 / 총 예산)에 수렴한다.

    ```text
    U = (log L_start - log L_terminal) / cumulative_cost
    ```

    ``cost_to_target`` (Track T): 목표까지의 **예상 총비용**을 최소화한다.
    시퀀스 안에서 도달했다면 실제 누적 비용이고, 아니면 남은 로그 거리를
    관측된 진행률로 나눈 값을 더한다. 전형적인 cost-to-go 추정이다.

    ```text
    rate = (log L_start - log L_terminal) / cumulative_cost
    U    = -(cumulative_cost + remaining_log_distance / rate)
    ```
    """
    if not math.isfinite(loss_terminal) or loss_terminal >= loss_start:
        return -math.inf
    start = max(loss_start, loss_floor)
    terminal = max(loss_terminal, loss_floor)
    gain = math.log(start) - math.log(terminal)
    cost = max(cumulative_cost, 1e-12)

    if track == "fixed_budget" or target_loss is None:
        return gain / cost

    # cost_to_target: 목표까지의 예상 총비용을 최소화한다.
    target = max(target_loss, loss_floor)
    if terminal <= target:
        return -cost
    rate = gain / cost
    if rate <= 0.0:
        return -math.inf
    remaining = math.log(terminal) - math.log(target)
    return -(cost + remaining / rate)


class FixedController:
    """항상 같은 action 을 고른다.

    프로토콜 D4의 ``best_static`` baseline 은 이 컨트롤러로 행동 공간의 모든
    조합을 각각 돌려 최고를 고른 것이다. "동일 탐색 예산" 원칙의 정직한 구현이다.
    """

    def __init__(self, action: ControllerAction, *, name: str | None = None) -> None:
        self._action = action
        if name is not None:
            self._name = name
        elif action.is_absolute:
            self._name = (
                f"fixed(lam={action.damping_absolute:g},k={action.cg_budget},"
                f"a={action.step_size:g})"
            )
        else:
            self._name = (
                f"fixed(m={action.damping_multiplier:g},k={action.cg_budget},"
                f"a={action.step_size:g})"
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def action(self) -> ControllerAction:
        return self._action

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        return self._action

    def reset(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"FixedController({self._action})"


@dataclass(frozen=True, slots=True)
class ScheduleSegment:
    """open-loop 스케줄의 한 구간.

    Attributes:
        until: 이 구간이 적용되는 progress 상한 (이하).
        action: 해당 구간에서 쓸 action.
    """

    until: float
    action: ControllerAction


class OpenLoopController:
    """``progress`` 만 보고 고른다. 상태를 전혀 보지 않는다.

    프로토콜 D4의 결정적 baseline 이다. 이 컨트롤러가 얻는 이득은 전부
    "시간에 따른 스케줄"로 설명된다. curvature 나 residual 같은
    optimizer-level 신호는 쓰지 않는다.

    Example:
        >>> from rl_newton.optimizers.action_space import NARROW
        >>> early = NARROW.action_from_flat(0)
        >>> late = NARROW.action_from_flat(len(NARROW) - 1)
        >>> ctrl = OpenLoopController([ScheduleSegment(0.5, early),
        ...                            ScheduleSegment(1.0, late)])
        >>> ctrl.action_at(0.1) is early
        True
        >>> ctrl.action_at(0.9) is late
        True
    """

    def __init__(self, segments: Sequence[ScheduleSegment], *, name: str = "open_loop") -> None:
        if not segments:
            raise ValueError("segments must not be empty")
        uppers = [s.until for s in segments]
        if uppers != sorted(uppers):
            raise ValueError(f"segments must be sorted by `until`, got {uppers}")
        if uppers[-1] < 1.0:
            raise ValueError(f"last segment must cover progress 1.0, got {uppers[-1]}")
        self._segments = tuple(segments)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def segments(self) -> tuple[ScheduleSegment, ...]:
        return self._segments

    def action_at(self, progress: float) -> ControllerAction:
        for segment in self._segments:
            if progress <= segment.until:
                return segment.action
        return self._segments[-1].action

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        return self.action_at(context.progress)

    def reset(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"OpenLoopController(n_segments={len(self._segments)})"


class HeuristicController:
    """trust ratio 기반 규칙 (README §4.3).

    ```text
    rho < rho_low             -> damping 증가, step size 감소, budget 증가
    rho_low <= rho < rho_high -> 유지
    rho >= rho_high           -> damping 감소, step size 증가
    ```

    RL 과 **같은 행동 공간**에서 고르므로, "RL 이 단순 적응 규칙보다 나은가"를
    행동 공간 차이 없이 비교할 수 있다.
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        rho_low: float = 0.25,
        rho_high: float = 0.75,
        initial_flat: int | None = None,
        name: str = "heuristic",
    ) -> None:
        if not 0.0 < rho_low < rho_high:
            raise ValueError(f"require 0 < rho_low < rho_high, got {rho_low}, {rho_high}")
        if space.is_absolute:
            raise ValueError(
                "HeuristicController 는 상대 damping 모드를 전제한다. "
                "absolute 공간은 분석용 오라클 전용이다."
            )
        self._space = space
        self._rho_low = rho_low
        self._rho_high = rho_high
        self._name = name

        nd, nb, ns = space.nvec
        self._initial = (
            space.indices_from_flat(initial_flat)
            if initial_flat is not None
            else (nd // 2, nb // 2, ns - 1)
        )
        self._current = self._initial

    @property
    def name(self) -> str:
        return self._name

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        previous = context.previous
        if previous is not None:
            rho = previous.trust_ratio
            d, b, s = self._current
            nd, nb, ns = self._space.nvec

            if not math.isfinite(rho) or rho < self._rho_low:
                d = min(d + 1, nd - 1)
                s = max(s - 1, 0)
                b = min(b + 1, nb - 1)
            elif rho >= self._rho_high:
                d = max(d - 1, 0)
                s = min(s + 1, ns - 1)
            self._current = (d, b, s)

        return self._space.action_from_indices(self._current)

    def reset(self) -> None:
        self._current = self._initial

    def __repr__(self) -> str:
        return (
            f"HeuristicController(rho_low={self._rho_low}, rho_high={self._rho_high}, "
            f"space={self._space.name})"
        )


@dataclass(slots=True)
class PlannerChoice:
    """컨트롤러가 한 step 에서 내린 선택의 기록.

    정책 분석과 Stage 4 behavior cloning 데이터셋의 원재료다.
    """

    step: int
    chosen_flat: int
    chosen_score: float
    best_loss: float
    worst_loss: float
    n_finite: int
    n_candidates: int
    damping_before: float = float("nan")
    damping_after: float = float("nan")
    n_cg_converged: int = 0
    """CG 가 수렴한 후보 수. loss 감소와 분리해서 본다."""
    chosen_depth: int = 1
    """채택된 계획의 시퀀스 길이. planner 가 실제로 깊은 계획을 쓰는지 분석용.

    항상 1이면 horizon 을 늘려도 의미가 없다는 직접적 증거다.
    """


class OneStepEfficiencyController:
    """매 step 전수 시도 후 ``Δlog L / cost_GE`` 가 최대인 action 을 고른다.

    **이것은 상한이 아니다** (프로토콜 D9). 이름이 ``greedy_oracle`` 이었을 때
    상한으로 오해됐으나, 파일럿에서 고정 설정보다 cost-to-target 이 나빴다
    (비율 0.967x). 국소 효율 최대화가 총비용 최소화와 다른 문제이기 때문이다.

    비교군의 하나로서 답하는 질문은 이것이다.

    > 매 step 즉시 효율이 가장 좋은 행동을 고르면 어떻게 되는가?

    이 컨트롤러가 나쁘게 나오는 것 자체가 결과다. RL 보상을 per-step ratio 로
    설계하면 같은 함정에 빠진다는 증거이므로, 프로토콜 D3의 보상 설계 근거가 된다.

    비용
    ----
    ``ActionSpace.iter_solve_groups`` 를 쓰므로 CG solve 는
    ``damping x budget`` 회만 수행된다. NARROW 기준 한 step 당

    ```text
    HvpGraph 생성    1회
    CG solve        12회  (3 damping x 4 budget)
    HVP 합계        3 x (3+5+10+20) = 114회
    forward 평가     36회  (12 방향 x 3 step size)
    ```

    순진한 구현(36 CG solve)보다 6배 이상 싸다. 이 비용은
    ``OptimizationTrace.search_cost_ge`` 로 **본문 비용과 분리해** 회계된다
    (프로토콜 D5).
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        loss_floor: float = 1.0e-30,
        name: str | None = None,
    ) -> None:
        self._space = space
        self._loss_floor = loss_floor
        self._name = name or f"one_step_efficiency({space.name})"
        self._choices: list[PlannerChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def choices(self) -> list[PlannerChoice]:
        return self._choices

    @property
    def trajectory(self) -> list[tuple[StepContext, ControllerAction]]:
        """``(state, action)`` 쌍. Stage 4의 behavior cloning 데이터셋이다."""
        return self._trajectory

    def _score(self, loss_before: float, candidate: Candidate) -> float:
        return efficiency_score(
            loss_before,
            candidate.candidate_loss,
            candidate.cost_ge,
            loss_floor=self._loss_floor,
        )

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        candidates = optimizer.evaluate_candidates(self._space.iter_solve_groups())

        best_index = 0
        best_score = -math.inf
        for i, candidate in enumerate(candidates):
            score = self._score(context.loss, candidate)
            if score > best_score:
                best_score = score
                best_index = i

        finite = [c.candidate_loss for c in candidates if c.is_finite]
        chosen = candidates[best_index]

        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=best_index,
                chosen_score=best_score,
                best_loss=min(finite) if finite else float("nan"),
                worst_loss=max(finite) if finite else float("nan"),
                n_finite=len(finite),
                n_candidates=len(candidates),
                damping_before=context.damping,
                damping_after=chosen.applied_damping,
                n_cg_converged=sum(1 for c in candidates if c.cg.converged),
            )
        )
        self._trajectory.append((context, chosen.action))
        return chosen.action

    def reset(self) -> None:
        self._choices = []
        self._trajectory = []

    def __repr__(self) -> str:
        return (
            f"OneStepEfficiencyController(space={self._space.name}, n_steps={len(self._choices)})"
        )


@dataclass(slots=True)
class _BeamNode:
    """beam search 의 한 노드.

    ``cumulative_cost`` 와 ``loss`` 를 함께 들고 다닌다. 효용은 시퀀스 끝에서
    이 두 값으로 한 번에 계산한다. **step 별 점수를 누적하지 않는다** —
    그것이 초판의 결함이었다.
    """

    first_action: ControllerAction
    cumulative_cost: float
    loss: float
    snapshot: tuple[object, float]
    depth: int = 1
    """이 노드가 대응하는 시퀀스 길이. 어느 depth 의 계획이 채택됐는지 분석용."""


class HorizonPlannerController:
    """``horizon`` step 앞을 beam search 로 보고 첫 action 을 고른다 (MPC).

    프로토콜 게이트 C를 담당한다. 답하는 질문은 하나다.

    > 지금 손해를 감수하고 미래 상태를 개선할 필요가 있는가?

    ```text
    H=1 ~= H=3 ~= H=5  -> 장기 의사결정이 거의 필요 없다.
                          contextual bandit 이나 heuristic 으로 충분하고
                          PPO 의 temporal credit assignment 는 정당화되지 않는다.

    H 증가에 단조 개선  -> 순차적 의사결정이 실제로 이득이다.
                          RL 진행 근거가 확보된다.
    ```

    terminal objective 를 쓴다
    --------------------------
    초판은 step 별 ``Δlog L / cost`` 를 **더했다**. 그러면 국소 효율의 합을
    최대화하는 것이므로 근시안성이 그대로 남는다. 파일럿에서 H=3 이 H=1 을
    개선하지 못한 이유가 이것이다.

    지금은 시퀀스를 **terminal loss 와 누적 비용**으로 평가한다
    (``horizon_utility``). 트랙에 따라 효용이 다르다.

    ```text
    fixed_budget    U = (log L_start - log L_terminal) / cumulative_cost
    cost_to_target  U = -(cumulative_cost + 남은거리 / 관측진행률)
    ```

    ``horizon=1`` 이면 ``fixed_budget`` 트랙에서 one-step efficiency 와 같은
    선택을 한다. 즉 H=1 은 자동으로 그 baseline 을 재현하므로 게이트 C 비교가
    같은 코드 경로에서 이루어진다.

    구현
    ----
    model predictive control 방식이다. 매 실제 step 에서 ``horizon`` 만큼
    앞을 보고 첫 action 만 적용한 뒤 다음 step 에서 다시 계획한다.

    전체 조합은 ``|A|^horizon`` 으로 폭발하므로 beam search 로 상위
    ``beam_width`` 개 궤적만 유지한다. 시뮬레이션은 ``optimizer.simulate_step``
    을 쓰고 ``snapshot`` / ``restore`` 로 상태를 되돌린다. 각 시뮬레이션 step 은
    새 HvpGraph 를 만든다 (파라미터가 바뀌므로).

    비용
    ----
    실제 step 당 시뮬레이션 횟수는 대략 ``|A| + (horizon-1) * beam_width * |A|`` 다.
    ``NARROW.with_fixed_step_size()`` 는 12 action 이므로 horizon 3, beam 4 에서
    step 당 약 108 회다. 전부 ``search_cost_ge`` 로 회계되며 본문 비용에
    섞이지 않는다.

    Args:
        space: 행동 공간. 비교군과 **동일**해야 게이트가 성립한다.
        horizon: 앞을 보는 step 수. 1이면 one-step 과 같은 선택을 한다.
        beam_width: 유지할 궤적 수.
        track: ``fixed_budget`` (Track E) 또는 ``cost_to_target`` (Track T).
        target_loss: ``cost_to_target`` 트랙의 목표 loss. 절대값이다.
        loss_floor: ``log`` 하한.
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        horizon: int = 3,
        beam_width: int = 4,
        track: PlannerTrack = "fixed_budget",
        target_loss: float | None = None,
        loss_floor: float = 1.0e-30,
        name: str | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if track == "cost_to_target" and target_loss is None:
            raise ValueError("cost_to_target track requires target_loss")
        self._space = space
        self._horizon = horizon
        self._beam_width = beam_width
        self._track: PlannerTrack = track
        self._target_loss = target_loss
        self._loss_floor = loss_floor
        self._name = name or f"mpc_H{horizon}_{track}({space.name})"
        self._choices: list[PlannerChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []
        self._last_utility = float("nan")

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def track(self) -> PlannerTrack:
        return self._track

    @property
    def choices(self) -> list[PlannerChoice]:
        return self._choices

    @property
    def trajectory(self) -> list[tuple[StepContext, ControllerAction]]:
        return self._trajectory

    @property
    def last_utility(self) -> float:
        """직전 ``select`` 에서 채택한 계획의 효용.

        구현 불변조건 검증용이다. **incumbent carry-over 덕분에 같은 상태에서
        H 를 늘리면 이 값은 감소할 수 없다.** 실현 성능의 단조성은 보장되지
        않는다 (MPC 는 매 step 재계획하므로). 따라서 테스트는 이 값에 대해서만
        단조성을 주장한다.
        """
        return self._last_utility

    def _utility(self, loss_start: float, node: _BeamNode) -> float:
        return horizon_utility(
            loss_start,
            node.loss,
            node.cumulative_cost,
            track=self._track,
            target_loss=self._target_loss,
            loss_floor=self._loss_floor,
        )

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        root = optimizer.snapshot()
        actions = list(self._space.iter_actions())
        loss_start = context.loss

        # --- depth 0 ---
        beam: list[_BeamNode] = []
        for action in actions:
            optimizer.restore(root)
            loss_after, cost_ge, _ = optimizer.simulate_step(action)
            if not math.isfinite(loss_after):
                continue
            beam.append(
                _BeamNode(
                    first_action=action,
                    cumulative_cost=cost_ge,
                    loss=loss_after,
                    snapshot=optimizer.snapshot(),
                )
            )
        optimizer.restore(root)

        if not beam:
            return self._fallback(context, optimizer, root, actions)

        beam.sort(key=lambda n: self._utility(loss_start, n), reverse=True)
        # incumbent: 지금까지 본 모든 depth 중 최선. depth 확장이 실패해도
        # 이것을 잃지 않는다.
        incumbent = beam[0]
        beam = beam[: self._beam_width]

        # --- depth 1..horizon-1 ---
        for depth in range(2, self._horizon + 1):
            expanded: list[_BeamNode] = []
            for node in beam:
                for action in actions:
                    optimizer.restore(node.snapshot)  # type: ignore[arg-type]
                    loss_after, cost_ge, _ = optimizer.simulate_step(action)
                    if not math.isfinite(loss_after):
                        continue
                    expanded.append(
                        _BeamNode(
                            first_action=node.first_action,
                            cumulative_cost=node.cumulative_cost + cost_ge,
                            loss=loss_after,
                            snapshot=optimizer.snapshot(),
                            depth=depth,
                        )
                    )
            optimizer.restore(root)
            if not expanded:
                break
            # 효용은 시퀀스 끝에서 terminal loss 와 누적 비용으로 계산한다.
            expanded.sort(key=lambda n: self._utility(loss_start, n), reverse=True)
            # **incumbent carry-over.** depth 를 늘렸다고 이전 depth 의 최선을
            # 버리면 안 된다. beam search 는 정확한 planner 가 아니므로 깊은
            # 탐색이 좋은 branch 를 중간에 잘라낼 수 있다. 이 처리가 없으면
            # H 를 늘렸을 때 오히려 나빠질 수 있고, 게이트 C 의 해석이 불가능해진다.
            #
            # 효용은 길이로 정규화되어 있으므로(fixed_budget: gain/cost) 서로 다른
            # 길이의 시퀀스를 비교하는 것이 타당하다.
            if self._utility(loss_start, expanded[0]) > self._utility(loss_start, incumbent):
                incumbent = expanded[0]
            beam = expanded[: self._beam_width]

        optimizer.restore(root)
        best = incumbent
        self._last_utility = self._utility(loss_start, best)

        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=actions.index(best.first_action),
                chosen_score=self._last_utility,
                best_loss=min(n.loss for n in beam),
                worst_loss=max(n.loss for n in beam),
                n_finite=len(beam),
                n_candidates=len(actions),
                damping_before=context.damping,
                chosen_depth=best.depth,
            )
        )
        self._trajectory.append((context, best.first_action))
        return best.first_action

    def _fallback(
        self,
        context: StepContext,
        optimizer: NewtonCGOptimizer,
        root: tuple[object, float],
        actions: Sequence[ControllerAction],
    ) -> ControllerAction:
        """유한한 결과를 내는 action 이 없을 때. 가장 loss 가 낮은 것을 고른다."""
        best_action = actions[0]
        best_loss = math.inf
        for action in actions:
            optimizer.restore(root)  # type: ignore[arg-type]
            loss_after, _, _ = optimizer.simulate_step(action)
            if math.isfinite(loss_after) and loss_after < best_loss:
                best_loss = loss_after
                best_action = action
        optimizer.restore(root)  # type: ignore[arg-type]
        self._last_utility = -math.inf
        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=list(actions).index(best_action),
                chosen_score=-math.inf,
                best_loss=best_loss,
                worst_loss=float("nan"),
                n_finite=0,
                n_candidates=len(actions),
                damping_before=context.damping,
            )
        )
        self._trajectory.append((context, best_action))
        return best_action

    def reset(self) -> None:
        self._choices = []
        self._trajectory = []

    def __repr__(self) -> str:
        return (
            f"HorizonPlannerController(space={self._space.name}, "
            f"horizon={self._horizon}, beam={self._beam_width}, track={self._track})"
        )


def make_open_loop_controller(
    space: ActionSpace, flats: Sequence[int], breakpoints: Sequence[float]
) -> OpenLoopController:
    """스케줄 파라미터로 ``OpenLoopController`` 를 만든다.

    Args:
        space: 행동 공간.
        flats: 각 구간에서 쓸 action 의 flat 인덱스.
        breakpoints: 구간 상한. ``len(flats)`` 와 같아야 하고 오름차순이며
            마지막은 1.0 이상이어야 한다.
    """
    if len(flats) != len(breakpoints):
        raise ValueError(
            f"flats and breakpoints must have equal length, got {len(flats)} and {len(breakpoints)}"
        )
    segments = [
        ScheduleSegment(until=float(b), action=space.action_from_flat(int(f)))
        for f, b in zip(flats, breakpoints, strict=True)
    ]
    label = "-".join(str(int(f)) for f in flats)
    return OpenLoopController(segments, name=f"open_loop[{label}]")
