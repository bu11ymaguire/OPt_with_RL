"""action 을 고르는 컨트롤러들.

모두 같은 ``Controller`` 프로토콜을 구현하고 같은 ``ActionSpace`` 에서 고른다.
차이는 **무엇을 보고 고르는가** 뿐이다.

```text
FixedController          아무것도 안 본다. 항상 같은 action
OpenLoopController       progress 만 본다 (step / total_steps)
HeuristicController      trust ratio 를 본다
GreedyOracleController   현재 step 결과를 본다 (1-step 전수 시도)
LookaheadOracleController 3-step 앞을 본다 (beam search)
```

이 계층이 프로토콜 게이트 A~C를 구성한다
-----------------------------------------
```text
게이트 A  absolute greedy oracle vs best_static
          -> 내재적 one-step 헤드룸. 작으면 적응 제어 연구를 접는다.

게이트 B  absolute vs wide vs narrow greedy oracle
          -> 도달성/행동범위 손실. 크면 행동 공간을 고친다.

게이트 C  greedy oracle vs 3-step lookahead oracle
          -> 장기 의사결정의 필요성. 차이가 작으면 contextual bandit 이나
             heuristic 으로 충분하고, PPO 의 temporal credit assignment 는
             정당화되지 않는다.
```

``OpenLoopController`` 는 별도로 결정적이다 (프로토콜 D4). RL 이 fixed 는
이기고 open_loop 는 못 이기면, 학습된 것은 **적응 제어가 아니라 스케줄**이다.

오라클은 엄밀한 상한이 아니다
-----------------------------
greedy 는 국소 최선이 궤적 전체로는 최적이 아니고, lookahead 도 유한 horizon 과
beam 폭에 제한된다. 따라서 이들은 **하한이 있는 대리 지표**다. 그러나 오라클이
best_static 과 비슷하다면 상태 기반 제어의 여지가 작다는 강한 신호다. 게이트
판정에 이 해석을 쓴다. 문서에는 "one-step 헤드룸" / "3-step 헤드룸" 으로
정확히 표기한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
    "GreedyOracleController",
    "LookaheadOracleController",
    "ScheduleSegment",
    "OracleChoice",
    "make_open_loop_controller",
    "efficiency_score",
]


def efficiency_score(
    loss_before: float, loss_after: float, cost_ge: float, *, loss_floor: float = 1.0e-30
) -> float:
    """``(log L_before - log L_after) / cost_GE``. 비용 대비 진행도.

    프로토콜 D3의 보상과 같은 형태다. 오라클이 최적화하는 목적과 정책이 학습할
    목적이 일치해야 헤드룸 수치가 의미를 갖는다.

    **CG 수렴 여부가 아니라 objective 감소를 본다.** 높은 damping 은
    ``(H + lambda I)^{-1} g ~ g / lambda`` 로 CG 를 쉽게 만들지만 실제 감소는
    느려질 수 있다. 오라클이 CG 성공률을 최대화하면 잘못된 것을 최적화한다.

    개선이 없으면 ``-inf`` 를 반환한다. optimizer 가 어차피 거절할 후보다.
    """
    if not math.isfinite(loss_after) or loss_after >= loss_before:
        return -math.inf
    before = max(loss_before, loss_floor)
    after = max(loss_after, loss_floor)
    return (math.log(before) - math.log(after)) / max(cost_ge, 1e-12)


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

    def select(
        self, context: StepContext, optimizer: NewtonCGOptimizer
    ) -> ControllerAction:
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

    def __init__(
        self, segments: Sequence[ScheduleSegment], *, name: str = "open_loop"
    ) -> None:
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

    def select(
        self, context: StepContext, optimizer: NewtonCGOptimizer
    ) -> ControllerAction:
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

    def select(
        self, context: StepContext, optimizer: NewtonCGOptimizer
    ) -> ControllerAction:
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
class OracleChoice:
    """오라클이 한 step 에서 내린 선택의 기록.

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


class GreedyOracleController:
    """매 step 전수 시도 후 ``Δlog L / cost_GE`` 가 최대인 action 을 고른다.

    미래를 보지 못하는 **1-step** 오라클이다. 따라서 이것으로 측정되는 것은
    정확히 "one-step 헤드룸" 이다. 장기 의사결정의 필요성은
    ``LookaheadOracleController`` 로 따로 본다.

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
    (프로토콜 D5). 섞으면 헤드룸이 과소평가된다.
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
        self._name = name or f"greedy_oracle({space.name})"
        self._choices: list[OracleChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def choices(self) -> list[OracleChoice]:
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

    def select(
        self, context: StepContext, optimizer: NewtonCGOptimizer
    ) -> ControllerAction:
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
            OracleChoice(
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
            f"GreedyOracleController(space={self._space.name}, "
            f"n_steps={len(self._choices)})"
        )


@dataclass(slots=True)
class _BeamNode:
    """beam search 의 한 노드."""

    first_action: ControllerAction
    cumulative_score: float
    snapshot: tuple[object, float]
    alive: bool = True


class LookaheadOracleController:
    """``horizon`` step 앞을 beam search 로 보고 첫 action 을 고른다.

    프로토콜 게이트 C를 담당한다. 답하는 질문은 하나다.

    > 지금 손해를 감수하고 미래 상태를 개선할 필요가 있는가?

    ```text
    greedy ~= lookahead  -> 장기 의사결정이 거의 필요 없다.
                            contextual bandit 이나 heuristic 으로 충분하고
                            PPO 의 temporal credit assignment 는 정당화되지 않는다.

    lookahead >> greedy  -> 순차적 의사결정이 실제로 이득이다.
                            RL 진행 근거가 확보된다.
    ```

    구현
    ----
    model predictive control 방식이다. 매 실제 step 에서 ``horizon`` 만큼
    앞을 보고 첫 action 만 적용한 뒤, 다음 step 에서 다시 계획한다.

    전체 조합은 ``|A|^horizon`` 으로 폭발하므로 beam search 로 상위
    ``beam_width`` 개 궤적만 유지한다. 시뮬레이션은
    ``optimizer.simulate_step`` 을 쓰고 ``snapshot`` / ``restore`` 로 상태를
    되돌린다. 각 시뮬레이션 step 은 새 HvpGraph 를 만든다 (파라미터가 바뀌므로).

    비용
    ----
    실제 step 당 시뮬레이션 횟수는 대략 ``|A| + (horizon-1) * beam_width * |A|`` 다.
    행동 공간을 줄여 쓰는 것을 권한다. 예: ``NARROW.with_fixed_step_size()`` 는
    12 action 이므로 horizon 3, beam 4 에서 step 당 약 108 회 시뮬레이션이다.

    이 비용은 전부 ``search_cost_ge`` 로 회계되며 본문 비용에 섞이지 않는다.

    Args:
        space: 행동 공간. greedy 와 **동일**해야 비교가 성립한다.
        horizon: 앞을 보는 step 수. 1이면 greedy 와 같다.
        beam_width: 유지할 궤적 수.
        loss_floor: ``log`` 하한.
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        horizon: int = 3,
        beam_width: int = 4,
        loss_floor: float = 1.0e-30,
        name: str | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        self._space = space
        self._horizon = horizon
        self._beam_width = beam_width
        self._loss_floor = loss_floor
        self._name = name or f"lookahead{horizon}_oracle({space.name})"
        self._choices: list[OracleChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []

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
    def choices(self) -> list[OracleChoice]:
        return self._choices

    @property
    def trajectory(self) -> list[tuple[StepContext, ControllerAction]]:
        return self._trajectory

    def select(
        self, context: StepContext, optimizer: NewtonCGOptimizer
    ) -> ControllerAction:
        root = optimizer.snapshot()
        actions = list(self._space.iter_actions())

        # --- depth 0: 모든 action 을 시뮬레이션 ---
        beam: list[_BeamNode] = []
        for action in actions:
            optimizer.restore(root)
            loss_after, cost_ge, _ = optimizer.simulate_step(action)
            score = efficiency_score(
                context.loss, loss_after, cost_ge, loss_floor=self._loss_floor
            )
            if not math.isfinite(score):
                continue
            beam.append(
                _BeamNode(
                    first_action=action,
                    cumulative_score=score,
                    snapshot=optimizer.snapshot(),
                )
            )

        optimizer.restore(root)

        if not beam:
            # 어떤 action 도 개선하지 못했다. greedy 와 동일하게 처리한다.
            fallback = self._fallback(context, optimizer, root, actions)
            return fallback

        beam.sort(key=lambda n: n.cumulative_score, reverse=True)
        beam = beam[: self._beam_width]

        # --- depth 1..horizon-1: beam 확장 ---
        for _ in range(self._horizon - 1):
            expanded: list[_BeamNode] = []
            for node in beam:
                optimizer.restore(node.snapshot)  # type: ignore[arg-type]
                loss_here = float(optimizer.task.loss().detach())
                if not math.isfinite(loss_here):
                    continue
                node_snapshot = optimizer.snapshot()
                for action in actions:
                    optimizer.restore(node_snapshot)
                    loss_after, cost_ge, _ = optimizer.simulate_step(action)
                    score = efficiency_score(
                        loss_here, loss_after, cost_ge, loss_floor=self._loss_floor
                    )
                    if not math.isfinite(score):
                        continue
                    expanded.append(
                        _BeamNode(
                            first_action=node.first_action,
                            cumulative_score=node.cumulative_score + score,
                            snapshot=optimizer.snapshot(),
                        )
                    )
            optimizer.restore(root)
            if not expanded:
                break
            expanded.sort(key=lambda n: n.cumulative_score, reverse=True)
            beam = expanded[: self._beam_width]

        optimizer.restore(root)
        best = max(beam, key=lambda n: n.cumulative_score)

        self._choices.append(
            OracleChoice(
                step=context.step,
                chosen_flat=actions.index(best.first_action),
                chosen_score=best.cumulative_score,
                best_loss=float("nan"),
                worst_loss=float("nan"),
                n_finite=len(beam),
                n_candidates=len(actions),
                damping_before=context.damping,
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
        """개선 가능한 action 이 없을 때. 가장 loss 를 덜 늘리는 것을 고른다."""
        best_action = actions[0]
        best_loss = math.inf
        for action in actions:
            optimizer.restore(root)  # type: ignore[arg-type]
            loss_after, _, _ = optimizer.simulate_step(action)
            if math.isfinite(loss_after) and loss_after < best_loss:
                best_loss = loss_after
                best_action = action
        optimizer.restore(root)  # type: ignore[arg-type]
        self._choices.append(
            OracleChoice(
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
            f"LookaheadOracleController(space={self._space.name}, "
            f"horizon={self._horizon}, beam={self._beam_width})"
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
            f"flats and breakpoints must have equal length, "
            f"got {len(flats)} and {len(breakpoints)}"
        )
    segments = [
        ScheduleSegment(until=float(b), action=space.action_from_flat(int(f)))
        for f, b in zip(flats, breakpoints, strict=True)
    ]
    label = "-".join(str(int(f)) for f in flats)
    return OpenLoopController(segments, name=f"open_loop[{label}]")
