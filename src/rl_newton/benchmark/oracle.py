"""헤드룸 측정과 게이트 A~D 판정 (프로토콜 Stage 2).

이 프로젝트의 분기점이다. RL 스택을 만들기 전에 "적응 제어에 여지가 있는가",
"행동 공간이 병목인가", "장기 의사결정이 필요한가"를 먼저 답한다.

두 트랙 (프로토콜 D9)
---------------------
초판은 단일 헤드룸을 재려 했고 실패했다. 국소 효율 목적(``Δlog L / cost``)으로
매 step 최선을 고른 컨트롤러가 고정 설정보다 cost-to-target 에서 나빴다
(0.967x). 서로 다른 두 최적화 문제를 섞고 있었기 때문이다.

```text
Track E  max log(L_0 / L_B)  s.t.  Σ c ≤ B      헤드룸 = 차이 [nat]
Track T  min Σ c             s.t.  L ≤ τ        헤드룸 = 비율 [배수]
```

두 트랙이 같은 답을 주지 않을 수 있고, **그 불일치 자체가 결과다.**

게이트
------
```text
A  Track E  absolute planner vs best_static      적응 제어의 내재적 여지
B  Track E  absolute vs wide vs narrow           도달성/행동범위 손실
C  Track E  H=1 vs H=3 vs H=5                    장기 의사결정의 가치 (PPO 착수 조건)
D  Track T  best_static vs planner (target별)     cost-to-target 헤드룸
```

pilot / confirmatory (프로토콜 D6)
----------------------------------
예산과 target을 결과를 본 뒤 고치면 사후 선택이 된다. ``phase`` 로 구분하며,
``pilot`` 은 dev seed 로 예산/target 선정에만 쓰고 ``confirmatory`` 는 held-out
seed 로 최종 판정한다.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from rl_newton.benchmark.metrics import (
    GroupSummary,
    PairedComparison,
    PairedDelta,
    RunSummary,
    TargetSpec,
    compare_paired,
    compare_paired_delta,
    summarize_group,
    summarize_run,
)
from rl_newton.benchmark.paired import SyntheticTask, TaskSpec, make_task
from rl_newton.benchmark.store import ResultStore, RunKey, experiment_id
from rl_newton.optimizers.action_space import ActionSpace
from rl_newton.optimizers.controllers import (
    FixedController,
    HeuristicController,
    HorizonPlannerController,
    OneStepEfficiencyController,
    make_open_loop_controller,
)
from rl_newton.optimizers.newton_cg import (
    Controller,
    NewtonCGConfig,
    NewtonCGOptimizer,
    OptimizationTrace,
)
from rl_newton.types import ControllerAction

__all__ = [
    "HeadroomConfig",
    "GateVerdict",
    "HeadroomReport",
    "Phase",
    "run_controller",
    "search_best_static",
    "search_best_open_loop",
    "run_headroom",
    "spec_kind_label",
]

UTILITY_TOLERANCE = 0.02
"""beam 선택의 상대 효용 허용 오차 (프로토콜 F). **실행 전에 고정된 값이다.**"""

UTILITY_EPSILON = 1.0e-6
"""상대 오차 분모의 하한. ``|J_b - J_4| / max(|J_4|, eps)``.

``J_4`` 가 0 근처면 상대 오차가 폭발한다. 그래서 분모를 ``+ eps`` 가 아니라
``max(|J_4|, eps)`` 로 둔다. 값도 config 에 고정해 사후 조정을 막는다.
"""

DEEP_FRACTION_TOLERANCE = 0.05
"""``chosen_depth > 1`` 비율의 허용 차이 (프로토콜 F).

효용이 비슷해도 깊은 계획을 쓰는 빈도가 달라지면 게이트 C의 해석이 바뀐다.
"크게 다르면 제외"를 사후에 판단하지 않기 위해 수치로 고정한다.
"""

PROTOCOL_VERSION = "stage2-v1"
"""프로토콜 버전. 실험 정체성에 포함된다.

C2 protocol freeze 시 태그와 함께 올린다. confirmatory 중 버그를 발견해
결과를 폐기해야 하면 이 값을 증가시켜 이전 결과와 섞이지 않게 한다.
"""

ControllerFactory = Callable[["SyntheticTask", TargetSpec], Controller]
"""컨트롤러 생성자. task 와 target 을 받는다.

Track T planner 는 절대 target loss 를 알아야 한다. 상대 target ``L/L_0 <= v``
의 절대값은 ``v * L_0`` 이므로 **인스턴스마다 다르다.** 그래서 팩토리가
task 를 받도록 한다.
"""

Phase = Literal["pilot", "confirmatory"]

DEV_SEEDS = (0, 1, 2)
"""pilot 국면의 dev seed. 예산/target 선정에만 쓴다."""

HELD_OUT_SEEDS = tuple(range(100, 110))
"""confirmatory 국면의 held-out seed. 최종 판정에만 쓴다."""


@dataclass(frozen=True, slots=True)
class HeadroomConfig:
    """헤드룸 실험 설정."""

    specs: Sequence[TaskSpec]
    seeds: Sequence[int]
    targets: dict[str, dict[str, TargetSpec]]
    """``targets[spec_kind][difficulty]``. difficulty 는 easy/medium/hard."""
    cost_budget_ge: float = 600.0
    """모든 컨트롤러에 동일하게 주는 GE 예산 (Track E의 B)."""
    max_steps: int = 200
    initial_damping: float = 1.0e-2
    tuning_budget: int | None = None
    """N_tune. ``None`` 이면 narrow 행동 공간 크기를 쓴다. 모든 baseline 동일."""
    horizons: Sequence[int] = (1, 3, 5)
    """게이트 C에서 비교할 planner horizon 들."""
    beam_width: int = 4
    tuning_seed: int = 0
    n_schedule_segments: int = 4
    phase: Phase = "pilot"
    primary_difficulty: str = "medium"
    """게이트 D의 주 target 난이도."""
    device: str = "cpu"
    """텐서 디바이스.

    **Stage 2 는 CPU 가 기본이며 그것이 옳다.** 대상은 quadratic(d=32~100)과
    Rosenbrock(d=2~10)뿐이다. Stage 0 실측에서 10만 파라미터 MNIST MLP 조차
    GPU 런치 오버헤드 지배(0.68 ms/gradient)였으므로, d=100 matvec 을 GPU 로
    보내면 순손실이다. GPU 는 Stage 3 이후에만 쓴다.
    """

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("specs must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if not self.targets:
            raise ValueError("targets must not be empty")

    def optimizer_config(self) -> NewtonCGConfig:
        return NewtonCGConfig(
            total_steps=self.max_steps,
            cost_budget_ge=self.cost_budget_ge,
            initial_damping=self.initial_damping,
        )

    def identity_payload(
        self,
        spaces: Mapping[str, ActionSpace],
        *,
        code_dirty: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """실험 정체성 payload (``store.experiment_id`` 입력).

        **여기에 빠진 항목은 재개 시 낡은 결과를 재사용하게 만든다.**
        beam, horizon, GE 예산, action space 정의(damping 값, CG budget,
        step size), damping 경계, target 정의, protocol 버전을 모두 넣는다.

        ``code_dirty`` 를 포함하는 이유: git commit 만으로는 커밋되지 않은
        변경을 구분할 수 없다.
        """
        optimizer = self.optimizer_config()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "phase": self.phase,
            "device": self.device,
            "cost_budget_ge": self.cost_budget_ge,
            "max_steps": self.max_steps,
            "initial_damping": self.initial_damping,
            "min_damping": optimizer.min_damping,
            "max_damping": optimizer.max_damping,
            "cg_tolerance": optimizer.cg_tolerance,
            "pap_eps": optimizer.pap_eps,
            "max_loss_increase_ratio": optimizer.max_loss_increase_ratio,
            "safe_fallback": optimizer.safe_fallback,
            "compute_trust_ratio": optimizer.compute_trust_ratio,
            "horizons": list(self.horizons),
            "beam_width": self.beam_width,
            "tuning_budget": self.tuning_budget,
            "n_schedule_segments": self.n_schedule_segments,
            "tuning_seed": self.tuning_seed,
            "primary_difficulty": self.primary_difficulty,
            "specs": [str(s) for s in self.specs],
            "seeds": list(self.seeds),
            "targets": {
                kind: {level: spec.label for level, spec in levels.items()}
                for kind, levels in self.targets.items()
            },
            "spaces": {
                name: {
                    "mode": space.damping_mode,
                    "damping_values": [float(v) for v in space.damping_values],
                    "cg_budgets": list(space.cg_budgets),
                    "step_sizes": [float(s) for s in space.step_sizes],
                }
                for name, space in spaces.items()
            },
            "code_dirty": code_dirty,
            **dict(extra or {}),
        }

    def target_for(self, spec: TaskSpec, difficulty: str) -> TargetSpec:
        return self.targets[spec_kind_label(spec)][difficulty]

    def difficulties(self) -> list[str]:
        first = next(iter(self.targets.values()))
        return list(first)


def spec_kind_label(spec: TaskSpec) -> str:
    """target 조회용 spec 분류 키."""
    kind = getattr(spec, "kind", None)
    return str(kind) if kind is not None else "rosenbrock"


def _is_eligible(task: SyntheticTask) -> bool:
    """집계 대상인지. indefinite 는 아래로 유계가 아니라 제외된다."""
    return bool(getattr(task, "is_bounded_below", True))


def absolute_target_loss(task: SyntheticTask, target: TargetSpec) -> float:
    """상대 target 을 절대 loss 값으로 환산한다.

    ``relative_loss`` 는 ``v * L_0``, ``absolute_loss`` 는 그대로다.
    Track T planner 가 cost-to-go 를 추정할 때 필요하다.
    """
    if target.metric == "absolute_loss":
        return target.value
    return target.value * task.initial_loss


def _action_counts(trace: OptimizationTrace) -> dict[str, int]:
    """선택한 action 의 빈도. 정책 분석용 (README §8 action heatmap)."""
    counts: dict[str, int] = {}
    for record in trace.records:
        key = f"m={record.extra.get('damping_multiplier')},k={record.cg_budget},a={record.step_size:g}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _chosen_depths(controller: Controller) -> dict[str, int] | None:
    """planner 가 채택한 계획 길이의 빈도.

    거의 항상 1이면 horizon 을 늘려도 실질적 가치가 없다는 직접적 증거다
    (프로토콜 게이트 C).
    """
    choices = getattr(controller, "choices", None)
    if not choices:
        return None
    counts: dict[str, int] = {}
    for choice in choices:
        depth = getattr(choice, "chosen_depth", 1)
        counts[str(depth)] = counts.get(str(depth), 0) + 1
    return counts


def run_controller(
    config: HeadroomConfig,
    factory: ControllerFactory,
    *,
    label: str,
    exp_id: str,
    difficulty: str | None = None,
    store: ResultStore | None = None,
    verbose: bool = False,
) -> list[RunSummary]:
    """모든 ``(spec, seed)`` 인스턴스에서 컨트롤러를 실행하고 집계한다.

    paired design 이므로 인스턴스는 ``make_task(spec, seed)`` 로 결정론적으로
    만들어진다. 컨트롤러가 난수를 얼마나 쓰든 문제는 동일하다.

    ``store`` 가 주어지면 **재개 가능**하다. 이미 완료된
    ``(controller, task_instance, seed, target)`` 조합은 건너뛰고 저장된 결과를
    쓴다. 각 run 이 끝나는 즉시 기록하므로 프로세스가 끊겨도 손실이 없다.

    예외는 삼키지 않고 ``failed`` 로 기록한 뒤 다음 run 으로 넘어간다. 하나의
    task 가 깨져도 전체 실험이 멈추지 않아야 하고, 실패 사실은 남아야 한다.

    Args:
        difficulty: target 난이도. ``None`` 이면 ``primary_difficulty``.
        store: 재개 가능한 결과 저장소.
        verbose: 건너뛴 run 수를 보고한다.
    """
    opt_config = config.optimizer_config()
    level = difficulty or config.primary_difficulty
    summaries: list[RunSummary] = []
    n_skipped = 0

    for spec in config.specs:
        target = config.target_for(spec, level)
        for seed in config.seeds:
            task = make_task(spec, seed, device=config.device)
            if not _is_eligible(task):
                continue

            key = RunKey(
                experiment_id=exp_id,
                controller=label,
                task_instance_id=task.instance_id,
                seed=seed,
                target=target.label,
            )
            if store is not None and store.is_completed(key):
                cached = store.get(key)
                if cached is not None and cached.summary is not None:
                    summaries.append(cached.summary)
                    n_skipped += 1
                    continue

            controller = factory(task, target)
            started = time.perf_counter()
            try:
                optimizer = NewtonCGOptimizer(
                    task,
                    controller,
                    opt_config,
                    run_id=f"{label}|{task.instance_id}",
                    seed=seed,
                )
                trace = optimizer.run()
            except Exception as exc:  # noqa: BLE001 - 실패를 기록하고 계속한다
                elapsed = time.perf_counter() - started
                message = f"{type(exc).__name__}: {exc}"
                if store is not None:
                    store.record_failure(key, message, wall_clock_sec=elapsed)
                print(f"  실패 {key.as_str()}: {message}", flush=True)
                continue

            elapsed = time.perf_counter() - started
            trace.controller = label
            summary = summarize_run(trace, target)
            summaries.append(summary)
            if store is not None:
                store.record_success(
                    summary,
                    exp_id,
                    wall_clock_sec=elapsed,
                    action_counts=_action_counts(trace),
                    chosen_depths=_chosen_depths(controller),
                )

    if verbose and n_skipped:
        print(f"  {label}: 캐시에서 {n_skipped}개 재사용", flush=True)
    return summaries


def _relabel(group: GroupSummary, name: str) -> GroupSummary:
    """탐색 우승자의 라벨을 정규화한다. 쌍별 비교가 이름으로 조회하기 때문이다."""
    for run in group.runs:
        run.controller = name
    group.controller = name
    return group


def _rank_key_track_e(group: GroupSummary) -> float:
    """Track E 정렬 키. terminal loss 를 가장 많이 줄인 것이 최고다."""
    value = group.median_log_improvement
    return -value if math.isfinite(value) else math.inf


def search_best_static(
    config: HeadroomConfig,
    space: ActionSpace,
    *,
    n_tune: int,
    exp_id: str,
    store: ResultStore | None = None,
) -> tuple[ControllerAction, GroupSummary]:
    """행동 공간 조합을 고정으로 돌려 Track E 기준 최고를 고른다.

    탐색 횟수를 ``n_tune`` 으로 제한한다. 프로토콜 D5의 "모든 컨트롤러에
    동일한 탐색 예산" 원칙을 지키기 위해서다. 행동 공간이 ``n_tune`` 보다
    크면 균등 간격으로 부분집합을 뽑는다.
    """
    total = len(space)
    if total <= n_tune:
        indices = list(range(total))
    else:
        stride = total / n_tune
        indices = sorted({int(i * stride) for i in range(n_tune)})

    best_group: GroupSummary | None = None
    best_flat = indices[0]
    for flat in indices:
        action = space.action_from_flat(flat)
        label = f"static[{flat}]"
        runs = run_controller(
            config,
            lambda _t, _g, a=action: FixedController(a),
            label=label,
            exp_id=exp_id,
            store=store,
        )
        group = summarize_group(runs, controller=label)
        if best_group is None or _rank_key_track_e(group) < _rank_key_track_e(best_group):
            best_group = group
            best_flat = flat

    assert best_group is not None
    return space.action_from_flat(best_flat), best_group


def search_best_open_loop(
    config: HeadroomConfig,
    space: ActionSpace,
    *,
    n_tune: int,
    exp_id: str,
    store: ResultStore | None = None,
) -> GroupSummary:
    """progress 만 보는 스케줄을 랜덤 서치한다 (프로토콜 D4).

    탐색 횟수는 ``best_static`` 과 **동일한** ``n_tune`` 이다. 파일럿에서 이
    예산이 12회였는데 우승자가 static 과 완전히 같은 결과를 냈다. 스케줄
    공간을 사실상 탐색하지 못한 것이므로 ``n_tune`` 을 충분히 크게 준다.
    """
    rng = random.Random(config.tuning_seed)
    n_seg = config.n_schedule_segments

    best_group: GroupSummary | None = None
    for trial in range(n_tune):
        flats = tuple(rng.randrange(len(space)) for _ in range(n_seg))
        cuts = sorted(rng.uniform(0.05, 0.95) for _ in range(n_seg - 1))
        breakpoints = (*cuts, 1.0)
        label = f"open_loop[{trial}]"
        runs = run_controller(
            config,
            lambda _t, _g, f=flats, b=breakpoints: make_open_loop_controller(space, f, b),
            label=label,
            exp_id=exp_id,
            store=store,
        )
        group = summarize_group(runs, controller=label)
        if best_group is None or _rank_key_track_e(group) < _rank_key_track_e(best_group):
            best_group = group

    assert best_group is not None
    return best_group


@dataclass(slots=True)
class GateVerdict:
    """게이트 하나의 판정."""

    name: str
    track: str
    question: str
    statistic: float
    unit: str
    go_threshold: float
    pivot_threshold: float
    detail: str = ""

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.statistic):
            return "판정불가"
        if self.statistic >= self.go_threshold:
            return "GO"
        if self.statistic < self.pivot_threshold:
            return "재설계"
        return "조건부"

    def describe(self) -> str:
        return (
            f"[{self.name}] ({self.track}) {self.question}\n"
            f"    {self.statistic:+.3f} {self.unit}  "
            f"(GO >= {self.go_threshold:g}, 재설계 < {self.pivot_threshold:g})  "
            f"-> {self.verdict}" + (f"\n    {self.detail}" if self.detail else "")
        )


@dataclass(slots=True)
class HeadroomReport:
    """헤드룸 실험 전체 결과."""

    phase: Phase = "pilot"
    groups: dict[str, GroupSummary] = field(default_factory=dict)
    track_e_deltas: list[PairedDelta] = field(default_factory=list)
    track_t_ratios: dict[str, PairedComparison] = field(default_factory=dict)
    """target 난이도 → cost-to-target 비율 비교."""
    gates: list[GateVerdict] = field(default_factory=list)
    best_static_action: ControllerAction | None = None
    tuning_budget: int = 0
    n_instances: int = 0
    tuning_runs: dict[str, int] = field(default_factory=dict)
    """컨트롤러별 실제 사용한 탐색 run 수 (프로토콜 D5 회계)."""
    experiment_id: str = ""
    identity: dict[str, object] = field(default_factory=dict)
    """실험 정체성 payload. 재개 판단과 결과 추적의 기준이다."""

    def summary_table(self) -> str:
        header = (
            f"{'controller':<28} {'logΔ(nat)':>10} {'도달률':>7} "
            f"{'cost→τ':>9} {'총 GE':>8} {'탐색 GE':>10} {'거절':>6} {'CG수렴':>7}"
        )
        lines = [header, "-" * len(header)]
        for name, g in self.groups.items():
            cost = (
                f"{g.median_cost_to_target_ge:.1f}"
                if math.isfinite(g.median_cost_to_target_ge)
                else "미도달"
            )
            lines.append(
                f"{name:<28} {g.median_log_improvement:>10.3f} "
                f"{g.success_rate:>6.0%} {cost:>9} "
                f"{g.median_total_cost_ge:>8.1f} {g.median_search_cost_ge:>10.1f} "
                f"{g.mean_rejection_rate:>6.2f} {g.mean_cg_convergence_rate:>7.2f}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class BeamCalibration:
    """beam width 민감도 측정 결과 (프로토콜 F).

    beam 은 추측으로 정하지 않고 pilot subset 에서 측정해 고른다. beam search 는
    정확한 planner 가 아니므로 폭을 줄이면 계획 품질이 떨어질 수 있고, 그것이
    게이트 C의 결론을 바꿀 수 있다.

    Attributes:
        rows: ``(space, horizon, beam)`` -> 측정값.
        selected_beam: 선택 규칙을 적용한 결과.
        reference_beam: 비교 기준이 된 최대 beam.
        tolerance: 상대 허용 오차.
    """

    rows: dict[tuple[str, int, int], dict[str, float]] = field(default_factory=dict)
    selected_beam: int = 0
    reference_beam: int = 0
    tolerance: float = UTILITY_TOLERANCE
    utility_epsilon: float = UTILITY_EPSILON
    deep_fraction_tolerance: float = DEEP_FRACTION_TOLERANCE
    rationale: str = ""
    rejections: dict[int, str] = field(default_factory=dict)
    """beam 별 배제 사유. 사후 해석이 아니라 사전 규칙의 적용 결과다."""

    def table(self) -> str:
        header = (
            f"{'space':<8} {'H':>3} {'beam':>5} {'logΔ(nat)':>11} "
            f"{'rel.diff':>9} {'depth>1':>8} {'Δdepth':>8} {'wall(s)':>9}"
        )
        lines = [header, "-" * len(header)]
        for (space, horizon, beam), row in sorted(self.rows.items()):
            lines.append(
                f"{space:<8} {horizon:>3} {beam:>5} {row['log_improvement']:>11.4f} "
                f"{row['relative_diff']:>9.4f} {row['deep_fraction']:>8.2f} "
                f"{row.get('deep_diff', float('nan')):>8.3f} "
                f"{row['wall_clock_sec']:>9.2f}"
            )
        return "\n".join(lines)


def calibrate_beam_width(
    config: HeadroomConfig,
    *,
    narrow: ActionSpace,
    wide: ActionSpace,
    beams: Sequence[int] = (1, 2, 4),
    horizons: Sequence[int] = (3, 5),
    tolerance: float = UTILITY_TOLERANCE,
    utility_epsilon: float = UTILITY_EPSILON,
    deep_fraction_tolerance: float = DEEP_FRACTION_TOLERANCE,
    store: ResultStore | None = None,
    code_dirty: bool = False,
    verbose: bool = True,
) -> BeamCalibration:
    """beam width 를 pilot subset 에서 측정해 고른다 (프로토콜 F).

    선택 규칙은 **사전 정의된 것**이며 결과를 본 뒤 바꾸지 않는다.

    ```text
    1. 최대 beam 을 기준으로 삼는다.
    2. 모든 (space, horizon) 조합에서 다음 둘을 모두 만족하는 beam 중
       가장 작은 것을 고른다.
         |J_b - J_ref| / max(|J_ref|, eps) < tolerance          (기본 0.02)
         |d_b - d_ref| <= deep_fraction_tolerance               (기본 0.05)
       여기서 d 는 chosen_depth > 1 비율이다.
    3. 동률이면 wall-clock 이 짧은 것, 그래도 같으면 beam 2.
    ```

    분모가 ``|J_ref| + eps`` 가 아니라 ``max(|J_ref|, eps)`` 인 이유는 ``J_ref``
    가 0 근처일 때 상대 오차가 폭발하기 때문이다. ``eps`` 값도 상수로 고정해
    사후 조정을 막는다.

    ``deep_fraction`` 을 함께 보는 이유: beam 을 줄여 깊은 계획이 사라지면
    효용이 비슷해도 게이트 C의 해석이 달라진다. "크게 다르면 제외"를 결과를
    본 뒤 판단하지 않기 위해 수치로 고정한다.

    Args:
        beams: 시험할 beam 폭.
        horizons: 시험할 horizon.
        tolerance: 효용 상대 허용 오차.
        utility_epsilon: 상대 오차 분모의 하한.
        deep_fraction_tolerance: ``depth > 1`` 비율의 허용 차이.
        store: 재개 가능한 저장소.
        code_dirty: 커밋되지 않은 변경이 있는지. 실험 정체성에 포함된다.

    Returns:
        ``BeamCalibration``.
    """
    calibration = BeamCalibration(
        tolerance=tolerance,
        utility_epsilon=utility_epsilon,
        deep_fraction_tolerance=deep_fraction_tolerance,
    )
    reference = max(beams)
    calibration.reference_beam = reference
    spaces = {"narrow": narrow, "wide": wide}

    measured: dict[tuple[str, int, int], dict[str, float]] = {}
    for space_label, space in spaces.items():
        for horizon in horizons:
            for beam in beams:
                label = f"cal_mpc_H{horizon}_{space_label}_b{beam}"
                if verbose:
                    print(f"  {label}", flush=True)
                # beam 과 horizon 이 실험 정체성에 들어가야 재개가 안전하다.
                exp_id = experiment_id(
                    config.identity_payload(
                        {space_label: space},
                        code_dirty=code_dirty,
                        extra={
                            "mode": "beam_calibration",
                            "cal_beam": beam,
                            "cal_horizon": horizon,
                            "cal_space": space_label,
                        },
                    )
                )
                started = time.perf_counter()
                runs = run_controller(
                    config,
                    lambda _t, _g, s=space, h=horizon, b=beam: HorizonPlannerController(
                        s, horizon=h, beam_width=b, track="fixed_budget"
                    ),
                    label=label,
                    exp_id=exp_id,
                    store=store,
                    verbose=verbose,
                )
                elapsed = time.perf_counter() - started
                group = summarize_group(runs, controller=label)
                deep = float("nan")
                if store is not None:
                    depth_totals: dict[str, int] = {}
                    for record in store:
                        if record.key.controller != label or not record.chosen_depths:
                            continue
                        for depth, count in record.chosen_depths.items():
                            depth_totals[depth] = depth_totals.get(depth, 0) + count
                    total = sum(depth_totals.values())
                    if total:
                        deep = 1.0 - depth_totals.get("1", 0) / total
                measured[space_label, horizon, beam] = {
                    "log_improvement": group.median_log_improvement,
                    "relative_diff": float("nan"),
                    "deep_fraction": deep,
                    "wall_clock_sec": elapsed,
                }

    # 상대 차이와 depth 차이를 채운다.
    # 분모는 ``|J_ref| + eps`` 가 아니라 ``max(|J_ref|, eps)`` 다. J_ref 가 0
    # 근처일 때 상대 오차가 폭발하는 것을 막는다.
    for (space_label, horizon, _beam), row in measured.items():
        ref_row = measured[space_label, horizon, reference]
        ref = ref_row["log_improvement"]
        if math.isfinite(ref) and math.isfinite(row["log_improvement"]):
            row["relative_diff"] = abs(row["log_improvement"] - ref) / max(
                abs(ref), utility_epsilon
            )
        ref_deep = ref_row["deep_fraction"]
        if math.isfinite(ref_deep) and math.isfinite(row["deep_fraction"]):
            row["deep_diff"] = abs(row["deep_fraction"] - ref_deep)
        else:
            row["deep_diff"] = float("nan")
    calibration.rows = measured

    # 선택 규칙 (사전 정의. 결과를 본 뒤 바꾸지 않는다)
    candidates: list[int] = []
    rejections: dict[int, str] = {}
    for beam in sorted(beams):
        reasons: list[str] = []
        for space_label in spaces:
            for horizon in horizons:
                row = measured[space_label, horizon, beam]
                tag = f"{space_label}/H{horizon}"
                if not math.isfinite(row["relative_diff"]):
                    reasons.append(f"{tag}: 효용 비교 불가")
                elif row["relative_diff"] >= tolerance:
                    reasons.append(f"{tag}: 효용 상대차 {row['relative_diff']:.4f}")
                deep_diff = row.get("deep_diff", float("nan"))
                if math.isfinite(deep_diff) and deep_diff > deep_fraction_tolerance:
                    reasons.append(f"{tag}: depth>1 비율 차이 {deep_diff:.3f}")
        if reasons:
            rejections[beam] = "; ".join(reasons[:3])
        else:
            candidates.append(beam)

    calibration.rejections = rejections
    if candidates:
        # tie-break: 최소 beam -> wall-clock 짧은 것 -> beam 2
        selected = min(
            candidates,
            key=lambda b: (
                b,
                sum(measured[s, h, b]["wall_clock_sec"] for s in spaces for h in horizons),
                0 if b == 2 else 1,
            ),
        )
        calibration.selected_beam = selected
        calibration.rationale = (
            f"beam {selected}: 모든 (space, horizon) 에서 기준 beam {reference} 대비 "
            f"효용 상대차 < {tolerance:g} 이고 depth>1 비율 차이 "
            f"<= {deep_fraction_tolerance:g}"
        )
    else:
        calibration.selected_beam = reference
        calibration.rationale = (
            f"어떤 축소 beam 도 기준을 만족하지 못했다 "
            f"(효용 {tolerance:g}, depth {deep_fraction_tolerance:g}). "
            f"기준 beam {reference} 를 유지한다. "
            + " | ".join(f"beam {b}: {why}" for b, why in rejections.items())
        )
    return calibration


def run_headroom(
    config: HeadroomConfig,
    *,
    narrow: ActionSpace,
    wide: ActionSpace,
    absolute: ActionSpace,
    store: ResultStore | None = None,
    code_dirty: bool = False,
    verbose: bool = True,
) -> HeadroomReport:
    """게이트 A~D를 측정한다.

    Args:
        config: 실험 설정.
        narrow: 정책이 실제로 쓸 행동 공간.
        wide: damping 배수를 넓힌 공간.
        absolute: 도달성 제약 없는 분석용 공간. **로그 해상도가 narrow 와
            같아야** 게이트 B의 해석이 성립한다.
        verbose: 진행 상황 출력.
    """
    report = HeadroomReport(phase=config.phase)
    report.n_instances = sum(
        1 for spec in config.specs for seed in config.seeds if _is_eligible(make_task(spec, seed))
    )
    n_tune = config.tuning_budget or len(narrow)
    report.tuning_budget = n_tune

    # 실험 정체성. 설정이 하나라도 다르면 재개 시 별개 run 으로 취급된다.
    identity = config.identity_payload(
        {"narrow": narrow, "wide": wide, "absolute": absolute},
        code_dirty=code_dirty,
        extra={"n_tune": n_tune, "mode": "headroom"},
    )
    exp_id = experiment_id(identity)
    report.experiment_id = exp_id
    report.identity = identity

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(
        f"[{config.phase}] 인스턴스 {report.n_instances}개, "
        f"GE 예산 {config.cost_budget_ge:g}, N_tune {n_tune}"
    )

    # --- baseline ---
    log(f"best_static (탐색 {n_tune}회)")
    best_action, static_group = search_best_static(
        config, narrow, n_tune=n_tune, exp_id=exp_id, store=store
    )
    static_group = _relabel(static_group, "best_static")
    static_runs = static_group.runs
    report.best_static_action = best_action
    report.groups["best_static"] = static_group
    report.tuning_runs["best_static"] = n_tune

    log(f"best_open_loop (탐색 {n_tune}회)")
    open_group = _relabel(
        search_best_open_loop(config, narrow, n_tune=n_tune, exp_id=exp_id, store=store),
        "best_open_loop",
    )
    report.groups["best_open_loop"] = open_group
    report.tuning_runs["best_open_loop"] = n_tune

    log("heuristic")
    heuristic_runs = run_controller(
        config,
        lambda _t, _g: HeuristicController(narrow),
        label="heuristic",
        exp_id=exp_id,
        store=store,
    )
    report.groups["heuristic"] = summarize_group(heuristic_runs, controller="heuristic")
    report.tuning_runs["heuristic"] = 1

    # --- H=1 (one-step efficiency): 행동 공간 3종. 게이트 A1, B ---
    #
    # H=1 에서 planner 의 fixed_budget 효용은 gain/cost 이고, one-step 의
    # efficiency_score 와 같은 식이다. 그런데 one-step 은 HVP 그래프를
    # 후보 전체에 공유하므로 약 10배 싸다. absolute (34 damping x 4 budget =
    # 136 action, sweep 1292 HVP) 를 감당할 수 있는 유일한 경로다.
    onestep_runs: dict[str, list[RunSummary]] = {}
    for space_label, space in (
        ("narrow", narrow),
        ("wide", wide),
        ("absolute", absolute),
    ):
        label = f"onestep_{space_label}"
        log(f"{label} (sweep {space.hvp_per_sweep} HVP)")
        runs = run_controller(
            config,
            lambda _t, _g, s=space: OneStepEfficiencyController(s),
            label=label,
            exp_id=exp_id,
            store=store,
        )
        onestep_runs[label] = runs
        report.groups[label] = summarize_group(runs, controller=label)
        report.tuning_runs[label] = 0

    # --- Track E planner: narrow / wide 만. 게이트 A2, C ---
    #
    # absolute 는 여기서 제외한다. 현재 damping 과 무관하게 순간 이동하므로
    # damping ramp-up 과 temporal credit assignment 를 **제거해 버린다.**
    # 장기 계획의 필요성을 묻는 게이트 C에 넣을 이유가 없고, 비용도
    # 감당할 수 없다 (실제 step 당 약 1,200회 시뮬레이션).
    planner_runs: dict[str, list[RunSummary]] = {}
    for space_label, space in (("narrow", narrow), ("wide", wide)):
        for horizon in config.horizons:
            label = f"mpc_H{horizon}_{space_label}"
            log(f"{label} ({len(space)} actions, beam {config.beam_width})")
            runs = run_controller(
                config,
                lambda _t, _g, s=space, h=horizon: HorizonPlannerController(
                    s,
                    horizon=h,
                    beam_width=config.beam_width,
                    track="fixed_budget",
                ),
                label=label,
                exp_id=exp_id,
                store=store,
            )
            planner_runs[label] = runs
            report.groups[label] = summarize_group(runs, controller=label)
            report.tuning_runs[label] = 0

    # --- Track E 쌍별 차이 ---
    def delta(base: Sequence[RunSummary], treat: Sequence[RunSummary]) -> PairedDelta:
        return compare_paired_delta(base, treat, metric="log_improvement")

    max_h = max(config.horizons)
    min_h = min(config.horizons)
    e_pairs = [
        # 게이트 A1: 순간적 absolute headroom (도달성 제약 제거, H=1)
        (static_runs, onestep_runs["onestep_absolute"]),
        # 게이트 A2: 도달 가능한 sequential headroom
        (static_runs, planner_runs[f"mpc_H{max_h}_narrow"]),
        (static_runs, planner_runs[f"mpc_H{max_h}_wide"]),
        # 게이트 B: action-space restriction (모두 H=1, 같은 조건)
        (onestep_runs["onestep_narrow"], onestep_runs["onestep_absolute"]),
        (onestep_runs["onestep_narrow"], onestep_runs["onestep_wide"]),
        # 게이트 C: temporal planning value (absolute 제외)
        (planner_runs[f"mpc_H{min_h}_narrow"], planner_runs[f"mpc_H{max_h}_narrow"]),
        (planner_runs[f"mpc_H{min_h}_wide"], planner_runs[f"mpc_H{max_h}_wide"]),
        # 참고 baseline
        (static_runs, open_group.runs),
        (static_runs, heuristic_runs),
    ]
    report.track_e_deltas = [delta(b, t) for b, t in e_pairs]

    # --- Track T: target 난이도별 cost-to-target ---
    log("Track T: target 난이도별 재집계")
    for level in config.difficulties():
        static_t = run_controller(
            config,
            lambda _t, _g, a=best_action: FixedController(a),
            label=f"best_static@{level}",
            difficulty=level,
            exp_id=exp_id,
            store=store,
        )
        # planner 는 task 별 **절대** target loss 를 받아야 한다. 상대 target
        # v 의 절대값은 v * L_0 이므로 인스턴스마다 다르다. 고정값을 넘기면
        # cost-to-go 추정이 무의미해진다.
        planner_t = run_controller(
            config,
            lambda task, target, s=narrow: HorizonPlannerController(
                s,
                horizon=max_h,
                beam_width=config.beam_width,
                track="cost_to_target",
                target_loss=absolute_target_loss(task, target),
            ),
            label=f"mpc_H{max_h}@{level}",
            difficulty=level,
            exp_id=exp_id,
            store=store,
        )
        report.groups[f"best_static@{level}"] = summarize_group(
            static_t, controller=f"best_static@{level}"
        )
        report.groups[f"mpc_H{max_h}@{level}"] = summarize_group(
            planner_t, controller=f"mpc_H{max_h}@{level}"
        )
        report.track_t_ratios[level] = compare_paired(
            static_t, planner_t, metric="cost_to_target_ge"
        )

    # --- 게이트 판정 ---
    by_e = {(d.baseline, d.treatment): d for d in report.track_e_deltas}

    def e_delta(base: str, treat: str) -> float:
        d = by_e.get((base, treat))
        return d.median_delta if d else float("nan")

    gate_a1 = e_delta("best_static", "onestep_absolute")
    report.gates.append(
        GateVerdict(
            name="A1",
            track="Track E",
            question=(
                "현재 상태에서 좋은 damping 이 존재하는가 "
                "(instantaneous absolute-action headroom, H=1)"
            ),
            statistic=gate_a1,
            unit="nat",
            go_threshold=1.0,
            pivot_threshold=0.3,
            detail=(
                (f"loss {math.exp(gate_a1):.2f}배 차이. " if math.isfinite(gate_a1) else "")
                + "도달성 제약을 완전히 없앤 **순간적** 이득이다. "
                "전역 상한이나 장기 헤드룸이 아니다."
            ),
        )
    )

    gate_a2_narrow = e_delta("best_static", f"mpc_H{max_h}_narrow")
    gate_a2_wide = e_delta("best_static", f"mpc_H{max_h}_wide")
    gate_a2 = (
        max(v for v in (gate_a2_narrow, gate_a2_wide) if math.isfinite(v))
        if any(math.isfinite(v) for v in (gate_a2_narrow, gate_a2_wide))
        else float("nan")
    )
    report.gates.append(
        GateVerdict(
            name="A2",
            track="Track E",
            question=(
                "현실적인 multiplier action 으로 그 이득에 접근할 수 있는가 "
                f"(narrow/wide H{max_h} vs best_static)"
            ),
            statistic=gate_a2,
            unit="nat",
            go_threshold=0.7,
            pivot_threshold=0.2,
            detail=(
                f"narrow {gate_a2_narrow:+.3f}, wide {gate_a2_wide:+.3f} nat. "
                "A1 대비 크게 낮으면 행동 공간 도달성이 병목이다."
            ),
        )
    )

    gate_b = e_delta("onestep_narrow", "onestep_absolute")
    wide_gain = e_delta("onestep_narrow", "onestep_wide")
    report.gates.append(
        GateVerdict(
            name="B",
            track="Track E",
            question="행동 공간이 병목인가 (absolute vs narrow, 모두 H=1, 해상도 정렬)",
            statistic=gate_b,
            unit="nat",
            go_threshold=0.5,
            pivot_threshold=0.1,
            detail=f"참고: wide - narrow = {wide_gain:+.3f} nat",
        )
    )

    gate_c_narrow = e_delta(f"mpc_H{min_h}_narrow", f"mpc_H{max_h}_narrow")
    gate_c_wide = e_delta(f"mpc_H{min_h}_wide", f"mpc_H{max_h}_wide")
    gate_c = (
        max(v for v in (gate_c_narrow, gate_c_wide) if math.isfinite(v))
        if any(math.isfinite(v) for v in (gate_c_narrow, gate_c_wide))
        else float("nan")
    )
    curves = []
    for label in ("narrow", "wide"):
        points = " → ".join(
            f"H{h}:{report.groups[f'mpc_H{h}_{label}'].median_log_improvement:.3f}"
            for h in config.horizons
        )
        curves.append(f"{label} {points}")
    report.gates.append(
        GateVerdict(
            name="C",
            track="Track E",
            question=f"그 접근에 여러 step 의 planning 이 필요한가 (H{max_h} vs H{min_h})",
            statistic=gate_c,
            unit="nat",
            go_threshold=0.3,
            pivot_threshold=0.05,
            detail=(
                f"horizon 곡선: {' | '.join(curves)}. "
                "beam search 는 정확한 planner 가 아니므로 실현 성능의 단조성은 "
                "보장되지 않는다. incumbent carry-over 로 planner 효용의 단조성만 "
                "보장된다. 재설계 판정이면 contextual bandit / heuristic 으로 "
                "충분하며 PPO 를 시작하지 않는다 (프로토콜 PPO 착수 조건 1)."
            ),
        )
    )

    primary = report.track_t_ratios.get(config.primary_difficulty)
    gate_d = primary.ratio_geometric_mean if primary else float("nan")
    all_levels = ", ".join(
        f"{level}:{c.ratio_geometric_mean:.3f}x(도달 {c.n_both_reached}/{c.n_pairs})"
        for level, c in report.track_t_ratios.items()
    )
    report.gates.append(
        GateVerdict(
            name="D",
            track="Track T",
            question=(
                f"cost-to-target 헤드룸이 있는가 "
                f"(planner vs best_static, {config.primary_difficulty})"
            ),
            statistic=gate_d,
            unit="배",
            go_threshold=1.20,
            pivot_threshold=1.05,
            detail=(
                f"난이도별 {all_levels}. "
                "게이트 A와 결론이 다르면 그 불일치 자체를 결과로 보고한다 "
                "(프로토콜 D9)."
            ),
        )
    )
    return report
