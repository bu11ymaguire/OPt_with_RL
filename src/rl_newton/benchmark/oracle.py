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
from collections.abc import Callable, Sequence
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


def run_controller(
    config: HeadroomConfig,
    factory: ControllerFactory,
    *,
    label: str,
    difficulty: str | None = None,
) -> list[RunSummary]:
    """모든 ``(spec, seed)`` 인스턴스에서 컨트롤러를 실행하고 집계한다.

    paired design 이므로 인스턴스는 ``make_task(spec, seed)`` 로 결정론적으로
    만들어진다. 컨트롤러가 난수를 얼마나 쓰든 문제는 동일하다.

    Args:
        difficulty: target 난이도. ``None`` 이면 ``primary_difficulty``.
    """
    opt_config = config.optimizer_config()
    level = difficulty or config.primary_difficulty
    summaries: list[RunSummary] = []
    for spec in config.specs:
        target = config.target_for(spec, level)
        for seed in config.seeds:
            task = make_task(spec, seed)
            if not _is_eligible(task):
                continue
            optimizer = NewtonCGOptimizer(
                task,
                factory(task, target),
                opt_config,
                run_id=f"{label}|{task.instance_id}",
                seed=seed,
            )
            trace = optimizer.run()
            trace.controller = label
            summaries.append(summarize_run(trace, target))
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
    config: HeadroomConfig, space: ActionSpace, *, n_tune: int
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
        runs = run_controller(config, lambda _t, _g, a=action: FixedController(a), label=label)
        group = summarize_group(runs, controller=label)
        if best_group is None or _rank_key_track_e(group) < _rank_key_track_e(best_group):
            best_group = group
            best_flat = flat

    assert best_group is not None
    return space.action_from_flat(best_flat), best_group


def search_best_open_loop(
    config: HeadroomConfig, space: ActionSpace, *, n_tune: int
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
            f"-> {self.verdict}"
            + (f"\n    {self.detail}" if self.detail else "")
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


def run_headroom(
    config: HeadroomConfig,
    *,
    narrow: ActionSpace,
    wide: ActionSpace,
    absolute: ActionSpace,
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
        1
        for spec in config.specs
        for seed in config.seeds
        if _is_eligible(make_task(spec, seed))
    )
    n_tune = config.tuning_budget or len(narrow)
    report.tuning_budget = n_tune

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(
        f"[{config.phase}] 인스턴스 {report.n_instances}개, "
        f"GE 예산 {config.cost_budget_ge:g}, N_tune {n_tune}"
    )

    # --- baseline ---
    log(f"best_static (탐색 {n_tune}회)")
    best_action, static_group = search_best_static(config, narrow, n_tune=n_tune)
    static_group = _relabel(static_group, "best_static")
    static_runs = static_group.runs
    report.best_static_action = best_action
    report.groups["best_static"] = static_group
    report.tuning_runs["best_static"] = n_tune

    log(f"best_open_loop (탐색 {n_tune}회)")
    open_group = _relabel(
        search_best_open_loop(config, narrow, n_tune=n_tune), "best_open_loop"
    )
    report.groups["best_open_loop"] = open_group
    report.tuning_runs["best_open_loop"] = n_tune

    log("heuristic")
    heuristic_runs = run_controller(
        config, lambda _t, _g: HeuristicController(narrow), label="heuristic"
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
    gate_a2 = max(
        v for v in (gate_a2_narrow, gate_a2_wide) if math.isfinite(v)
    ) if any(math.isfinite(v) for v in (gate_a2_narrow, gate_a2_wide)) else float("nan")
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
    gate_c = max(
        v for v in (gate_c_narrow, gate_c_wide) if math.isfinite(v)
    ) if any(math.isfinite(v) for v in (gate_c_narrow, gate_c_wide)) else float("nan")
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
