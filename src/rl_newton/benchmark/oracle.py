"""헤드룸 측정과 게이트 A~C 판정 (프로토콜 Stage 2).

이 프로젝트의 분기점이다. RL 스택을 만들기 전에 "적응 제어에 여지가 있는가",
"행동 공간이 병목인가", "장기 의사결정이 필요한가"를 먼저 답한다.

```text
게이트 A  absolute greedy oracle vs best_static
          내재적 one-step 헤드룸. 작으면 적응 제어 연구를 접거나
          음성 결과로 정리한다.

게이트 B  absolute vs wide vs narrow greedy oracle
          도달성/행동범위 손실. 크면 행동 공간을 고친다.
          absolute 는 해상도를 narrow 수준으로 맞춘 것이어야 해석이 성립한다.

게이트 C  greedy oracle vs lookahead oracle (동일 행동 공간)
          장기 의사결정의 필요성. 차이가 작으면 contextual bandit 이나
          heuristic 으로 충분하고 PPO 는 정당화되지 않는다.
```

비교는 GE 예산 기준이다
-----------------------
step 수를 맞추고 최종 loss 를 비교하면 안 된다. Newton-CG 는 action 에 따라
step 비용이 6배 이상 달라지므로 (k=3 vs k=20) 비용이 다른 것들을 비교하게
된다 (README §4.2). 모든 컨트롤러에 **동일한 GE 예산**을 주고 cost-to-target
과 최종 loss 를 함께 본다.

step_size 는 기본적으로 고정한다
--------------------------------
damping 이 큰 구간에서 ``(H + lambda I)^{-1} g ~ g / lambda`` 이므로 update 가
``-(alpha/lambda) g`` 로 근사되고, ``(lambda, alpha)`` 와 ``(10 lambda, 10 alpha)``
가 거의 같은 결과를 낸다. 이 action aliasing 이 있으면 어떤 축이 이득을
만들었는지 귀속시킬 수 없다. 그래서 기본 조건은 step_size 고정이고, 제어를
추가한 조건은 부가로 돌려 기여를 분리한다.

indefinite task 는 제외된다
---------------------------
아래로 유계가 아니므로 target 도달 개념과 log 보상이 정의되지 않는다.
negative curvature 경로 검증에만 쓴다 (``QuadraticTask.is_bounded_below``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rl_newton.benchmark.metrics import (
    GroupSummary,
    PairedComparison,
    RunSummary,
    TargetSpec,
    compare_paired,
    summarize_group,
    summarize_run,
)
from rl_newton.benchmark.paired import SyntheticTask, TaskSpec, make_task
from rl_newton.optimizers.action_space import ActionSpace
from rl_newton.optimizers.controllers import (
    FixedController,
    GreedyOracleController,
    HeuristicController,
    LookaheadOracleController,
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
    "run_controller",
    "search_best_static",
    "search_best_open_loop",
    "run_headroom",
]

ControllerFactory = Callable[[], Controller]


@dataclass(frozen=True, slots=True)
class HeadroomConfig:
    """헤드룸 실험 설정."""

    specs: Sequence[TaskSpec]
    seeds: Sequence[int]
    targets: dict[str, TargetSpec]
    """spec 종류별 사전 등록 target. 키는 ``spec_kind_label`` 의 결과."""
    cost_budget_ge: float = 600.0
    """모든 컨트롤러에 동일하게 주는 GE 예산."""
    max_steps: int = 200
    """step 수 상한. 예산이 실질적 종료 조건이므로 넉넉하게 둔다."""
    initial_damping: float = 1.0e-2
    lookahead_horizon: int = 3
    lookahead_beam: int = 4
    tuning_seed: int = 0
    """open_loop 랜덤 서치 시드."""
    n_schedule_segments: int = 3

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("specs must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")

    def optimizer_config(self) -> NewtonCGConfig:
        return NewtonCGConfig(
            total_steps=self.max_steps,
            cost_budget_ge=self.cost_budget_ge,
            initial_damping=self.initial_damping,
        )


def spec_kind_label(spec: TaskSpec) -> str:
    """target 조회용 spec 분류 키."""
    kind = getattr(spec, "kind", None)
    if kind is not None:
        return str(kind)
    return "rosenbrock"


def _is_eligible(task: SyntheticTask) -> bool:
    """cost-to-target 집계 대상인지. indefinite 는 제외된다."""
    return bool(getattr(task, "is_bounded_below", True))


def run_controller(
    config: HeadroomConfig,
    factory: ControllerFactory,
    *,
    label: str,
    include_ineligible: bool = False,
) -> list[RunSummary]:
    """모든 ``(spec, seed)`` 인스턴스에서 컨트롤러를 실행하고 집계한다.

    paired design 이므로 인스턴스는 ``make_task(spec, seed)`` 로 결정론적으로
    만들어진다. 컨트롤러가 난수를 얼마나 쓰든 문제는 동일하다.
    """
    opt_config = config.optimizer_config()
    summaries: list[RunSummary] = []
    for spec in config.specs:
        target = config.targets[spec_kind_label(spec)]
        for seed in config.seeds:
            task = make_task(spec, seed)
            if not include_ineligible and not _is_eligible(task):
                continue
            controller = factory()
            optimizer = NewtonCGOptimizer(
                task,
                controller,
                opt_config,
                run_id=f"{label}|{task.instance_id}",
                seed=seed,
            )
            trace = optimizer.run()
            trace.controller = label
            summaries.append(summarize_run(trace, target))
    return summaries


def _rank_key(group: GroupSummary) -> tuple[float, float, float]:
    """정렬 키. 도달률 우선, 그다음 cost-to-target, 그다음 최종 loss.

    도달률을 먼저 보는 것이 프로토콜 D6의 절단 규칙과 일치한다. 절반이
    발산하지만 성공한 절반이 빠른 설정을 최고로 뽑지 않는다.
    """
    cost = (
        group.median_cost_to_target_ge
        if math.isfinite(group.median_cost_to_target_ge)
        else math.inf
    )
    ratio = (
        group.median_final_loss_ratio
        if math.isfinite(group.median_final_loss_ratio)
        else math.inf
    )
    return (-group.success_rate, cost, ratio)


def search_best_static(
    config: HeadroomConfig, space: ActionSpace
) -> tuple[ControllerAction, GroupSummary, list[GroupSummary]]:
    """행동 공간의 모든 조합을 고정으로 돌려 최고를 고른다.

    프로토콜 D5의 "동일 탐색 예산" 원칙을 정직하게 구현한 것이다. 탐색 횟수는
    ``len(space)`` 이며, open_loop 랜덤 서치에도 같은 횟수를 준다.

    Returns:
        ``(최고 action, 그 group summary, 전체 group summary 목록)``.
    """
    all_groups: list[GroupSummary] = []
    for flat in range(len(space)):
        action = space.action_from_flat(flat)
        label = f"static[{flat}]"
        runs = run_controller(
            config, lambda a=action: FixedController(a), label=label
        )
        all_groups.append(summarize_group(runs, controller=label))

    best_index = min(range(len(all_groups)), key=lambda i: _rank_key(all_groups[i]))
    best_action = space.action_from_flat(best_index)
    return best_action, all_groups[best_index], all_groups


def search_best_open_loop(
    config: HeadroomConfig, space: ActionSpace, *, n_trials: int | None = None
) -> tuple[Controller, GroupSummary, list[GroupSummary]]:
    """progress 만 보는 스케줄을 랜덤 서치한다 (프로토콜 D4).

    RL 이 fixed 는 이기고 이것은 못 이기면, 학습된 것은 적응 제어가 아니라
    스케줄이다. 탐색 예산은 ``best_static`` 과 동일하게 ``len(space)`` 다.
    """
    trials = n_trials if n_trials is not None else len(space)
    rng = random.Random(config.tuning_seed)
    n_seg = config.n_schedule_segments

    all_groups: list[GroupSummary] = []
    factories: list[ControllerFactory] = []
    for trial in range(trials):
        flats = [rng.randrange(len(space)) for _ in range(n_seg)]
        cuts = sorted(rng.uniform(0.1, 0.9) for _ in range(n_seg - 1))
        breakpoints = [*cuts, 1.0]
        label = f"open_loop[{trial}]"

        def factory(f=tuple(flats), b=tuple(breakpoints)) -> Controller:
            return make_open_loop_controller(space, f, b)

        factories.append(factory)
        runs = run_controller(config, factory, label=label)
        all_groups.append(summarize_group(runs, controller=label))

    best_index = min(range(len(all_groups)), key=lambda i: _rank_key(all_groups[i]))
    return factories[best_index](), all_groups[best_index], all_groups


@dataclass(slots=True)
class GateVerdict:
    """게이트 하나의 판정."""

    name: str
    question: str
    comparison: PairedComparison | None
    ratio: float
    go_threshold: float
    pivot_threshold: float
    note: str = ""

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.ratio):
            return "판정불가"
        if self.ratio >= self.go_threshold:
            return "GO"
        if self.ratio < self.pivot_threshold:
            return "재설계"
        return "조건부"

    def describe(self) -> str:
        return (
            f"[{self.name}] {self.question}\n"
            f"    비율 {self.ratio:.3f}x  "
            f"(GO >= {self.go_threshold:.2f}, 재설계 < {self.pivot_threshold:.2f})  "
            f"-> {self.verdict}"
            + (f"\n    {self.note}" if self.note else "")
        )


@dataclass(slots=True)
class HeadroomReport:
    """헤드룸 실험 전체 결과."""

    groups: dict[str, GroupSummary] = field(default_factory=dict)
    comparisons: list[PairedComparison] = field(default_factory=list)
    gates: list[GateVerdict] = field(default_factory=list)
    best_static_action: ControllerAction | None = None
    static_search_size: int = 0
    n_instances: int = 0

    def summary_table(self) -> str:
        header = (
            f"{'controller':<26} {'도달률':>7} {'cost→target':>12} "
            f"{'final/L0':>11} {'총 GE':>8} {'탐색 GE':>10} "
            f"{'거절':>6} {'CG수렴':>7}"
        )
        lines = [header, "-" * len(header)]
        for name, g in self.groups.items():
            cost = (
                f"{g.median_cost_to_target_ge:.1f}"
                if math.isfinite(g.median_cost_to_target_ge)
                else "미도달"
            )
            lines.append(
                f"{name:<26} {g.success_rate:>6.0%} {cost:>12} "
                f"{g.median_final_loss_ratio:>11.3e} {g.median_total_cost_ge:>8.1f} "
                f"{g.median_search_cost_ge:>10.1f} "
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
    """게이트 A~C를 측정한다.

    Args:
        config: 실험 설정.
        narrow: 프로토콜 원안 행동 공간 (정책이 실제로 쓸 것).
        wide: damping 배수를 넓힌 공간.
        absolute: 도달성 제약 없는 분석용 공간. 해상도가 narrow 수준이어야 한다.
        verbose: 진행 상황 출력.

    Returns:
        ``HeadroomReport``.
    """
    report = HeadroomReport()
    eligible = sum(
        1
        for spec in config.specs
        for seed in config.seeds
        if _is_eligible(make_task(spec, seed))
    )
    report.n_instances = eligible
    report.static_search_size = len(narrow)

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(f"인스턴스 {eligible}개 (indefinite 제외), GE 예산 {config.cost_budget_ge:g}")

    def relabel(group: GroupSummary, name: str) -> GroupSummary:
        """탐색 우승자의 라벨을 정규화한다.

        ``search_best_static`` 은 ``static[7]`` 같은 시행별 라벨을 쓴다. 쌍별
        비교는 ``RunSummary.controller`` 를 키로 조회하므로 여기서 통일해야
        게이트가 결과를 찾을 수 있다.
        """
        for run in group.runs:
            run.controller = name
        group.controller = name
        return group

    # --- baseline: best static (탐색 예산 = len(narrow)) ---
    log(f"best_static 탐색: {len(narrow)}개 조합")
    best_action, static_group, _ = search_best_static(config, narrow)
    static_group = relabel(static_group, "best_static")
    static_runs = static_group.runs
    report.best_static_action = best_action
    report.groups["best_static"] = static_group

    # --- baseline: best open loop (동일 탐색 예산) ---
    log(f"best_open_loop 랜덤 서치: {len(narrow)}회")
    _, open_group, _ = search_best_open_loop(config, narrow)
    open_group = relabel(open_group, "best_open_loop")
    report.groups["best_open_loop"] = open_group

    # --- heuristic ---
    log("heuristic")
    heuristic_runs = run_controller(
        config, lambda: HeuristicController(narrow), label="heuristic"
    )
    report.groups["heuristic"] = summarize_group(heuristic_runs, controller="heuristic")

    # --- oracles ---
    oracle_runs: dict[str, list[RunSummary]] = {}
    for name, space in (
        ("greedy_narrow", narrow),
        ("greedy_wide", wide),
        ("greedy_absolute", absolute),
    ):
        log(f"{name} (sweep {space.hvp_per_sweep} HVP/step)")
        runs = run_controller(
            config, lambda s=space: GreedyOracleController(s), label=name
        )
        oracle_runs[name] = runs
        report.groups[name] = summarize_group(runs, controller=name)

    log(
        f"lookahead{config.lookahead_horizon}_narrow "
        f"(beam {config.lookahead_beam}, 비용 큼)"
    )
    look_label = f"lookahead{config.lookahead_horizon}_narrow"
    look_runs = run_controller(
        config,
        lambda: LookaheadOracleController(
            narrow,
            horizon=config.lookahead_horizon,
            beam_width=config.lookahead_beam,
        ),
        label=look_label,
    )
    oracle_runs[look_label] = look_runs
    report.groups[look_label] = summarize_group(look_runs, controller=look_label)

    # --- 쌍별 비교 ---
    def compare(base: Sequence[RunSummary], treat: Sequence[RunSummary]) -> PairedComparison:
        return compare_paired(base, treat, metric="cost_to_target_ge")

    pairs = [
        (static_runs, oracle_runs["greedy_absolute"]),
        (static_runs, oracle_runs["greedy_narrow"]),
        (static_runs, oracle_runs["greedy_wide"]),
        (oracle_runs["greedy_narrow"], oracle_runs["greedy_wide"]),
        (oracle_runs["greedy_wide"], oracle_runs["greedy_absolute"]),
        (oracle_runs["greedy_narrow"], oracle_runs[look_label]),
        (static_runs, open_group.runs),
        (static_runs, heuristic_runs),
    ]
    report.comparisons = [compare(b, t) for b, t in pairs]

    # --- 게이트 판정 ---
    by_pair = {(c.baseline, c.treatment): c for c in report.comparisons}

    def ratio_of(base: str, treat: str) -> tuple[float, PairedComparison | None]:
        c = by_pair.get((base, treat))
        return (c.ratio_geometric_mean if c else float("nan")), c

    gate_a_ratio, gate_a_cmp = ratio_of("best_static", "greedy_absolute")
    report.gates.append(
        GateVerdict(
            name="A",
            question="적응 제어에 내재적 여지가 있는가 (absolute oracle vs best_static)",
            comparison=gate_a_cmp,
            ratio=gate_a_ratio,
            go_threshold=1.30,
            pivot_threshold=1.10,
            note="one-step 헤드룸이다. 장기 의사결정 여지는 게이트 C가 본다.",
        )
    )

    gate_b_ratio, gate_b_cmp = ratio_of("greedy_narrow", "greedy_absolute")
    _, wide_cmp = ratio_of("greedy_narrow", "greedy_wide")
    report.gates.append(
        GateVerdict(
            name="B",
            question="행동 공간이 병목인가 (absolute vs narrow oracle)",
            comparison=gate_b_cmp,
            ratio=gate_b_ratio,
            go_threshold=1.20,
            pivot_threshold=1.05,
            note=(
                "GO 면 narrow 를 고쳐야 한다. "
                + (
                    f"참고: wide/narrow = {wide_cmp.ratio_geometric_mean:.3f}x"
                    if wide_cmp
                    else ""
                )
            ),
        )
    )

    gate_c_ratio, gate_c_cmp = ratio_of("greedy_narrow", look_label)
    report.gates.append(
        GateVerdict(
            name="C",
            question=f"장기 의사결정이 필요한가 (lookahead{config.lookahead_horizon} vs greedy)",
            comparison=gate_c_cmp,
            ratio=gate_c_ratio,
            go_threshold=1.15,
            pivot_threshold=1.03,
            note=(
                "재설계 판정이면 contextual bandit 이나 heuristic 으로 충분하고 "
                "PPO 의 temporal credit assignment 는 정당화되지 않는다."
            ),
        )
    )
    return report
