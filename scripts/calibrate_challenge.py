"""Challenge set 선정: **비적응 baseline 만으로 측정 가능성**을 판정한다 (D20).

D19 에서 dev subset 3 spec 중 2개가 컨트롤러를 구분하지 못한다는 것이 확인됐다.
측정 가능한 regime 이 ``quad_ill κ=1e5`` 하나뿐이다.

**planner 결과를 보지 않는다.** 선정 기준은 성능 우열이 아니라 측정 가능성이다.
어떤 컨트롤러가 이기는지로 benchmark 를 고르면 결과를 본 뒤 유리한 task 를
추가한 것이 된다.

```text
사용 baseline:  best_static / best open_loop / heuristic / C0(onestep_narrow)
비공개:         shrinking / committed / fresh / beam 결과
```

채택 조건 (사전 고정, 프로토콜 D20)
-----------------------------------
```text
failure_rate = 0                        numerical failure 없음
joint floor-hit rate <= 1/3             포화가 과도하지 않음
각 baseline median logΔ >= 1 nat        문제를 전혀 못 줄이는 조건 아님
median distance-to-ceiling >= 3 nat     floor 까지 e^3 ~ 20배 여유
```

``ceiling = log(L0 / loss_floor)`` 이다. 비율 기준(``0.8 x ceiling``)은 현재
open-loop 가 25.456 이라 지나치게 빡빡하므로 **절대 여유 3 nat** 으로 정한다.

seed 분리
---------
```text
calibration seeds  0, 1        이 스크립트. spec 선정에만
beam-8 dev seeds   2, 3, 4     설정 선택에만
held-out           5 ~ 14      최종 평가에만
```

사용법:
    python scripts/calibrate_challenge.py [--budget 150] [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import math

from rl_newton.benchmark.metrics import (
    RELATIVE_LOSS_FLOOR,
    TargetSpec,
    summarize_run,
)
from rl_newton.optimizers.action_space import NARROW
from rl_newton.optimizers.controllers import (
    FixedController,
    HeuristicController,
    OneStepEfficiencyController,
    make_open_loop_controller,
)
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask

# --- 채택 조건 (사전 고정) ---
MAX_JOINT_FLOOR_RATE = 1.0 / 3.0
MIN_MEDIAN_LOG_IMPROVEMENT = 1.0
MIN_DISTANCE_TO_CEILING = 3.0
MAX_SPECS = 4

# --- 후보군. conditioning 축을 촘촘히 (D20) ---
CANDIDATES: list[tuple[str, object]] = [
    ("quad_d100_k1e3", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e3)),
    ("quad_d100_k1e4", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e4)),
    ("quad_d100_k1e5", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5)),
    ("quad_d100_k1e6", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)),
    ("rosen_d5", RosenbrockSpec(dimension=5)),
]

TARGET = TargetSpec("relative_loss", 1.0e-6)


def make(spec, seed: int):
    if isinstance(spec, QuadraticSpec):
        return QuadraticTask(spec, seed=seed)
    return RosenbrockTask(spec, seed=seed)


def run_one(controller, spec, seed: int, budget: float):
    task = make(spec, seed)
    config = NewtonCGConfig(total_steps=300, cost_budget_ge=budget, initial_damping=1.0e-2)
    trace = NewtonCGOptimizer(task, controller, config, run_id="cal", seed=seed).run()
    return summarize_run(trace, TARGET, budget_ge=budget)


def baseline_panel(space, budget: float, spec, seed: int, n_tune: int):
    """비적응 baseline 만. planner 는 만들지 않는다."""
    out: dict[str, object] = {}

    # best_static: 후보를 균등 간격으로 n_tune 개 평가하고 최고를 고른다.
    best = None
    for i in range(min(n_tune, len(space))):
        flat = int(i * len(space) / min(n_tune, len(space)))
        s = run_one(FixedController(space.action_from_flat(flat)), spec, seed, budget)
        if best is None or s.log_improvement > best.log_improvement:
            best = s
    out["best_static"] = best

    # best open_loop: resource-clock 스케줄 (D17). 동일 n_tune.
    import random

    rng = random.Random(0)
    best_ol = None
    for _ in range(n_tune):
        flats = tuple(rng.randrange(len(space)) for _ in range(4))
        cuts = sorted(rng.uniform(0.05, 0.95) for _ in range(3))
        ctrl = make_open_loop_controller(space, flats, (*cuts, 1.0))
        s = run_one(ctrl, spec, seed, budget)
        if best_ol is None or s.log_improvement > best_ol.log_improvement:
            best_ol = s
    out["best_open_loop"] = best_ol

    out["heuristic"] = run_one(HeuristicController(space), spec, seed, budget)
    out["onestep_narrow"] = run_one(OneStepEfficiencyController(space), spec, seed, budget)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n-tune", type=int, default=6, help="baseline 튜닝 후보 수")
    args = parser.parse_args()

    space = NARROW.with_fixed_step_size(1.0)
    print(f"Challenge set calibration  예산 {args.budget:g} GE  "
          f"calibration seeds={args.seeds}  N_tune={args.n_tune}")
    print(f"채택 조건: failure=0, joint floor<= {MAX_JOINT_FLOOR_RATE:.2f}, "
          f"median logΔ>= {MIN_MEDIAN_LOG_IMPROVEMENT:g}, "
          f"ceiling 여유>= {MIN_DISTANCE_TO_CEILING:g} nat")
    print("**planner 결과는 열지 않는다.**\n")

    verdicts: list[tuple[str, bool, float, str]] = []
    for name, spec in CANDIDATES:
        rows = []
        for seed in args.seeds:
            rows.append(baseline_panel(space, args.budget, spec, seed, args.n_tune))

        l0 = float(make(spec, args.seeds[0]).initial_loss)
        floor = max(2.2250738585072014e-308, abs(l0) * RELATIVE_LOSS_FLOOR)
        ceiling = math.log(l0) - math.log(floor)

        print(f"=== {name}  L0={l0:.4e}  ceiling={ceiling:.2f} nat ===")
        print(f"  {'baseline':<18} {'median logΔ':>12} {'여유':>8} {'floor':>6} {'fail':>6}")

        ok = True
        reasons: list[str] = []
        worst_gap = math.inf
        for label in ("best_static", "best_open_loop", "heuristic", "onestep_narrow"):
            vals = [r[label].log_improvement for r in rows]  # type: ignore[index]
            finite = sorted(v for v in vals if math.isfinite(v))
            med = finite[len(finite) // 2] if finite else float("nan")
            n_floor = sum(1 for r in rows if r[label].floor_hit)  # type: ignore[index]
            n_fail = sum(1 for r in rows if r[label].failure_rate > 0.0)  # type: ignore[index]
            gap = ceiling - med
            worst_gap = min(worst_gap, gap)
            print(
                f"  {label:<18} {med:>12.4f} {gap:>8.2f} "
                f"{n_floor}/{len(rows)}  {n_fail}/{len(rows)}"
            )
            if n_fail:
                ok = False
                reasons.append(f"{label} numerical failure")
            if n_floor / len(rows) > MAX_JOINT_FLOOR_RATE:
                ok = False
                reasons.append(f"{label} floor-hit {n_floor}/{len(rows)}")
            if not math.isfinite(med) or med < MIN_MEDIAN_LOG_IMPROVEMENT:
                ok = False
                reasons.append(f"{label} median logΔ {med:.3f} < {MIN_MEDIAN_LOG_IMPROVEMENT}")
            if gap < MIN_DISTANCE_TO_CEILING:
                ok = False
                reasons.append(f"{label} ceiling 여유 {gap:.2f} < {MIN_DISTANCE_TO_CEILING}")

        verdict = "채택" if ok else "탈락"
        print(f"  -> {verdict}" + (f"  ({'; '.join(reasons[:3])})" if reasons else ""))
        print()
        verdicts.append((name, ok, worst_gap, "; ".join(reasons[:3])))

    accepted = [v for v in verdicts if v[1]]
    print("=== 결과 ===")
    for name, ok, gap, why in verdicts:
        print(f"  {name:<18} {'채택' if ok else '탈락':<4} 최소여유={gap:>6.2f}  {why}")
    print()
    print(f"  통과 {len(accepted)}개 / 후보 {len(verdicts)}개")
    if len(accepted) > MAX_SPECS:
        print(f"  최대 {MAX_SPECS}개 초과. log10(κ) 간격을 고르게 덮는 spec 을 선택한다.")
        print("  **planner 성능은 선택에 쓰지 않는다.**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
