"""Stage 2 중간 점검. 게이트 A~C의 신호를 소규모로 확인한다.

일회성 진단 도구다. 본 판정은 ``scripts/run_headroom.py`` 가 여러 seed와
paired design으로 수행한다. 여기서는 구현이 의도대로 동작하는지와 신호의
방향만 본다.

확인 대상:
  - damping 이 로그 공간에서 누적되고 경계에서 클립되는가
  - absolute / wide / narrow 오라클의 서열이 예상과 맞는가 (게이트 A/B)
  - lookahead 가 greedy 보다 나은가 (게이트 C)
  - 높은 damping 이 CG 는 쉽게 만들지만 loss 감소는 느려지는 현상이 보이는가
"""

from __future__ import annotations

import math

from rl_newton.optimizers.action_space import ABSOLUTE, NARROW, WIDE
from rl_newton.optimizers.controllers import (
    FixedController,
    GreedyOracleController,
    HeuristicController,
    LookaheadOracleController,
)
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask

HEADER = (
    f"  {'controller':<30} {'final/L0':>12} {'cost_GE':>9} {'HVP':>6} "
    f"{'rej':>4} {'negc':>5} {'cgconv':>7} {'search_GE':>10}"
)


def run_one(task, controller, config, label: str) -> None:
    opt = NewtonCGOptimizer(task, controller, config, run_id=label, seed=0)
    trace = opt.run()
    ratio = trace.final_loss / trace.initial_loss if trace.initial_loss else float("nan")
    print(
        f"  {label:<30} {ratio:>12.3e} {trace.total_cost_ge:>9.1f} "
        f"{trace.total_hvp:>6} {trace.n_rejected:>4} "
        f"{trace.n_negative_curvature:>5} {trace.n_cg_converged:>7} "
        f"{trace.search_cost_ge:>10.1f}"
    )


def section(title: str) -> None:
    print(f"\n=== {title} ===")
    print(HEADER)
    print("  " + "-" * (len(HEADER) - 2))


def main() -> int:
    print("=== 행동 공간 요약 ===")
    for space in (NARROW, WIDE, ABSOLUTE):
        fixed = space.with_fixed_step_size(1.0)
        print(
            f"  {space.name:<9} mode={space.damping_mode:<9} actions={len(space):>3} "
            f"HVP/sweep={space.hvp_per_sweep:>4} log10범위={space.log10_span:>5.2f} "
            f"| step고정: actions={len(fixed):>2}"
        )

    spec_spd = QuadraticSpec(dimension=64, condition_number=1.0e3)
    spec_ill = QuadraticSpec(
        kind="ill_conditioned", dimension=100, condition_number=1.0e6
    )
    config = NewtonCGConfig(total_steps=30, initial_damping=1.0e-2)

    # step_size 를 고정한 공간이 기본 조건이다 (action aliasing 회피).
    n_fixed = NARROW.with_fixed_step_size(1.0)
    w_fixed = WIDE.with_fixed_step_size(1.0)
    a_fixed = ABSOLUTE.with_fixed_step_size(1.0)

    for name, spec in (("SPD kappa=1e3 d=64", spec_spd), ("ill kappa=1e6 d=100", spec_ill)):
        section(f"{name}, 30 steps, step_size 고정 1.0")
        run_one(
            QuadraticTask(spec, seed=0),
            FixedController(n_fixed.action_from_flat(len(n_fixed) - 1)),
            config,
            "fixed(m=3,k=20)",
        )
        run_one(
            QuadraticTask(spec, seed=0),
            FixedController(n_fixed.action_from_flat(len(n_fixed) // 2)),
            config,
            "fixed(mid)",
        )
        run_one(
            QuadraticTask(spec, seed=0), HeuristicController(n_fixed), config, "heuristic"
        )
        run_one(
            QuadraticTask(spec, seed=0),
            GreedyOracleController(n_fixed),
            config,
            "greedy(narrow)",
        )
        run_one(
            QuadraticTask(spec, seed=0),
            GreedyOracleController(w_fixed),
            config,
            "greedy(wide)",
        )
        run_one(
            QuadraticTask(spec, seed=0),
            GreedyOracleController(a_fixed),
            config,
            "greedy(absolute)",
        )
        run_one(
            QuadraticTask(spec, seed=0),
            LookaheadOracleController(n_fixed, horizon=3, beam_width=4),
            config,
            "lookahead3(narrow)",
        )

    print("\n=== absolute 오라클이 고른 damping 추이 (ill kappa=1e6) ===")
    task = QuadraticTask(spec_ill, seed=0)
    oracle = GreedyOracleController(a_fixed)
    opt = NewtonCGOptimizer(task, oracle, config, run_id="probe", seed=0)
    trace = opt.run()
    print(
        f"  {'step':>4} {'damping':>10} {'k':>3} {'L_after':>12} "
        f"{'residual':>10} {'cgconv':>7} {'trust':>8}"
    )
    for record in trace.records[:12]:
        cg_conv = record.extra.get("cg_converged")
        resid = record.extra.get("cg_residual_ratio", float("nan"))
        print(
            f"  {record.step:>4} {record.damping:>10.2e} {record.cg_budget:>3} "
            f"{record.train_loss_after:>12.4e} {float(resid):>10.2e} "
            f"{str(cg_conv):>7} {record.trust_ratio:>8.3f}"
        )

    print("\n=== 높은 damping: CG는 쉬워지고 loss 감소는 느려지는가 ===")
    print(
        f"  {'damping':>10} {'final/L0':>12} {'cgconv/30':>10} "
        f"{'평균 residual':>14} {'cost_GE':>9}"
    )
    for damping in (1e-2, 1.0, 1e2, 1e4, 1e6):
        task = QuadraticTask(spec_ill, seed=0)
        cfg = NewtonCGConfig(total_steps=30, initial_damping=damping)
        # 배수 1.0 고정이므로 damping 이 그 값에 머문다
        action = next(
            a
            for a in n_fixed.iter_actions()
            if a.damping_multiplier == 1.0 and a.cg_budget == 20
        )
        opt = NewtonCGOptimizer(
            task, FixedController(action), cfg, run_id="damp", seed=0
        )
        trace = opt.run()
        residuals = [
            float(r.extra.get("cg_residual_ratio", float("nan"))) for r in trace.records
        ]
        finite = [r for r in residuals if math.isfinite(r)]
        mean_resid = sum(finite) / len(finite) if finite else float("nan")
        print(
            f"  {damping:>10.0e} {trace.final_loss / trace.initial_loss:>12.3e} "
            f"{trace.n_cg_converged:>10} {mean_resid:>14.3e} "
            f"{trace.total_cost_ge:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
