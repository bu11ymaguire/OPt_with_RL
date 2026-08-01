"""Stage 2 헤드룸 측정과 게이트 A~C 판정 (프로토콜 §4 Stage 2).

이 프로젝트의 분기점이다. RL 스택을 만들기 전에 세 가지를 답한다.

```text
게이트 A  적응 제어에 내재적 여지가 있는가
게이트 B  행동 공간이 병목인가
게이트 C  장기 의사결정이 필요한가 (= PPO 가 정당화되는가)
```

사용법:

    # 기본 조건: step_size 고정 (action aliasing 회피)
    uv run python scripts/run_headroom.py

    # step_size 제어 추가 조건 (기여 분리)
    uv run python scripts/run_headroom.py --control-step-size

    # 빠른 확인
    uv run python scripts/run_headroom.py --seeds 3 --budget 300
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from rl_newton.benchmark.metrics import TargetSpec
from rl_newton.benchmark.oracle import HeadroomConfig, run_headroom
from rl_newton.optimizers.action_space import ABSOLUTE, NARROW, WIDE
from rl_newton.tasks.quadratics import QuadraticSpec
from rl_newton.tasks.rosenbrock import RosenbrockSpec
from rl_newton.utils.provenance import collect_provenance

# 프로토콜 D6 사전 등록 target. 실험 전에 확정된 값이며 임의로 바꾸지 않는다.
TARGETS = {
    "spd": TargetSpec(metric="relative_loss", value=1.0e-6),
    "ill_conditioned": TargetSpec(metric="relative_loss", value=1.0e-4),
    "indefinite": TargetSpec(metric="relative_loss", value=1.0e-2),
    "rosenbrock": TargetSpec(metric="absolute_loss", value=1.0e-4),
}


def build_specs() -> list:
    """헤드룸 측정 대상. indefinite 는 러너가 자동 제외한다."""
    return [
        QuadraticSpec(kind="spd", dimension=64, condition_number=1.0e2),
        QuadraticSpec(kind="spd", dimension=64, condition_number=1.0e3),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6),
        RosenbrockSpec(dimension=2),
        RosenbrockSpec(dimension=10, randomize_start=True),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2 헤드룸 측정 (게이트 A~C)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5, help="seed 개수 (0..n-1)")
    parser.add_argument("--budget", type=float, default=600.0, help="GE 예산")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=3, help="lookahead horizon")
    parser.add_argument("--beam", type=int, default=4, help="lookahead beam width")
    parser.add_argument(
        "--control-step-size",
        action="store_true",
        help="step_size 축을 제어에 포함한다. 기본은 1.0 고정 (aliasing 회피)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summaries"),
        help="결과 JSON 저장 위치",
    )
    args = parser.parse_args()

    if args.control_step_size:
        narrow, wide, absolute = NARROW, WIDE, ABSOLUTE
        condition = "step_size_controlled"
    else:
        narrow = NARROW.with_fixed_step_size(1.0)
        wide = WIDE.with_fixed_step_size(1.0)
        absolute = ABSOLUTE.with_fixed_step_size(1.0)
        condition = "step_size_fixed"

    config = HeadroomConfig(
        specs=build_specs(),
        seeds=list(range(args.seeds)),
        targets=TARGETS,
        cost_budget_ge=args.budget,
        max_steps=args.max_steps,
        lookahead_horizon=args.horizon,
        lookahead_beam=args.beam,
    )

    print("=" * 96)
    print(f"Stage 2 헤드룸 측정  조건={condition}")
    print(
        f"  narrow  actions={len(narrow):>3}  HVP/sweep={narrow.hvp_per_sweep:>4}  "
        f"log10범위={narrow.log10_span:.2f}"
    )
    print(
        f"  wide    actions={len(wide):>3}  HVP/sweep={wide.hvp_per_sweep:>4}  "
        f"log10범위={wide.log10_span:.2f}"
    )
    print(
        f"  absolute actions={len(absolute):>3} HVP/sweep={absolute.hvp_per_sweep:>4}  "
        f"log10범위={absolute.log10_span:.2f}"
    )
    print("=" * 96)

    report = run_headroom(
        config, narrow=narrow, wide=wide, absolute=absolute, verbose=True
    )

    print("\n" + "=" * 96)
    print("결과 요약 (cost→target 은 도달한 run 의 중앙값, 미도달은 별도 표기)")
    print("=" * 96)
    print(report.summary_table())

    print("\n" + "=" * 96)
    print("쌍별 비교 (비율 > 1 이면 treatment 가 더 싸다)")
    print("=" * 96)
    for comparison in report.comparisons:
        print("  " + comparison.describe())

    print("\n" + "=" * 96)
    print("게이트 판정")
    print("=" * 96)
    for gate in report.gates:
        print("  " + gate.describe().replace("\n", "\n  "))

    verdicts = {g.name: g.verdict for g in report.gates}
    print("\n" + "-" * 96)
    print(f"  A(내재적 헤드룸)={verdicts.get('A')}  "
          f"B(행동공간 병목)={verdicts.get('B')}  "
          f"C(장기 의사결정)={verdicts.get('C')}")

    # --- 저장 ---
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"headroom_{condition}.json"
    payload = {
        "condition": condition,
        "n_instances": report.n_instances,
        "cost_budget_ge": args.budget,
        "static_search_size": report.static_search_size,
        "best_static_action": (
            asdict(report.best_static_action) if report.best_static_action else None
        ),
        "groups": {
            name: {
                k: v
                for k, v in asdict(g).items()
                if k != "runs" and (not isinstance(v, float) or math.isfinite(v))
            }
            for name, g in report.groups.items()
        },
        "comparisons": [asdict(c) for c in report.comparisons],
        "gates": [
            {
                "name": g.name,
                "question": g.question,
                "ratio": g.ratio if math.isfinite(g.ratio) else None,
                "verdict": g.verdict,
                "go_threshold": g.go_threshold,
                "pivot_threshold": g.pivot_threshold,
            }
            for g in report.gates
        ],
        "provenance": collect_provenance({"condition": condition}, include_diff=False).to_dict(),
    }

    def _clean(obj):
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [_clean(v) for v in obj]
        return obj

    path.write_text(
        json.dumps(_clean(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
