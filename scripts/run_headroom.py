"""Stage 2 헤드룸 측정. 게이트 A1/A2/B/C/D 판정 (프로토콜 §4 Stage 2).

이 프로젝트의 분기점이다. RL 스택을 만들기 전에 네 가지를 답한다.

```text
A1  현재 상태에서 좋은 damping 이 존재하는가          absolute, H=1
A2  현실적 multiplier 로 접근 가능한가                narrow/wide, H=1/3/5
B   행동 범위 제한의 손해                             absolute vs wide vs narrow, H=1
C   여러 step planning 이 필요한가 (PPO 착수 조건)     narrow/wide H=1 vs 3 vs 5
D   cost-to-target 헤드룸                             target 난이도별
```

디바이스
--------
**CPU 가 기본이며 그것이 옳다.** 대상은 quadratic(d=32~100)과 Rosenbrock(d=2~10)
뿐이다. Stage 0 실측에서 10만 파라미터 MNIST MLP 조차 GPU 런치 오버헤드
지배(0.68 ms/gradient)였으므로 d=100 matvec 을 GPU 로 보내면 순손실이다.
GPU 는 Stage 3 이후에만 쓴다.

재개 가능
---------
run 하나가 끝나는 즉시 ``results/raw/<run>.jsonl`` 에 기록한다. 프로세스가
끊겨도 같은 명령을 다시 실행하면 완료된 조합을 건너뛴다.

사용법:

    # beam calibration (프로토콜 F). 이것을 먼저 하고 beam 을 고정한다.
    uv run python scripts/run_headroom.py --mode calibrate-beam

    # pilot: 예산/target 선정용. dev seed 만 사용
    uv run python scripts/run_headroom.py --mode pilot --beam 2

    # confirmatory: held-out seed. 프로토콜 freeze 이후에만
    uv run python scripts/run_headroom.py --mode confirmatory --beam 2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from rl_newton.benchmark.metrics import TargetSpec
from rl_newton.benchmark.oracle import (
    DEV_SEEDS,
    HELD_OUT_SEEDS,
    HeadroomConfig,
    calibrate_beam_width,
    run_headroom,
)
from rl_newton.benchmark.store import ResultStore, environment_fingerprint
from rl_newton.optimizers.action_space import ABSOLUTE, NARROW, WIDE
from rl_newton.tasks.quadratics import QuadraticSpec
from rl_newton.tasks.rosenbrock import RosenbrockSpec
from rl_newton.utils.provenance import collect_provenance, config_hash, git_commit

# ---------------------------------------------------------------------------
# 프로토콜 D6 사전 등록 target. easy / medium / hard 3단계.
# **pilot 에서 확정한 뒤에는 바꾸지 않는다.** 변경 시 프로토콜 §9에 기록한다.
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict[str, TargetSpec]] = {
    "spd": {
        "easy": TargetSpec("relative_loss", 1.0e-2),
        "medium": TargetSpec("relative_loss", 1.0e-4),
        "hard": TargetSpec("relative_loss", 1.0e-6),
    },
    "ill_conditioned": {
        "easy": TargetSpec("relative_loss", 1.0e-2),
        "medium": TargetSpec("relative_loss", 1.0e-4),
        "hard": TargetSpec("relative_loss", 1.0e-6),
    },
    "indefinite": {
        "easy": TargetSpec("relative_loss", 1.0e-1),
        "medium": TargetSpec("relative_loss", 1.0e-2),
        "hard": TargetSpec("relative_loss", 1.0e-3),
    },
    "rosenbrock": {
        "easy": TargetSpec("absolute_loss", 1.0e-1),
        "medium": TargetSpec("absolute_loss", 1.0e-2),
        "hard": TargetSpec("absolute_loss", 1.0e-4),
    },
}


def pilot_specs() -> list:
    """pilot subset. 예산/target/beam 선정에만 쓴다."""
    return [
        QuadraticSpec(kind="spd", dimension=64, condition_number=1.0e2),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        RosenbrockSpec(dimension=2),
    ]


def confirmatory_specs() -> list:
    """confirmatory. pilot 과 다른 조건수와 차원을 포함한다."""
    return [
        QuadraticSpec(kind="spd", dimension=64, condition_number=1.0e2),
        QuadraticSpec(kind="spd", dimension=128, condition_number=1.0e3),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        QuadraticSpec(kind="ill_conditioned", dimension=200, condition_number=1.0e6),
        RosenbrockSpec(dimension=2),
        RosenbrockSpec(dimension=10, randomize_start=True),
    ]


def _clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def build_config(args: argparse.Namespace) -> tuple[HeadroomConfig, dict]:
    if args.mode == "confirmatory":
        specs = confirmatory_specs()
        seeds = list(HELD_OUT_SEEDS)[: args.seeds]
        phase = "confirmatory"
    else:
        specs = pilot_specs()
        seeds = list(DEV_SEEDS)[: args.seeds]
        phase = "pilot"
    if args.max_tasks is not None:
        specs = specs[: args.max_tasks]

    if args.control_step_size:
        narrow, wide, absolute = NARROW, WIDE, ABSOLUTE
        condition = "step_size_controlled"
    else:
        narrow = NARROW.with_fixed_step_size(1.0)
        wide = WIDE.with_fixed_step_size(1.0)
        absolute = ABSOLUTE.with_fixed_step_size(1.0)
        condition = "step_size_fixed"

    config = HeadroomConfig(
        specs=specs,
        seeds=seeds,
        targets=TARGETS,
        cost_budget_ge=args.budget,
        max_steps=args.max_steps,
        horizons=tuple(args.horizons),
        beam_width=args.beam,
        tuning_budget=args.tuning_budget,
        phase=phase,  # type: ignore[arg-type]
        primary_difficulty=args.difficulty,
        device="cpu",
    )
    meta = {
        "mode": args.mode,
        "condition": condition,
        "phase": phase,
        "seeds": seeds,
        "budget_ge": args.budget,
        "beam": args.beam,
        "horizons": list(args.horizons),
        "difficulty": args.difficulty,
        "n_specs": len(specs),
        "narrow_only": bool(args.narrow_only),
    }
    return config, meta | {"spaces": (narrow, wide, absolute)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2 헤드룸 측정 (게이트 A1/A2/B/C/D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["calibrate-beam", "pilot", "confirmatory"],
        default="pilot",
    )
    parser.add_argument("--seeds", type=int, default=3, help="사용할 seed 개수")
    parser.add_argument("--budget", type=float, default=600.0, help="GE 예산")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--beams",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="calibrate-beam 에서 시험할 beam 폭",
    )
    parser.add_argument(
        "--cal-horizons",
        type=int,
        nargs="+",
        default=[3, 5],
        help="calibrate-beam 에서 시험할 horizon",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="사용할 task spec 수 상한. dry run 용",
    )
    parser.add_argument(
        "--narrow-only",
        action="store_true",
        help="calibrate-beam 에서 narrow 만 사용. dry run 용",
    )
    parser.add_argument("--tuning-budget", type=int, default=None, help="N_tune")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument(
        "--control-step-size",
        action="store_true",
        help="step_size 축을 제어에 포함. 기본은 1.0 고정 (aliasing 회피)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU 스레드 수. wall-clock tie-break 안정화를 위해 고정한다",
    )
    parser.add_argument(
        "--concurrent-processes",
        type=int,
        default=1,
        help="동시 실행 중인 다른 실험 프로세스 수 (기록용)",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/summaries"))
    parser.add_argument(
        "--fresh", action="store_true", help="캐시를 무시하고 새 파일에 기록"
    )
    args = parser.parse_args()

    config, meta = build_config(args)
    narrow, wide, absolute = meta.pop("spaces")

    commit, dirty = git_commit(".")
    cfg_hash = config_hash(_clean(meta))
    tag = f"{args.mode}_{meta['condition']}_b{args.beam}_{cfg_hash[:8]}"
    raw_path = args.raw_dir / f"headroom_{tag}.jsonl"
    if args.fresh and raw_path.exists():
        raw_path = raw_path.with_name(f"{raw_path.stem}_fresh{raw_path.suffix}")

    store = ResultStore(raw_path, git_commit=commit, config_hash=cfg_hash)

    # wall-clock 은 beam 선택의 마지막 tie-break 로만 쓰이지만, 다른 실험과 CPU
    # 를 공유하면 그 tie-break 가 흔들린다. 스레드를 고정하고 환경을 기록한다.
    env = environment_fingerprint(pin_threads=args.threads)
    meta["environment"] = env
    meta["concurrent_processes"] = args.concurrent_processes

    print("=" * 96)
    print(f"Stage 2 {args.mode}  조건={meta['condition']}  device=cpu (GPU 미사용)")
    print(f"  seeds={config.seeds}  GE 예산={config.cost_budget_ge:g}  beam={args.beam}")
    print(f"  git={commit}{' (dirty)' if dirty else ''}  config_hash={cfg_hash}")
    print(
        f"  torch threads={env['torch_num_threads']} "
        f"interop={env['torch_num_interop_threads']} cpu={env['cpu_count']}"
    )
    print(f"  raw={raw_path}  (이미 {len(store)}개 기록됨, 완료분은 건너뜀)")
    for space in (narrow, wide, absolute):
        print(
            f"  {space.name:<26} actions={len(space):>4} "
            f"HVP/sweep={space.hvp_per_sweep:>5} log10범위={space.log10_span:>6.2f}"
        )
    print("=" * 96)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "calibrate-beam":
        calibration = calibrate_beam_width(
            config,
            narrow=narrow,
            wide=narrow if args.narrow_only else wide,
            beams=tuple(args.beams),
            horizons=tuple(args.cal_horizons),
            store=store,
            code_dirty=dirty,
        )
        print("\n" + calibration.table())
        print(f"\n선택: beam {calibration.selected_beam}")
        print(f"  근거: {calibration.rationale}")
        path = args.out_dir / f"beam_calibration_{cfg_hash[:8]}.json"
        path.write_text(
            json.dumps(
                _clean(
                    {
                        "meta": meta,
                        "selected_beam": calibration.selected_beam,
                        "reference_beam": calibration.reference_beam,
                        "tolerance": calibration.tolerance,
                        "rationale": calibration.rationale,
                        "rows": {
                            f"{s}|H{h}|b{b}": row
                            for (s, h, b), row in calibration.rows.items()
                        },
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"저장: {path}")
        return 0

    report = run_headroom(
        config,
        narrow=narrow,
        wide=wide,
        absolute=absolute,
        store=store,
        code_dirty=dirty,
        verbose=True,
    )

    print("\n" + "=" * 96)
    print("결과 요약 (Track E = logΔ, Track T = 도달률 + cost→τ)")
    print("=" * 96)
    print(report.summary_table())

    print("\n" + "=" * 96)
    print("Track E 쌍별 차이 (양수면 treatment 가 좋다, 단위 nat)")
    print("=" * 96)
    for d in report.track_e_deltas:
        print("  " + d.describe())

    print("\n" + "=" * 96)
    print("Track T 쌍별 비율 (1보다 크면 treatment 가 싸다)")
    print("=" * 96)
    for level, c in report.track_t_ratios.items():
        print(f"  [{level}] " + c.describe())

    print("\n" + "=" * 96)
    print("게이트 판정")
    print("=" * 96)
    for gate in report.gates:
        print("  " + gate.describe().replace("\n", "\n  "))

    verdicts = {g.name: g.verdict for g in report.gates}
    print("\n" + "-" * 96)
    print("  " + "  ".join(f"{name}={v}" for name, v in verdicts.items()))
    if args.mode == "pilot":
        print(
            "\n  주의: pilot 결과다. 예산/target/beam 선정에만 쓰고 결론에 쓰지 않는다."
        )

    print(f"\n  {store.describe()}")
    failures = store.failures()
    if failures:
        print(f"  실패 {len(failures)}건:")
        for record in failures[:5]:
            print(f"    {record.key.as_str()}: {record.error}")

    path = args.out_dir / f"headroom_{tag}.json"
    payload = {
        "meta": meta,
        "experiment_id": report.experiment_id,
        "identity": report.identity,
        "n_instances": report.n_instances,
        "tuning_budget": report.tuning_budget,
        "tuning_runs": report.tuning_runs,
        "best_static_action": (
            asdict(report.best_static_action) if report.best_static_action else None
        ),
        "groups": {
            name: {k: v for k, v in asdict(g).items() if k != "runs"}
            for name, g in report.groups.items()
        },
        "track_e_deltas": [asdict(d) for d in report.track_e_deltas],
        "track_t_ratios": {k: asdict(v) for k, v in report.track_t_ratios.items()},
        "gates": [
            {
                "name": g.name,
                "track": g.track,
                "question": g.question,
                "statistic": g.statistic,
                "unit": g.unit,
                "verdict": g.verdict,
                "go_threshold": g.go_threshold,
                "pivot_threshold": g.pivot_threshold,
            }
            for g in report.gates
        ],
        "n_failures": len(failures),
        "raw_path": str(raw_path),
        "provenance": collect_provenance(_clean(meta), include_diff=False).to_dict(),
    }
    path.write_text(
        json.dumps(_clean(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
