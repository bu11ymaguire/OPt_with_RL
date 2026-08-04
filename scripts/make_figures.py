"""논문 그림을 raw 결과에서 직접 생성한다.

**손으로 숫자를 옮기지 않는다.** 표와 같은 경로(`metrics.median_of`, 같은 paired
비교)를 쓰므로 그림과 표가 어긋날 수 없다.

```text
Figure 1  controller ladder 와 held-out paired improvement
Figure 2  committed vs shrinking. planning 과 feedback 분해
Figure 3  model mismatch. full-batch 와 minibatch regime
Figure 4  acceptance criterion ablation
```

사용법:
    python scripts/make_figures.py --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl_newton.benchmark.metrics import (  # noqa: E402
    RunSummary,
    compare_paired_delta,
    median_of,
)
from rl_newton.benchmark.store import ResultStore  # noqa: E402

_SEED_SUFFIX = re.compile(r"_seed\d+$")

HELD_OUT = "headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl"
MICRO_CONTROL = "headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl"
MICRO_FIXED = "headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl"

# 사다리 순서. 아래에서 위로 강해진다 (draft §3).
LADDER = [
    ("best_static", "tuned constant"),
    ("best_open_loop", "open-loop schedule"),
    ("onestep_narrow", "one-step greedy"),
    ("committed_Q4_narrow", "committed plan"),
    ("shrinking_Q4_narrow", "replanning (MPC)"),
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def spec_of(run: RunSummary) -> str:
    return _SEED_SUFFIX.sub("", run.task_instance_id)


def regime_label(run: RunSummary) -> str:
    base = spec_of(run)
    if base.endswith("_fb"):
        return "full-batch"
    tail = base.rsplit("_", 1)[-1]
    return f"batch {tail[2:]}" if tail.startswith("cs") else base


def load(path: Path) -> tuple[dict[str, list[RunSummary]], dict[str, str]]:
    by_controller: dict[str, list[RunSummary]] = {}
    for record in ResultStore(path):
        if record.status != "completed" or record.summary is None:
            continue
        by_controller.setdefault(record.key.controller, []).append(record.summary)
    alias: dict[str, str] = {}
    summary = path.parents[1] / "summaries" / f"{path.stem}.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for family, manifest in (payload.get("selections") or {}).items():
            label = manifest.get("selected_label")
            if label and label in by_controller:
                alias["best_static" if family == "static" else f"best_{family}"] = label
    return by_controller, alias


def get(by_controller, alias, name) -> list[RunSummary]:
    return by_controller.get(alias.get(name, name), [])


def figure1(raw_dir: Path, out: Path) -> None:
    """사다리별 절대 개선과 튜닝 상수 대비 쌍별 차이."""
    by_controller, alias = load(raw_dir / HELD_OUT)
    base = get(by_controller, alias, "best_static")

    labels, absolute, deltas, los, his = [], [], [], [], []
    for name, pretty in LADDER:
        runs = get(by_controller, alias, name)
        if not runs:
            continue
        labels.append(pretty)
        absolute.append(median_of([r.log_improvement for r in runs]))
        if name == "best_static":
            deltas.append(0.0)
            los.append(0.0)
            his.append(0.0)
            continue
        d = compare_paired_delta(base, runs, metric="log_improvement")
        deltas.append(d.median_delta)
        los.append(d.median_delta - d.delta_ci[0])
        his.append(d.delta_ci[1] - d.median_delta)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    y = range(len(labels))

    axes[0].barh(list(y), absolute, color="#4C72B0", height=0.6)
    axes[0].set_yticks(list(y), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("median log improvement $J_E$ [nat]")
    axes[0].set_title("(a) terminal improvement at 150 GE")
    for i, v in enumerate(absolute):
        axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

    axes[1].barh(list(y), deltas, xerr=[los, his], color="#DD8452", height=0.6, capsize=3)
    axes[1].set_yticks(list(y), labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("paired $\\Delta$ vs tuned constant [nat]")
    axes[1].set_title("(b) paired improvement, 95% CI, $n=40$")
    for i, v in enumerate(deltas):
        if i:
            axes[1].text(v + 0.04, i, f"{v:+.2f}", va="center", fontsize=8)

    # **막대를 서로 빼서 읽으면 안 된다.** 쌍별 차이의 median 은 선형이 아니다.
    # 실측: median(committed−constant)=+2.09, median(replanning−constant)=+1.69 인데
    # median(replanning−committed)=+0.010 이다. 각 spec 안에서는 두 planner 가
    # 사실상 동률이고(+0.47/−0.02/+0.01/+0.00) pooled median 이 서로 다른 인스턴스에
    # 떨어져 생기는 현상이다. 증분 효과는 Figure 2 를 봐야 한다.
    axes[1].text(
        0.5,
        -0.32,
        "Paired medians are not additive: bars must not be subtracted from each other.\n"
        "For incremental effects (planning vs feedback) see Figure 2.",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    # **역할 분리.** Figure 1 은 절대 성능을 설명한다. 인과적 분해와 핵심 주장은
    # Figure 2 의 직접 쌍별 delta 에서만 가져온다.
    fig.suptitle(
        "Figure 1. Absolute performance of the controller ladder on held-out instances\n"
        "(4 ill-conditioned SPD quadratic specs $\\times$ 10 seeds). "
        "Descriptive; see Figure 2 for the decomposition.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure2(raw_dir: Path, out: Path) -> None:
    """planning 과 feedback 의 분해. 증분과 개별 쌍 분포."""
    by_controller, alias = load(raw_dir / HELD_OUT)

    steps = [
        ("open-loop\n$-$ constant", "best_static", "best_open_loop"),
        ("one-step\n$-$ constant", "best_static", "onestep_narrow"),
        ("planning\n$-$ one-step", "onestep_narrow", "shrinking_Q4_narrow"),
        ("feedback\n$-$ committed", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    ]
    names, meds, los, his = [], [], [], []
    per_pair: list[float] = []
    for pretty, b, t in steps:
        rb, rt = get(by_controller, alias, b), get(by_controller, alias, t)
        if not rb or not rt:
            continue
        d = compare_paired_delta(rb, rt, metric="log_improvement")
        names.append(pretty)
        meds.append(d.median_delta)
        los.append(d.median_delta - d.delta_ci[0])
        his.append(d.delta_ci[1] - d.median_delta)
        if pretty.startswith("feedback"):
            bmap = {(r.task_instance_id, r.seed): r for r in rb}
            tmap = {(r.task_instance_id, r.seed): r for r in rt}
            per_pair = sorted(
                tmap[k].log_improvement - bmap[k].log_improvement
                for k in set(bmap) & set(tmap)
                if math.isfinite(tmap[k].log_improvement)
                and math.isfinite(bmap[k].log_improvement)
            )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#C44E52"]
    axes[0].bar(names, meds, yerr=[los, his], color=colors[: len(names)], capsize=3)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("paired $\\Delta$ [nat]")
    axes[0].set_title("(a) where the improvement comes from")
    for i, v in enumerate(meds):
        axes[0].text(i, v + 0.05, f"{v:+.3f}", ha="center", fontsize=8)

    axes[0].text(
        0.5,
        -0.30,
        "Each bar is a paired median over the same 40 instances.\n"
        "Bars are separate statistics and do not sum to $A2$.",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    if per_pair:
        axes[1].scatter(per_pair, range(len(per_pair)), s=14, color="#C44E52")
        axes[1].axvline(0.0, color="black", lw=0.8)
        med = median_of(per_pair)
        axes[1].axvline(med, color="#4C72B0", lw=1.2, ls="--", label=f"median {med:+.3f}")
        n_pos = sum(1 for v in per_pair if v > 0.0)
        axes[1].set_xlabel("$J_E$(replanning) $-$ $J_E$(committed) [nat]")
        axes[1].set_ylabel("instance (sorted)")
        axes[1].set_title(
            f"(b) per-instance feedback effect\n{n_pos} positive / {len(per_pair)}"
        )
        axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Figure 2. Directly measured paired effects. Multi-step planning improves over "
        "one-step control;\nreplanning during execution shows no practically large "
        "benefit over a committed plan.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _regime_table(by_controller, alias) -> tuple[list[str], dict[str, list[float]]]:
    regimes: set[str] = set()
    for name, _ in LADDER:
        for run in get(by_controller, alias, name):
            regimes.add(regime_label(run))
    order = ["full-batch", "batch 128", "batch 64"]
    ordered = [r for r in order if r in regimes] + sorted(regimes - set(order))

    values: dict[str, list[float]] = {}
    for name, pretty in LADDER:
        runs = get(by_controller, alias, name)
        if not runs:
            continue
        values[pretty] = [
            median_of([r.log_improvement for r in runs if regime_label(r) == reg])
            for reg in ordered
        ]
    return ordered, values


def figure3(raw_dir: Path, out: Path) -> None:
    """model mismatch. regime 별 절대 개선과 committed 거절률."""
    path = raw_dir / MICRO_CONTROL
    if not path.exists():
        return
    by_controller, alias = load(path)
    regimes, values = _regime_table(by_controller, alias)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    width = 0.16
    for i, (pretty, vals) in enumerate(values.items()):
        xs = [j + (i - len(values) / 2) * width for j in range(len(regimes))]
        axes[0].bar(xs, vals, width=width, label=pretty)
    axes[0].set_xticks(range(len(regimes)), regimes)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("median $J_E$ [nat]")
    axes[0].set_title("(a) terminal improvement by regime  ($n=3$ each)")
    axes[0].legend(fontsize=7, ncols=2)
    # GE 는 regime 내부에서만 compute-matched 다 (draft §2.2). regime 간 막대 높이를
    # FLOP 비교로 읽으면 안 된다.
    axes[0].text(
        0.5,
        -0.30,
        "Regimes are matched on gradient-equivalent calls, not on FLOPs.\n"
        "Compare within a regime; across regimes this panel is descriptive only.",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    rej = {}
    for name, pretty in LADDER:
        runs = get(by_controller, alias, name)
        if not runs or name not in ("committed_Q4_narrow", "shrinking_Q4_narrow"):
            continue
        rej[pretty] = [
            median_of([r.rejection_rate for r in runs if regime_label(r) == reg])
            for reg in regimes
        ]
    for pretty, vals in rej.items():
        axes[1].plot(range(len(regimes)), vals, marker="o", label=pretty)
    axes[1].set_xticks(range(len(regimes)), regimes)
    axes[1].set_ylim(-0.05, 1.0)
    axes[1].set_ylabel("median step rejection rate")
    axes[1].set_title("(b) a stale plan is rejected")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Figure 3. Model mismatch: planning loses its advantage, "
        "committed plans collapse (exploratory)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure4(raw_dir: Path, out: Path) -> None:
    """acceptance criterion ablation. 같은 비교를 두 규칙에서."""
    paths = {
        "minibatch-local": raw_dir / MICRO_CONTROL,
        "fixed-evaluation": raw_dir / MICRO_FIXED,
    }
    if not all(p.exists() for p in paths.values()):
        return

    comparisons = [
        ("$A2$: replanning $-$ constant", "best_static", "shrinking_Q4_narrow"),
        ("$C2$: replanning $-$ one-step", "onestep_narrow", "shrinking_Q4_narrow"),
        ("$C3$: replanning $-$ committed", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    ]
    fig, axes = plt.subplots(1, len(comparisons), figsize=(10.0, 3.2), sharey=False)
    for ax, (title, b, t) in zip(axes, comparisons, strict=True):
        for offset, (rule, path) in enumerate(paths.items()):
            by_controller, alias = load(path)
            regimes, _ = _regime_table(by_controller, alias)
            rb, rt = get(by_controller, alias, b), get(by_controller, alias, t)
            meds = []
            for reg in regimes:
                sb = [r for r in rb if regime_label(r) == reg]
                st = [r for r in rt if regime_label(r) == reg]
                if not sb or not st:
                    meds.append(float("nan"))
                    continue
                meds.append(
                    compare_paired_delta(sb, st, metric="log_improvement").median_delta
                )
            xs = [j + (offset - 0.5) * 0.34 for j in range(len(regimes))]
            ax.bar(xs, meds, width=0.32, label=rule)
            ax.set_xticks(range(len(regimes)), regimes, fontsize=8)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("paired $\\Delta$ [nat]")
    axes[0].legend(fontsize=8)

    fig.suptitle(
        "Figure 4. Acceptance criterion ablation. The alternative criterion reduced, "
        "but did not eliminate,\nthe apparent advantage of replanning (exploratory, "
        "$n=3$ per regime)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("figure1_ladder.png", figure1),
        ("figure2_planning_vs_feedback.png", figure2),
        ("figure3_model_mismatch.png", figure3),
        ("figure4_acceptance_ablation.png", figure4),
    ]
    for name, fn in jobs:
        out = args.out_dir / name
        fn(args.raw_dir, out)
        print(f"{'저장' if out.exists() else '건너뜀'}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
