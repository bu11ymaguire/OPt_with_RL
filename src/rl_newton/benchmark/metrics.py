"""cost-to-target 집계와 쌍별 통계.

두 가지를 강제한다
------------------
**절단 규칙 (프로토콜 D6).** 예산 내 목표 미달 run 을 버리거나 최댓값으로
대입하지 않는다. ``success_rate`` 와 도달한 run 의 **중앙값**을 항상 함께
보고한다. 하나만 보면 왜곡된다. 예를 들어 아주 공격적인 설정은 절반이
발산하지만 성공한 절반은 매우 빠를 수 있다. 중앙값만 보면 최고로 보인다.

**쌍별 비교 (프로토콜 D7).** ``mean +- std`` 는 쓰지 않는다. n=5 에서
무의미하고 정규성 가정도 없다. 대신 같은 ``(task, seed)`` 쌍에서의 비율을
모아 기하평균과 부트스트랩 CI 를 내고, Wilcoxon signed-rank 로 검정한다.

CG 수렴과 최적화 진행을 분리한다
--------------------------------
높은 damping 은 ``(H + lambda I)^{-1} g ~ g / lambda`` 로 CG 를 쉽게 만들지만
실제 loss 감소를 느리게 한다. 실측에서 damping ``1e6`` 은 CG 를 30/30 수렴시키고도
최종 loss 가 damping ``1e-2`` 보다 1500배 나빴다. 따라서 두 지표를 절대 섞지
않고 ``RunSummary`` 에 각각 기록한다.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # 타입 힌트 전용. 런타임 import 하면 ``optimizers`` 와 순환이 된다.
    from rl_newton.optimizers.newton_cg import OptimizationTrace

__all__ = [
    "TargetSpec",
    "RunSummary",
    "GroupSummary",
    "PairedComparison",
    "PairedDelta",
    "summarize_run",
    "summarize_group",
    "compare_paired",
    "compare_paired_delta",
    "recovery_ratio",
    "geometric_mean",
    "bootstrap_ci",
    "wilcoxon_signed_rank_p",
    "holm_adjust",
]

TargetMetric = Literal["relative_loss", "absolute_loss"]


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """사전 등록된 목표치 (프로토콜 D6).

    Attributes:
        metric: ``relative_loss`` 는 ``L / L_0``, ``absolute_loss`` 는 ``L``.
        value: 임계값. 이하가 되면 도달로 본다.
    """

    metric: TargetMetric = "relative_loss"
    value: float = 1.0e-6

    def reached(self, loss: float, initial_loss: float) -> bool:
        if not math.isfinite(loss):
            return False
        if self.metric == "absolute_loss":
            return loss <= self.value
        if initial_loss <= 0.0 or not math.isfinite(initial_loss):
            return False
        return loss / initial_loss <= self.value

    @property
    def label(self) -> str:
        return f"{self.metric}<={self.value:g}"


@dataclass(slots=True)
class RunSummary:
    """run 하나의 집계. 도달 여부와 비용을 분리해서 담는다."""

    run_id: str
    controller: str
    task_instance_id: str
    seed: int
    target: str

    reached: bool
    cost_to_target_ge: float | None
    """도달했을 때 소모한 누적 GE. 미도달이면 ``None``. 절단 규칙의 핵심이다."""
    steps_to_target: int | None
    hvp_to_target: int | None

    initial_loss: float
    final_loss: float
    total_cost_ge: float
    total_hvp: int
    search_cost_ge: float
    n_steps: int
    stop_reason: str

    # --- 안정성 (loss 감소와 분리) ---
    rejection_rate: float
    failure_rate: float
    negative_curvature_rate: float
    cg_convergence_rate: float
    """CG 가 tolerance 를 만족한 step 비율. **최적화 성능이 아니다.**"""
    median_residual_ratio: float
    median_damping: float
    median_trust_ratio: float

    @property
    def log_improvement(self) -> float:
        """``log(L_0 / L_final)``. 총 감소 자릿수(nat)."""
        if self.initial_loss <= 0.0 or self.final_loss <= 0.0:
            return float("nan")
        return math.log(self.initial_loss) - math.log(self.final_loss)

    @property
    def log_improvement_per_ge(self) -> float:
        """GE 당 loss 감소. 프로토콜 D3 보상의 run 단위 대응물이다."""
        if self.total_cost_ge <= 0.0:
            return float("nan")
        return self.log_improvement / self.total_cost_ge


def _median(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.median(finite) if finite else float("nan")


def budget_respecting_prefix(
    trace: OptimizationTrace, budget: float | None
) -> tuple[float, float, int]:
    """누적비용이 ``budget`` 을 넘지 않는 마지막 지점. ``(loss, 누적비용, step 수)``.

    **Track E 공정성 수정 (프로토콜 D11).** optimizer 루프는
    ``spent >= budget`` 에서 종료하므로 **마지막 step 이 예산을 초과한다.**
    초과량은 컨트롤러가 고른 action 에 비례하므로 비교가 불공정해진다.

    ```text
    C0 (평균 k=17.9)  150 GE 예산에 실제 소모 171 GE   <- 공짜로 큰 step 하나
    Q=4 (평균 k=3.3)  150 GE 예산에 실제 소모 154 GE
    ```

    큰 step 을 고르는 컨트롤러가 최대 한 step 만큼 예산을 더 쓴다. 고정 예산
    비교에서 이것은 그대로 이득이 된다. 그래서 집계 시 **예산을 넘지 않는
    마지막 prefix** 에서 잘라 평가한다. 모든 컨트롤러의 ``total_cost_ge`` 가
    예산 이하가 되므로 planner 의 쿼터 회계와도 의미가 일치한다.

    optimizer 의 동역학은 바꾸지 않는다. 절단된 step 들은 raw trace 에 남아
    있으므로 필요하면 다시 볼 수 있다.

    Args:
        trace: 실행 기록.
        budget: GE 예산. ``None`` 이면 절단하지 않는다.

    Returns:
        ``(prefix 최종 loss, prefix 누적비용, prefix step 수)``.
        어떤 step 도 예산에 들어가지 않으면 ``(초기 loss, 0.0, 0)``.
    """
    if budget is None:
        return trace.final_loss, trace.total_cost_ge, trace.n_steps
    spent = 0.0
    loss = trace.initial_loss
    steps = 0
    for record in trace.records:
        if not math.isfinite(record.cost_ge):
            break
        if spent + record.cost_ge > budget:
            break
        spent += record.cost_ge
        loss = record.train_loss_after
        steps += 1
        if not math.isfinite(loss):
            # NaN 이 난 step 은 그 자체가 결과다. 여기서 멈춘다.
            break
    return loss, spent, steps


def summarize_run(
    trace: OptimizationTrace, target: TargetSpec, *, budget_ge: float | None = None
) -> RunSummary:
    """``OptimizationTrace`` 를 집계한다.

    cost-to-target 은 목표에 처음 도달한 step 까지의 **누적** GE 다. 도달하지
    못하면 ``None`` 이며, 이를 큰 값으로 대체하지 않는다 (프로토콜 D6).

    Args:
        trace: 실행 기록.
        target: 목표 규격 (Track T).
        budget_ge: 주면 Track E 지표(``final_loss``, ``total_cost_ge``,
            ``n_steps``)를 예산을 넘지 않는 prefix 에서 평가한다
            (``budget_respecting_prefix`` 참조). Track T 지표는 목표 도달
            시점으로 정의되므로 영향받지 않는다.
    """
    cumulative = trace.cumulative_cost_ge()
    reached = False
    cost_to_target: float | None = None
    steps_to_target: int | None = None
    hvp_to_target: int | None = None

    hvp_running = 0
    for i, record in enumerate(trace.records):
        hvp_running += record.hvp_count
        if target.reached(record.train_loss_after, trace.initial_loss):
            reached = True
            cost_to_target = cumulative[i]
            steps_to_target = i + 1
            hvp_to_target = hvp_running
            break

    n = max(len(trace.records), 1)
    residuals = [float(r.extra.get("cg_residual_ratio", float("nan"))) for r in trace.records]
    final_loss, total_cost, n_steps = budget_respecting_prefix(trace, budget_ge)
    return RunSummary(
        run_id=trace.run_id,
        controller=trace.controller,
        task_instance_id=trace.task_instance_id,
        seed=trace.seed,
        target=target.label,
        reached=reached,
        cost_to_target_ge=cost_to_target,
        steps_to_target=steps_to_target,
        hvp_to_target=hvp_to_target,
        initial_loss=trace.initial_loss,
        final_loss=final_loss,
        total_cost_ge=total_cost,
        total_hvp=trace.total_hvp,
        search_cost_ge=trace.search_cost_ge,
        n_steps=n_steps,
        stop_reason=trace.stop_reason,
        rejection_rate=trace.n_rejected / n,
        failure_rate=trace.n_failures / n,
        negative_curvature_rate=trace.n_negative_curvature / n,
        cg_convergence_rate=trace.n_cg_converged / n,
        median_residual_ratio=_median(residuals),
        median_damping=_median([r.damping for r in trace.records]),
        median_trust_ratio=_median([r.trust_ratio for r in trace.records]),
    )


@dataclass(slots=True)
class GroupSummary:
    """한 컨트롤러의 여러 인스턴스에 대한 집계.

    ``success_rate`` 와 ``median_cost_to_target_ge`` 를 **항상 함께** 본다
    (프로토콜 D6). 하나만 보면 왜곡된다.
    """

    controller: str
    target: str
    n_runs: int
    n_reached: int
    median_cost_to_target_ge: float
    """도달한 run 만의 중앙값. 미도달 run 은 여기 포함되지 않는다."""
    median_final_loss_ratio: float
    """``L_final / L_0`` 의 중앙값. 전체 run 대상."""
    median_log_improvement: float
    median_total_cost_ge: float
    median_search_cost_ge: float
    mean_rejection_rate: float
    mean_failure_rate: float
    mean_cg_convergence_rate: float
    median_damping: float
    runs: list[RunSummary] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.n_reached / self.n_runs if self.n_runs else float("nan")

    def describe(self) -> str:
        cost = (
            f"{self.median_cost_to_target_ge:.1f}"
            if math.isfinite(self.median_cost_to_target_ge)
            else "미도달"
        )
        return (
            f"{self.controller}: 도달률 {self.success_rate:.0%} "
            f"({self.n_reached}/{self.n_runs}), cost-to-target 중앙값 {cost} GE"
        )


def summarize_group(runs: Sequence[RunSummary], *, controller: str | None = None) -> GroupSummary:
    """여러 run 을 컨트롤러 단위로 집계한다."""
    if not runs:
        raise ValueError("runs must not be empty")
    name = controller or runs[0].controller
    reached = [r for r in runs if r.reached and r.cost_to_target_ge is not None]
    ratios = [
        r.final_loss / r.initial_loss
        for r in runs
        if r.initial_loss > 0.0 and math.isfinite(r.final_loss)
    ]
    return GroupSummary(
        controller=name,
        target=runs[0].target,
        n_runs=len(runs),
        n_reached=len(reached),
        median_cost_to_target_ge=_median([float(r.cost_to_target_ge) for r in reached]),
        median_final_loss_ratio=_median(ratios),
        median_log_improvement=_median([r.log_improvement for r in runs]),
        median_total_cost_ge=_median([r.total_cost_ge for r in runs]),
        median_search_cost_ge=_median([r.search_cost_ge for r in runs]),
        mean_rejection_rate=_median([r.rejection_rate for r in runs]),
        mean_failure_rate=_median([r.failure_rate for r in runs]),
        mean_cg_convergence_rate=_median([r.cg_convergence_rate for r in runs]),
        median_damping=_median([r.median_damping for r in runs]),
        runs=list(runs),
    )


# ---------------------------------------------------------------------------
# 통계 (프로토콜 D7)
# ---------------------------------------------------------------------------


def geometric_mean(values: Sequence[float]) -> float:
    """양수 값들의 기하평균. 비율 비교의 표준 요약이다."""
    positive = [v for v in values if math.isfinite(v) and v > 0.0]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: str = "geometric_mean",
) -> tuple[float, float]:
    """비율 표본의 부트스트랩 신뢰구간.

    Args:
        values: 쌍별 비율들.
        n_boot: 재표본 횟수.
        confidence: 신뢰수준.
        seed: 결정론적 재현을 위한 시드.
        statistic: ``geometric_mean`` 또는 ``median``.

    Returns:
        ``(lower, upper)``. 표본이 부족하면 ``(nan, nan)``.
    """
    import random

    clean = [v for v in values if math.isfinite(v) and v > 0.0]
    if len(clean) < 2:
        return float("nan"), float("nan")

    rng = random.Random(seed)
    stat_fn = geometric_mean if statistic == "geometric_mean" else _median
    samples: list[float] = []
    n = len(clean)
    for _ in range(n_boot):
        resample = [clean[rng.randrange(n)] for _ in range(n)]
        samples.append(stat_fn(resample))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[int(alpha * len(samples))]
    hi = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    return lo, hi


def wilcoxon_signed_rank_p(differences: Sequence[float]) -> float:
    """양측 Wilcoxon signed-rank 검정의 p-value.

    ``scipy.stats.wilcoxon`` 을 쓴다. 0 차이는 제외한다(Wilcoxon 관례).
    표본이 부족하면 ``nan``.
    """
    from scipy import stats

    nonzero = [d for d in differences if math.isfinite(d) and d != 0.0]
    if len(nonzero) < 5:
        # n<5 에서는 양측 검정이 유의수준 0.05 에 도달할 수 없다.
        return float("nan")
    try:
        return float(stats.wilcoxon(nonzero).pvalue)
    except ValueError:
        return float("nan")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni 보정. 주 가설 3개에 적용한다 (프로토콜 D7)."""
    indexed = [(p, i) for i, p in enumerate(p_values)]
    finite = [(p, i) for p, i in indexed if math.isfinite(p)]
    finite.sort()
    m = len(finite)
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(finite):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[i] = running
    return adjusted


@dataclass(slots=True)
class PairedComparison:
    """두 컨트롤러의 쌍별 비교 결과.

    비율은 ``baseline / treatment`` 로 정의한다. 즉 **1보다 크면 treatment 가
    싸다(좋다)**. cost-to-target 은 작을수록 좋기 때문이다.
    """

    baseline: str
    treatment: str
    metric: str
    n_pairs: int
    n_both_reached: int
    ratio_geometric_mean: float
    ratio_ci: tuple[float, float]
    p_value: float
    baseline_median: float
    treatment_median: float

    @property
    def improvement_pct(self) -> float:
        """treatment 가 baseline 대비 몇 % 절감했는가."""
        if not math.isfinite(self.ratio_geometric_mean) or self.ratio_geometric_mean <= 0:
            return float("nan")
        return (1.0 - 1.0 / self.ratio_geometric_mean) * 100.0

    def describe(self) -> str:
        ci = self.ratio_ci
        p = f"{self.p_value:.4f}" if math.isfinite(self.p_value) else "n/a"
        return (
            f"{self.treatment} vs {self.baseline} [{self.metric}]: "
            f"비율 {self.ratio_geometric_mean:.3f}x "
            f"(95% CI {ci[0]:.3f}-{ci[1]:.3f}), "
            f"절감 {self.improvement_pct:.1f}%, p={p}, "
            f"쌍 {self.n_both_reached}/{self.n_pairs}"
        )


def compare_paired(
    baseline: Sequence[RunSummary],
    treatment: Sequence[RunSummary],
    *,
    metric: str = "cost_to_target_ge",
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedComparison:
    """같은 ``(task_instance_id, seed)`` 쌍끼리 비교한다 (프로토콜 D7).

    Args:
        baseline: 기준 컨트롤러의 run 들.
        treatment: 비교 대상의 run 들.
        metric: ``cost_to_target_ge`` | ``final_loss_ratio`` | ``total_cost_ge``.
        n_boot: 부트스트랩 재표본 횟수.
        seed: 부트스트랩 시드.

    Returns:
        ``PairedComparison``. 쌍이 없으면 ``nan`` 으로 채워진다.

    Raises:
        ValueError: 알 수 없는 metric.
    """
    if metric not in ("cost_to_target_ge", "final_loss_ratio", "total_cost_ge"):
        raise ValueError(f"unknown metric: {metric!r}")

    def key(r: RunSummary) -> tuple[str, int]:
        return r.task_instance_id, r.seed

    def value(r: RunSummary) -> float | None:
        if metric == "cost_to_target_ge":
            return r.cost_to_target_ge if r.reached else None
        if metric == "final_loss_ratio":
            if r.initial_loss <= 0.0 or not math.isfinite(r.final_loss):
                return None
            return max(r.final_loss / r.initial_loss, 1e-300)
        return r.total_cost_ge

    base_map = {key(r): r for r in baseline}
    treat_map = {key(r): r for r in treatment}
    shared = sorted(set(base_map) & set(treat_map))

    ratios: list[float] = []
    diffs: list[float] = []
    base_values: list[float] = []
    treat_values: list[float] = []
    for k in shared:
        b = value(base_map[k])
        t = value(treat_map[k])
        if b is None or t is None or b <= 0.0 or t <= 0.0:
            continue
        ratios.append(b / t)
        diffs.append(b - t)
        base_values.append(b)
        treat_values.append(t)

    return PairedComparison(
        baseline=baseline[0].controller if baseline else "?",
        treatment=treatment[0].controller if treatment else "?",
        metric=metric,
        n_pairs=len(shared),
        n_both_reached=len(ratios),
        ratio_geometric_mean=geometric_mean(ratios),
        ratio_ci=bootstrap_ci(ratios, n_boot=n_boot, seed=seed),
        p_value=wilcoxon_signed_rank_p(diffs),
        baseline_median=_median(base_values),
        treatment_median=_median(treat_values),
    )


@dataclass(slots=True)
class PairedDelta:
    """높을수록 좋은 지표의 쌍별 **차이** 비교.

    Track E(고정 GE 예산)의 헤드룸은 비율이 아니라 차이로 정의된다
    (프로토콜 D9).

    ```text
    H_E = J_E(planner) - J_E(best_static)      [nat]
    ```

    비율을 쓰면 ``J_E`` 가 0에 가까울 때 폭발하고, nat 단위의 해석
    ("몇 배 loss 차이")도 잃는다.
    """

    baseline: str
    treatment: str
    metric: str
    n_pairs: int
    n_valid: int
    median_delta: float
    """중앙값 차이. 양수면 treatment 가 좋다."""
    delta_ci: tuple[float, float]
    p_value: float
    baseline_median: float
    treatment_median: float

    @property
    def loss_ratio_equivalent(self) -> float:
        """``exp(median_delta)``. "loss 몇 배 차이" 로 읽을 수 있다."""
        if not math.isfinite(self.median_delta):
            return float("nan")
        return math.exp(self.median_delta)

    def describe(self) -> str:
        ci = self.delta_ci
        p = f"{self.p_value:.4f}" if math.isfinite(self.p_value) else "n/a"
        return (
            f"{self.treatment} vs {self.baseline} [{self.metric}]: "
            f"차이 {self.median_delta:+.3f} nat "
            f"(95% CI {ci[0]:+.3f}~{ci[1]:+.3f}), "
            f"loss {self.loss_ratio_equivalent:.2f}배, p={p}, "
            f"쌍 {self.n_valid}/{self.n_pairs}"
        )


def _bootstrap_median_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """차이 표본의 중앙값 부트스트랩 CI. 음수를 허용한다."""
    import random

    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(clean)
    samples = [
        statistics.median([clean[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    ]
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[int(alpha * len(samples))]
    hi = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    return lo, hi


def compare_paired_delta(
    baseline: Sequence[RunSummary],
    treatment: Sequence[RunSummary],
    *,
    metric: str = "log_improvement",
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedDelta:
    """같은 ``(task_instance_id, seed)`` 쌍에서 **차이**를 비교한다.

    Track E 용이다. ``metric`` 은 높을수록 좋은 지표여야 한다.

    Args:
        baseline: 기준 컨트롤러의 run 들.
        treatment: 비교 대상.
        metric: ``log_improvement`` | ``log_improvement_per_ge``.
        n_boot: 부트스트랩 재표본 횟수.
        seed: 부트스트랩 시드.

    Raises:
        ValueError: 알 수 없는 metric.
    """
    if metric not in ("log_improvement", "log_improvement_per_ge"):
        raise ValueError(f"unknown metric: {metric!r}")

    def key(r: RunSummary) -> tuple[str, int]:
        return r.task_instance_id, r.seed

    def value(r: RunSummary) -> float:
        return getattr(r, metric)

    base_map = {key(r): r for r in baseline}
    treat_map = {key(r): r for r in treatment}
    shared = sorted(set(base_map) & set(treat_map))

    deltas: list[float] = []
    base_values: list[float] = []
    treat_values: list[float] = []
    for k in shared:
        b = value(base_map[k])
        t = value(treat_map[k])
        if not (math.isfinite(b) and math.isfinite(t)):
            continue
        deltas.append(t - b)
        base_values.append(b)
        treat_values.append(t)

    return PairedDelta(
        baseline=baseline[0].controller if baseline else "?",
        treatment=treatment[0].controller if treatment else "?",
        metric=metric,
        n_pairs=len(shared),
        n_valid=len(deltas),
        median_delta=_median(deltas),
        delta_ci=_bootstrap_median_ci(deltas, n_boot=n_boot, seed=seed),
        p_value=wilcoxon_signed_rank_p(deltas),
        baseline_median=_median(base_values),
        treatment_median=_median(treat_values),
    )


def recovery_ratio(static: float, learned: float, oracle: float) -> float:
    """학습된 컨트롤러가 도달 가능한 헤드룸의 몇 %를 회수했는가 (프로토콜 게이트 E).

    ```text
    Recovery = (J_learned - J_static) / (J_oracle - J_static)
    ```

    ``J`` 는 클수록 좋은 지표여야 한다 (예: ``log_improvement_per_ge``).
    비용처럼 작을수록 좋은 지표라면 부호를 뒤집어 넣는다.

    **분모의 오라클은 absolute 가 아니라 정책과 같은 행동 공간을 쓰는
    reachable 오라클이어야 한다.** absolute 오라클을 분모에 두면 정책이
    구조적으로 도달할 수 없는 부분까지 요구하게 되어 불공정하다.

    Returns:
        회수율. 오라클과 static 이 같으면 ``nan`` (헤드룸이 없어 정의 불가).
    """
    span = oracle - static
    if not math.isfinite(span) or abs(span) < 1e-300:
        return float("nan")
    return (learned - static) / span
