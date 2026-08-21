# When Does Feedback Help? Planning and Model Mismatch in Hessian-Free Newton Control

**초안 상태.** 이 문서는 `paper/claim_ledger.md` 에 등록된 주장만 쓴다. 새 해석을
추가하지 않았다. Abstract / Introduction / Discussion / Conclusion 의 주장 강도는
리뷰어 검토 대상이다.

`Planning Is Not Feedback` 은 Discussion 소제목으로만 쓴다. 첫 arXiv 원고 제목으로는
단정적이다.

모든 수치는 `docs/results_stage2.md` 에서 왔고 출처는 `paper/evidence_map.md` 의
SHA-256 으로 고정된다. **본문에서 숫자를 손으로 고치지 않는다.**

---

## Abstract

*(리뷰어 검토 필요. 다섯 요소만 담았다: 문제 / 분해 / planning 결과 / feedback 결과
/ stochastic 결과와 정책 학습 미진행.)*

<!-- CLAIM: scope -->
Hessian-free Newton-CG [@martens2010hessianfree; @nash2000survey] 에서 damping 과 CG
반복 예산의 배분을 **순차적 의사결정 문제로 구성한다.** 그 이득이 다단계 계획에서
오는지 실행 중 상태 피드백에서 오는지를 분리해 측정하는 것이 목적이다.

<!-- CLAIM: C07 -->
고정 예산(150 gradient-equivalent, GE) 아래에서 상수 설정, 자원 시계 open-loop
스케줄, 1-step greedy, 초기 상태에서 한 번 계획하고 그대로 실행하는 committed
planner, 매 step 재계획하는 planner 를 같은 사다리에서 비교했다. 설정은 dev seed 에서
한 번 고정하고 분리된 held-out seed 에서 측정했다.

<!-- CLAIM: C03 -->
**차원 `d=100` 의 ill-conditioned 이면서 양정인(SPD) quadratic 네 족
(`κ ∈ {10³, 10⁴, 10⁵, 10⁶}`) × held-out seed 10개, 40 인스턴스**에서, 다단계
lookahead 는 1-step greedy 대비 `+0.456 nat`(95% CI `+0.254 ~ +0.720`, `p<0.0001`,
35/40 인스턴스) 개선했다.

<!-- CLAIM: C04 -->
같은 40 인스턴스에서 실행 중 재계획은 committed planner 대비 `+0.010 nat`
(95% CI `−0.033 ~ +0.053`, `p=0.97`, 21/40)였다. 신뢰구간이 좁고 0 을 포함했으며
**실용적으로 큰 feedback 이득은 관측되지 않았다.**

<!-- CLAIM: C12 -->
탐색적 확장으로, 같은 모델과 데이터에 대해 optimizer 가 보는 표본만 바꾼
micro-neural 문제를 결정론적 full-batch 와 minibatch regime 에서 비교했다
(`n=3` per regime). minibatch regime 에서는 낡은 계획을 고수하는 것이 큰 손해였고
재계획이 그 손해를 줄였으나, **planner 는 값싼 1-step 제어를 안정적으로 이기지
못했다.** 정책 학습은 사전 등록한 조건부 다음 단계였고, 이 증거로는 그 단계로
진행하지 않았다.

---

## 1. Introduction

Truncated-Newton 또는 inexact Newton 방법은 Hessian 을 만들지 않고 Newton 방정식을
반복법으로 근사해 푼다 [@dembo1982inexact; @steihaug1983cg; @nash1984lanczos;
@nash2000survey]. 곡률 접근을 Hessian-vector product 로만 하는 구현은
[@pearlmutter1994hvp] 가 제시한 정확 HVP 에 기반하며, 심층망 규모에서의 적용은
[@martens2010hessianfree; @martens2011rnn] 가 보고했다. 실제 성능은 두 계산 자원
결정에 크게 의존한다.

```text
damping (Levenberg-Marquardt 유형 정칙화)
step 당 CG 반복 예산
```

damping 은 Levenberg-Marquardt 정칙화 [@levenberg1944; @marquardt1963] 계열이고
trust-region 반경과 대응 관계가 있다 [@steihaug1983cg; @conn2000trustregion].
Hessian-free 최적화에서 damping 조절 규칙이 성능을 좌우한다는 보고가 있다
[@martens2010hessianfree; @martens2011rnn]. CG 반복 예산의 절단은 inexact Newton 의
핵심 설계 변수다 [@dembo1982inexact; @nash2000survey].

이 두 값을 고정하지 않고 최적화 도중 조절하려는 시도는 learned optimizer 문헌과
맞닿아 있다 [@andrychowicz2016l2l; @metz2019pathologies; @metz2020effective;
@bae2022apo]. 강화학습으로 그 조절 정책을 학습하려면 [@schulman2017ppo] 상태를
관찰하고 그에 따라 행동을 바꾸는 것에 가치가 있어야 한다. 계획을 미리 정해 두는 것과
실행 중 상태에 반응하는 것의 구분은 open-loop 대 closed-loop 제어의 고전적 구분이다
[@bertsekas2017dp].

우리의 질문은 성능 향상 여부가 아니라 **그 향상이 어디서 오는가**다.

> Hessian-free Newton-CG 에서 상태 의존적 계산 자원 제어의 이득은 다단계 planning
> 에서 발생하는가, 실행 중 feedback 에서 발생하는가?

이 질문을 검증 가능한 두 명제로 나눈다.

```text
[Q1] 좋은 action sequence 가 존재하는가
[Q2] 그 sequence 를 실행 중 feedback 으로 수정할 가치가 있는가
```

두 질문은 서로 다른 대상을 함의한다. `Q1` 만 성립하면 필요한 것은 초기 상태에서
스케줄을 정하는 예측기이고, `Q2` 가 성립해야 step 단위 feedback 정책이 정당화된다.

### 기여

*(강도 조정은 리뷰어 검토 대상이다.)*

> 1. We operationalize the control of damping and CG effort as a sequential decision
>    problem.
> 2. We present a controller ladder that separates multi-step planning from
>    execution-time feedback.
> 3. On held-out instances of ill-conditioned SPD quadratics, we confirm an additional
>    value for multi-step planning (`+0.456 nat`, CI `+0.254 ~ +0.720`, `p<0.0001`,
>    35/40) and do not observe a practically large additional value for feedback
>    (`+0.010 nat`, CI `−0.033 ~ +0.053`).
> 4. We document and apply an audit procedure required to make our controller
>    comparisons identifiable.

`1` 을 "새로운 MDP formulation" 처럼 쓰지 않는다. `4` 를 "general benchmark
framework" 라고 쓰지 않는다. **보조 기여**이며 본문에서는 Methods 의 일부(`§5`)로
두고 Discussion `§13.3` 에서 회수한다.

정책 학습을 진행하지 않은 결정은 기여 목록에 넣지 않는다. `§12` 에서 **과학적
결론과 프로젝트 go/no-go 결정을 분리해** 서술한다.

---

## 2. Problem setup and cost accounting

### 2.1 Newton-CG step 과 행동 공간

각 step 에서 damped Newton 방정식을 truncated conjugate gradient [@hestenes1952cg;
@steihaug1983cg] 로 푼다. 행렬은 만들지 않고 HVP 만 쓴다 [@pearlmutter1994hvp].

```text
(H + λI) p = −g,   CG 반복 k 회에서 절단
```

컨트롤러가 고르는 행동은 두 축이다.

```text
damping 배수 m      λ <- clip(λ · m)   로그 공간에서 누적된다
CG 반복 예산 k      k ∈ {3, 5, 10, 20}
```

`step_size` 는 `1.0` 으로 고정했다. damping 축과 aliasing 되어 두 축의 기여를
분리할 수 없기 때문이다.

행동 공간 세 가지를 쓴다.

```text
narrow    12 action.  damping 배수 log10 범위 0.95.  배포 대상
wide      28 action.  범위 2.86
absolute 132 action.  범위 15.27.  도달성 제약을 없앤 분석용
```

`absolute` 는 `narrow` 와 **로그 해상도가 같다.** 그래야 "행동 공간이 병목인가" 라는
비교가 해상도 차이로 오염되지 않는다.

### 2.2 비용 회계와 그 범위

wall-clock 을 주 지표로 쓰지 않는다. 이 규모(수천~수만 파라미터)에서는 GPU 시간이
FLOP 이 아니라 커널 런치와 인터프리터 오버헤드에 지배되므로, 측정값이 최적화 효율이
아니라 구현 오버헤드를 반영한다.

대신 하드웨어 독립 단위를 쓴다.

```text
1 GE = gradient batch 1회 forward + backward
cost_GE(k) = c_grad_graph + k · c_hvp + c_fwd
```

모든 비교는 **동일 GE 예산**에서 한다. Newton-CG 는 step 비용이 행동에 따라 크게
달라지므로(`k=3` 과 `k=20` 은 6배 이상), step 수를 맞추고 최종 loss 를 비교하면 비용이
다른 것들을 비교하게 된다.

**이 정의의 범위를 미리 밝힌다.**

> GE normalizes gradient-equivalent oracle calls within a regime, not total
> floating-point operations across different batch sizes. Cross-regime absolute
> performance comparisons are therefore descriptive rather than compute-normalized.

`§7` 의 quadratic 비교는 모두 같은 목적함수 위에서 이루어지므로 이 한계와 무관하다.
`§9` 의 micro-neural regime 간 절대값 비교에만 해당한다.

### 2.3 실행 환경과 GE 의 성격

<!-- CLAIM: C06 -->
> All Stage 2 optimization and planning experiments were executed on CPU. The
> GPU-derived cost-model files in the repository originate from an earlier Stage 1
> calibration and were not used in Stage 2; Stage 2 GE accounting used HVP-equivalent
> oracle counts.

저장소는 Stage 1 설정 파일을 `device: cuda:0` 항목까지 provenance 로 보존한다. 이
파일들은 Stage 2 의 실행 환경을 서술하지 않는다.

탐색 비용을 읽을 때도 구분이 필요하다.

> Search GE measures simulated oracle work rather than wall-clock GPU compute.

`§7.6` 의 `1,294배` 는 planner 가 수행한 **oracle 호출량**의 비율이다. GPU wall-clock
비율이 아니다.

### 2.4 평가 지표

고정 예산에서의 개선을 nat 단위 로그 개선으로 잰다.

```text
J_E = log L_0 − log L_final
```

수치 하한을 `100 · eps` 상대값으로 두고 그 아래 값을 cap 한다. cap 되지 않은 쌍과
전체 쌍을 함께 보고한다. 쌍이 통계에서 빠지면 반드시 사유를 기록한다.

---

## 3. Controllers and oracles

사다리를 아래에서 위로 정의한다. 모든 컨트롤러가 같은 GE 예산을 받는다.

```text
best_static        상수 action. 후보 6개를 dev 에서 튜닝해 고른 결과
best_open_loop     4구간 스케줄. 구간 경계는 소모 GE 비율로 정한다. 상태 미관측
heuristic          규칙 기반 damping 조절
onestep (C0)       매 step 후보를 전수 평가해 즉시 효율이 최대인 행동을 고른다
committed          초기 상태에서 계획을 한 번 세우고 **그대로 실행**한다
shrinking          매 step 남은 쿼터로 재계획한다 (shrinking-horizon MPC)
```

`shrinking` 은 model predictive control 의 receding-horizon 구조를 따르되 남은 예산이
줄어들므로 horizon 이 축소된다 [@rawlings2017mpc]. `committed` 와 `shrinking` 의 대비가
open-loop 계획과 closed-loop 정책의 구분에 해당한다 [@bertsekas2017dp].

### 3.1 `committed` 가 `best_open_loop` 와 다른 점

둘 다 실행 중 상태를 보지 않는다. 그러나 `committed` 는 **그 인스턴스의 초기 상태에
조건화된 oracle** 이고 `best_open_loop` 는 인스턴스 집합에서 튜닝된 단일 스케줄이다.

이 구분이 `Q2` 의 핵심이다. `shrinking − committed` 가 0 에 가깝다는 것은 "스케줄로
충분하다" 가 아니라 **"초기 상태에서 정한 계획으로 충분하다"** 를 뜻한다.

### 3.2 `best_static` 과 `best_open_loop` 는 컨트롤러가 아니라 선택 결과다

두 baseline 은 튜닝 산물이므로 선택 근거를 manifest 로 남긴다. 후보 목록, 후보별
점수, 선택 지표, tie-break 규칙, 선택된 라벨을 모두 기록한다. held-out 에서 선택된
설정은 각각 `static[2]` 와 `open_loop[4]` 였다.

튜닝 비용도 보고한다. `best_open_loop` 는 인스턴스당 약 `878 GE` 를 썼다. 배포 예산이
`150 GE` 이므로 이 baseline 도 공짜가 아니다.

---

## 4. Experimental protocol

### 4.1 Paired design

`seed` 는 난수 시드가 아니라 **실험 조건의 이름**이다. `seed = s` 일 때 모든
컨트롤러가 동일한 인스턴스, 동일한 초기점, 동일한 minibatch 순서를 본다. task 생성용
난수 스트림을 optimizer 실행 스트림과 완전히 분리했다. 그래야 컨트롤러가 난수를
얼마나 쓰든 인스턴스가 같다. 이것은 문헌에서 가져온 주장이 아니라 우리 실행기의 설계
사실이며 `§4.3` 의 bitwise 재현 검사로 확인했다.

통계는 쌍별 차이에 대한 Wilcoxon signed-rank 검정 [@wilcoxon1945] 과 중앙값 부트스트랩
CI [@efron1979bootstrap] 를 쓴다. `p` 는 `0.0000` 으로 쓰지 않고 `p<0.0001` 로 쓴다.

### 4.2 Seed 역할 분리

```text
CALIBRATION_SEEDS   0, 1          benchmark spec 선정에만
SELECTION_SEEDS     2, 3, 4       설정 선택에만
HELD_OUT_SEEDS      100 ~ 109     최종 효과 추정에만
```

**설정은 held-out 결과로 고르지 않았다.** `§6` 에서 dev seed 로 한 번 고르고
`§7` 에서 다시 열지 않았다.

남은 중복 하나를 밝힌다. `quad_d100_k1e5` 가 초기 dev subset 과 최종 spec 집합에 모두
있어 `(quad_d100_k1e5, seed 2)` 가 두 국면에서 같은 인스턴스다. 설정 선택 12 인스턴스
중 1개다.

### 4.3 결과 정체성

집계 코드나 문서를 고쳐도 optimizer 가 재실행되지 않아야 한다. 정체성을 세 층으로
분리했다.

```text
run_semantics_id   이 컨트롤러가 실제 쓰는 optimizer 설정만
sweep_id           이번 실행이 요청한 run 집합
aggregation_id     집계 정책
```

`git_commit` 과 `code_dirty` 는 어떤 식별자에도 넣지 않고 별도 provenance 로
기록한다. 문서만 수정해도 해시가 바뀌면 "어떤 집합을 요청했는가" 라는 의미가 깨진다.

이 분리가 실제로 유효한지 확인하려고, 정체성·로깅·baseline 시계 변경 전후의 planner
궤적을 비교했다. 동일 CPU, 단일 스레드, 동일 seed 이므로 기대값은 bitwise exact 다.
사전 등록한 72쌍 전부가 bitwise 일치했다.

### 4.4 사전 등록한 게이트

```text
A1  1-step 절대 행동 공간의 순간 헤드룸        GO ≥ 1.0 nat
A2  현실적 배수 행동으로 그 이득에 접근 가능    GO ≥ 0.7 nat
B   행동 공간이 병목인가                      GO ≥ 0.5 nat
C1  쿼터를 매 step 초기화하면 나빠지는가        GO ≥ 0.3 nat  (진단 전용)
C2  다단계 재계획이 1-step 보다 나은가          GO ≥ 0.3 nat
C3  상태 관찰 재계획이 고정 실행보다 나은가     GO ≥ 0.3 nat
D   cost-to-target 헤드룸                     GO ≥ 1.2 배
```

`C2` 의 GO 판정에는 **두 조건이 모두** 필요하다. 개선이 있어야 하고, 동시에 깊이
`> 1` 의 계획이 실제로 채택되어야 한다. 개선만 있고 깊이가 계속 1 이면 planning 이
아니라 탐색량이 기여한 것이다.

`C1` 은 seed 1개 진단 baseline 이므로 판정에 쓰지 않는다.

---

## 5. Benchmark eligibility

### 5.1 수치 하한 기준이 실패하는 사례

<!-- CLAIM: C08 -->

초기 적격성 조건은 상한을 수치 하한으로 계산했다.

```text
ceiling = log L_0 − log(L_0 · 100·eps) ≈ 31.44 nat
```

이 기준으로 Rosenbrock(`d=5`, 표준 시작점)은 적격이었다. 네 baseline 의 median 로그
개선이 `1.8175 nat` 이고 상한까지 여유가 `29.62 nat` 로 보였다.

실제로는 네 baseline 이 **모두 정확히 `1.8175 nat`** 였다. 원인을 특정했다. 표준
시작점의 basin 에 strict 국소최소점이 있다.

문헌은 확장 Rosenbrock 함수의 **두 변종을 구별한다** [@kok2009rosenbrock]. 짝수 차원의
비결합 2D 합은 전역최소점만 갖고, 좌표가 결합된 변종은 `d ≥ 4` 에서 추가 정류점을
갖는다 [@shang2006rosenbrock; @kok2009rosenbrock]. 우리가 쓴 것은 후자다.

```text
L(x) = sum_{i=1}^{d-1} [ 100 (x_{i+1} − x_i²)² + (1 − x_i)² ]
```

아래 점은 이 정의와 `d=5` 에서 우리가 직접 수렴시켜 확인한 것이다. 문헌의 표를 옮긴
것이 아니다. 좌표 `x_1 = −0.962` 는 문헌이 보고하는 국소최소점의 대략적 위치와
부합한다.

```text
x*        (−0.96205102, 0.93573939, 0.88071360, 0.77787767, 0.60509367)
loss      3.930839434133
‖grad‖    1.06e−08
Hessian   최소 고유값 +0.595 (양정)
```

표준 시작점에서 L-BFGS 를 수렴시켜도 전역최소점이 아니라 이 점에 도달한다. 도달 가능한
로그 개선의 상한은 `log(24.2 / 3.930839) = 1.8175 nat` 이고, 관측된 baseline 값이
정확히 그것이다. 즉 **여유가 0 이었다.**

### 5.2 도달 가능 상한으로 교정

```text
L_ref        = min over reference runs started from the task's own initial point
J_achievable = log L_0 − log max(L_ref, L_floor)
```

<!-- CLAIM: C09 -->
참조 solver panel 을 쓴다. 단일 solver 는 금지한다. `κ=10⁶` quadratic 에서 실측 근거가
나왔다.

```text
lbfgs   final 1.007485e−06   ‖grad‖ 3.456e−02   미수렴
newton  final 3.059391e−32   ‖grad‖ 2.650e−16   수렴
sgd     final nan
```

L-BFGS 만 썼다면 `J_achievable` 을 약 `28.3 nat` 로 과소평가했을 것이다.

<!-- CLAIM: C10 -->
**다른 초기화의 결과는 상한을 올리지 않는다.** 컨트롤러는 항상 task 의 시작점에서
출발하므로 다른 basin 의 최적값은 도달 가능한 값이 아니다. 진단으로만 기록한다.
`(0.9, …, 0.9)` 에서 출발하면 Rosenbrock `d=5` 는 전역최소점에 도달하고, 그 값을
`L_ref` 에 넣으면 `J_achievable` 이 `2.58` 에서 `31.44` 로 부풀어 국소최소점에 갇힌
spec 이 적격으로 통과한다.

`planner` 계열은 panel 에 넣지 않는다. 넣으면 spec 선정에 planner 결과가 새어 든다.

### 5.3 seed 복제 검출

`initial_loss` 만 보면 우연히 같을 수 있으므로 시작점 벡터도 함께 본다. 이 검사로
Rosenbrock `d=5` 의 기본 설정이 시작점을 무작위화하지 않아 seed 2/3/4 가 **같은
인스턴스**였음을 발견했다. 24개 run 이 모든 컬럼에서 bitwise 동일했고, 따라서 그
진단의 `n=3` 은 실제로 `n=1` 이었다.

### 5.4 최종 적격 조건과 판정 결과

```text
failure_rate = 0
joint floor-hit rate ≤ 1/3
각 baseline median J_E ≥ 1 nat
J_achievable − median J_E ≥ 3 nat
수렴한 참조 solver 간 산포 ≤ 0.5 nat
seed 마다 실제로 다른 인스턴스
```

| spec | 판정 | `J_achievable` | 제한 요인 | 최소 여유 |
|---|---|---|---|---|
| `quad d=100, κ=10³` | 채택 | 31.44 | numerical floor | 13.71 |
| `quad d=100, κ=10⁴` | 채택 | 31.44 | numerical floor | 20.98 |
| `quad d=100, κ=10⁵` | 채택 | 31.44 | numerical floor | 21.88 |
| `quad d=100, κ=10⁶` | 채택 | 31.44 | numerical floor | 21.93 |
| `rosen d=5` | 탈락 | 1.82 | critical point | −0.00 |
| `rosen d=5` (무작위 시작) | 탈락 | 2.58 | critical point | 0.00 |
| `micro-neural full-batch` | 채택 | 31.44 | numerical floor | 11.68 |
| `micro-neural minibatch` | 채택 | 31.44 | numerical floor | 28.75 |

Rosenbrock 계열의 여유가 **정확히 0** 이다. 이 spec 들은 컨트롤러 구분력을 갖지
않는다. 우리는 이것을 "비선형 문제에서 방법이 실패했다" 로 해석하지 않는다.

---

## 6. Configuration selection

이 절의 결과는 **설정 선택 근거로만** 쓴다. 효과 추정이 아니다.

### 6.1 사전 등록한 승격 대상

beam 4 결과로 쿼터를 고르지 않았다. 어떤 조건에서 beam 4 가 `−4.21 nat`, beam 8 이
`+4.59 nat` 였으므로, 얕은 beam 으로 유망성을 판정하면 실제로 좋은 설정을 탈락시킨다.

```text
Q = 1   beam 8 불필요 (깊이 1만 가능)
Q = 2   narrow, wide 모두 재평가
Q = 4   narrow, wide 모두 재평가
```

### 6.2 선택 규칙

선택 통계를 실행 전에 하나로 고정했다.

```text
선택 통계   challenge 12 인스턴스에서 shrinking 자신의 median J_E. 최대화
tie-break   0.05 nat 이내 동률이면  decision-search GE → 작은 Q → narrow
```

baseline 과의 차이로 고르지 않았다. 가장 강한 baseline 의 순위가 표본에서
안정적이지 않았기 때문이다(두 baseline 이 동일 값, `p=0.91`). 불안정한 기준점으로
나누면 설정 선택이 baseline 잡음을 따라간다.

### 6.3 결과

| configuration | n | median `J_E` | decision-search GE |
|---|---|---|---|
| `shrinking_Q4_narrow` | 12 | **10.5306** | 193,894 |
| `shrinking_Q2_wide` | 12 | 10.3691 | 127,609 |
| `shrinking_Q2_narrow` | 12 | 10.3208 | 55,802 |
| `shrinking_Q4_wide` | 12 | 10.2508 | 441,612 |

2위와 `0.16 nat` 차이로 동률 허용 범위를 넘어 단독 선택됐다. tie-break 를 쓰지 않았다.

`Q4_wide` 가 `Q4_narrow` 보다 낮다. 쿼터가 커지면 탐색 가능 집합은 포함관계로
커지지만 **실현 성능은 비감소가 아니다.**

### 6.4 프로토콜 이탈

비용 측정을 위해 1 인스턴스 dry run(`quad d=100 κ=10³`, seed 2, 30 run)을 먼저 돌리고
그 게이트 표를 본 뒤 선택 규칙을 확정했다. `Q × space` 별 planner 순위는 열지
않았으나 순서가 "규칙 확정 → 실행" 이 아니었다.

이 때문에 beam 8 결과를 **설정 선택과 가설 정교화로만** 쓰고, 효과 추정은 별도
held-out seed 에서 수행한다.

---

## 7. Held-out confirmation

사전 고정한 `shrinking_Q4_narrow` 를 spec 4개 × seed 100–109 = **40 인스턴스**에
적용했다. 960 run, 실패 0, floor cap 0. **설정을 다시 고르지 않았다.**

### 7.1 사다리

Table 1. 절대 median `J_E` (nat).

| controller | κ=10³ | κ=10⁴ | κ=10⁵ | κ=10⁶ |
|---|---|---|---|---|
| `best_static` | 12.741 | 8.825 | 8.980 | 8.724 |
| `best_open_loop` | 13.656 | 9.332 | 9.251 | 9.086 |
| `heuristic` | 12.740 | 8.825 | 8.980 | 8.755 |
| `onestep_narrow` | 17.032 | 10.357 | 9.879 | 9.833 |
| `onestep_absolute` | 17.036 | 10.571 | 9.882 | 9.723 |
| `committed_Q4_narrow` | 19.417 | 11.345 | 10.205 | 10.123 |
| `shrinking_Q4_narrow` | 19.911 | 11.222 | 10.186 | 10.032 |

### 7.2 쌍별 차이

Table 2. `n=40`.

| 비교 | median | 95% CI | p | 양수 |
|---|---|---|---|---|
| `shrinking` − `best_static` (A2) | +1.690 | [+1.462, +2.368] | <0.0001 | 40/40 |
| `shrinking` − `onestep` (C2) | +0.456 | [+0.254, +0.720] | <0.0001 | 35/40 |
| `shrinking` − `committed` (C3) | +0.010 | [−0.033, +0.053] | 0.97 | 21/40 |
| `onestep_absolute` − `onestep_narrow` (B) | +0.005 | [−0.002, +0.026] | 0.59 | 23/40 |
| `onestep_wide` − `onestep_narrow` (B) | −0.001 | [−0.041, +0.005] | 0.36 | 18/40 |
| `best_open_loop` − `best_static` | +0.395 | [+0.350, +0.476] | <0.0001 | 40/40 |
| `heuristic` − `best_static` | −0.000 | [−0.000, +0.000] | 0.78 | 20/40 |

게이트 판정: `A1=GO  A2=GO  B=재설계  C1=판정불가  C2=GO  C3=재설계  D=GO`.

### 7.3 다단계 lookahead 는 1-step 을 이긴다

<!-- CLAIM: C03 -->

`C2 = +0.456 nat` 이고 CI 하한이 `+0.254` 로 GO 임계값 `0.3` 을 넘는다. P3 의 두
조건이 모두 충족됐다.

```text
개선            +0.456 nat, 35/40 양수
깊이 > 1 채택률  0.84
계획 깊이 상한 도달 0.00
```

계산 상한 때문에 쿼터를 다 쓰지 못한 step 이 없으므로 쿼터 사다리 비교가 훼손되지
않았다.

**dev 에서는 조건부였다**(`+0.251`, `p=0.0771`). `n=40` 에서 GO 로 승격됐다.

### 7.4 실행 중 재계획

<!-- CLAIM: C04 -->
`C3 = +0.010 nat`, 95% CI `[−0.033, +0.053]`, 21승 1무 18패. CI 폭이 `0.086 nat` 로
`A2` 의 `0.906 nat` 보다 한 자릿수 좁다.

> held-out 결과에서 feedback 효과는 `+0.010 nat` 였으며 95% CI 가
> `[−0.033, +0.053]` 으로 좁게 0 을 포함했다. 따라서 **실용적으로 큰 feedback 이득은
> 관측되지 않았다.**

우리는 등가성을 주장하지 않는다. equivalence margin 을 사전 등록하지 않았으므로
"효과가 0 이다" 나 "효과가 `0.053 nat` 보다 작다" 를 검정 결과로 말할 수 없다.
TOST 절차는 사전에 정한 경계를 요구하며 [@schuirmann1987tost], 그 경계를 미리 지정하는
것이 권고 사항이다 [@lakens2017equivalence].

### 7.5 행동 공간은 병목이 아니다

<!-- CLAIM: C05 -->
`absolute`(132 action, log10 범위 15.27)가 `narrow`(12 action, 범위 0.95) 대비 얻는
것은 `+0.005 nat`(CI `[−0.002, +0.026]`, `p=0.59`, 23/40) 이다. `wide − narrow` 는
`−0.001 nat`(CI `[−0.041, +0.005]`, `p=0.36`, 18/40) 다. damping 을 자유롭게 고를 수
있게 해도 1-step 성능이 오르지 않는다. **음의 결과지만 값싼 좁은 행동 공간을
정당화한다.**

게이트 B 의 GO 임계값은 `0.5 nat` 였고 관측값이 그보다 두 자릿수 작으므로 판정은
`재설계` 다. CI 가 0 을 포함하지만 등가성을 주장하지 않는다 (`§7.4` 와 같은 이유).

### 7.6 탐색 비용

<!-- CLAIM: C06 -->
Table 6. held-out median.

| controller | decision-search GE | 예산 대비 |
|---|---|---|
| `onestep_narrow` | 1,186 | 7.9× |
| `committed_Q4_narrow` | 69,401 | 463× |
| `shrinking_Q4_narrow` | 194,095 | 1,294× |

보고한 헤드룸은 배포 예산의 `1,294배` 를 쓴 oracle 값이다. 이 비용 격차의 함의는
`§13.4` 에서 다룬다.

---

## 8. Where does the headroom come from?

**쌍별 차이의 median 은 선형이 아니다.** 따라서 아래 값들은 하나의 합으로 분해되지
않는다. 각각을 **독립적으로 측정한 통계**로 읽어야 한다.

<!-- CLAIM: C07 -->
<!-- CLAIM: C01 -->
<!-- CLAIM: C02 -->
Table 7. held-out `n=40`, 모두 튜닝된 상수를 기준으로 한 쌍별 median.

| treatment | median | 95% CI | p | 양수 |
|---|---|---|---|---|
| `best_open_loop` | +0.395 | [+0.350, +0.476] | <0.0001 | 40/40 |
| `onestep_narrow` | +1.155 | [+1.092, +1.811] | <0.0001 | 40/40 |
| `committed_Q4_narrow` | +2.090 | [+1.532, +2.407] | <0.0001 | 40/40 |
| `shrinking_Q4_narrow` (A2) | +1.690 | [+1.462, +2.368] | <0.0001 | 40/40 |

증분 비교는 **직접 쌍별로 측정한 것만** 쓴다.

| 비교 | median | 95% CI | p | 양수 |
|---|---|---|---|---|
| `shrinking` − `onestep` (C2) | +0.456 | [+0.254, +0.720] | <0.0001 | 35/40 |
| `shrinking` − `committed` (C3) | +0.010 | [−0.033, +0.053] | 0.97 | 21/40 |

### 8.1 표의 값을 서로 빼면 안 된다

`committed` 의 상수 대비 값(`+2.090`)이 `shrinking` 의 값(`+1.690`)보다 크다. 그러나
직접 측정한 `shrinking − committed` 는 `+0.010` 이다. 두 진술은 모순이 아니다.

```text
spec 별 shrinking − committed
κ=10³  +0.472    κ=10⁴  −0.019    κ=10⁵  +0.009    κ=10⁶  +0.000
```

각 spec 안에서 두 planner 는 사실상 동률이다. pooled median 이 서로 다른 인스턴스에
떨어져 `+2.090` 과 `+1.690` 이라는 marginal 값 차이가 생긴다. **비교는 직접 쌍별
통계로만 한다.** Figure 1(b) 에도 같은 경고를 넣었다.

초판 초안은 `onestep` 의 상수 대비 값을 `A2 − C2 = 1.690 − 0.456 = 1.233` 으로
계산했다. **틀렸다.** 직접 측정값은 `+1.155` 다.

### 8.2 정성적 결론

```text
비정상적 제어 자체        상수 -> 1-step 에서 +1.155.  가장 큰 단일 기여
다단계 lookahead 의 증분  1-step -> planner 에서 +0.456,  CI 하한 +0.254
실행 중 재계획의 증분     committed -> planner 에서 +0.010,  CI [−0.033, +0.053]
```

`Q1` 은 지지된다. `Q2` 는 이 조건에서 지지되지 않는다.

---

## 9. Conditioning (exploratory)

Table 3. spec 별 `A2`, `n=10` each.

| κ | median | 95% CI | 양수 |
|---|---|---|---|
| 10³ | +6.983 | [+6.956, +7.101] | 10/10 |
| 10⁴ | +2.317 | [+1.529, +2.507] | 10/10 |
| 10⁵ | +1.399 | [+0.953, +1.517] | 10/10 |
| 10⁶ | +1.253 | [+0.966, +1.693] | 10/10 |

<!-- CLAIM: C18 -->
> Adaptive headroom did not increase monotonically with condition number; the largest
> effect was observed at `κ=10³`. This suggests that condition number alone does not
> explain the headroom.

반대 방향의 단조 관계도 주장하지 않는다. damping 격자, CG 예산, 초기 gradient 정렬,
스펙트럼 분포가 함께 영향을 줄 수 있다.

---

## 10. Model mismatch (exploratory)

### 10.1 설계

`§7` 의 결과는 결정론적 목적함수에서 얻었다. 그 조건에서는 planner 의 내부 모델이
정확하므로 초기 상태에서 세운 계획이 이미 최적 예측이고 재계획이 새 정보를 얻을 수
없다. 따라서 `C3 ≈ 0` 을 두 가지로 해석할 수 있다.

```text
해석 1  feedback 은 원래 가치가 없다
해석 2  이 task 족이 예측 가능해서 feedback 이 필요 없었다
```

핵심 질문은 "모델이 비선형인가" 가 아니라 **"초기 계획 시점에 미래 상태를 정확히
예측할 수 없는가"** 다. 그래서 **같은 모델과 데이터**에 대해 optimizer 가 보는 표본만
바꿨다.

```text
R1 full-batch   전체 데이터로 gradient 와 HVP. 결정론적
R2 minibatch    고정 seed 의 batch 시퀀스. step 마다 표본이 바뀐다
```

`R2` 는 부분표본 곡률을 Newton-CG 에 넣는 설정이며, 그 자체가 어려운 문제라는 것은
알려져 있다 [@byrd2011stochastic]. 우리 목적은 그 어려움을 해결하는 것이 아니라
planner 의 내부 예측 모델이 틀렸을 때 재계획의 가치가 달라지는지를 보는 것이다.

2-layer MLP(4,869 파라미터), 고정 teacher network 로 만든 512 샘플 5-class 분류
문제다. 평가 목적함수는 항상 전체 데이터이므로 `J_E` 가 regime 간 비교 가능하다.

batch 는 **실제 step 뒤에만** 전진한다. planner 의 look-ahead 시뮬레이션 중에는
전진하지 않으므로 planner 는 현재 batch 로 미래를 예측한다. 미래 batch 를 미리 보면
데이터에 대한 oracle 지식이 되어 `R2` 의 목적이 무너진다.

### 10.2 결과

Table 4. 절대 median `J_E`, regime 당 `n=3`.

| controller | R1 full-batch | R2 batch 128 | R2 batch 64 |
|---|---|---|---|
| `best_static` | 5.220 | 3.765 | 3.029 |
| `best_open_loop` | 3.299 | 2.100 | 2.984 |
| `heuristic` | 5.105 | 2.162 | 0.986 |
| `onestep_narrow` | 19.886 | 4.296 | 3.468 |
| `committed_Q4_narrow` | 20.491 | −0.402 | 1.086 |
| `shrinking_Q4_narrow` | 20.395 | 3.185 | 2.757 |

Table 5. regime 별 쌍별 차이.

| 비교 | R1 full-batch | R2 batch 128 | R2 batch 64 |
|---|---|---|---|
| `shrinking` − `best_static` | +15.176 (3/3) | −0.900 (1/3) | −0.277 (1/3) |
| `shrinking` − `onestep` | +0.547 (2/3) | −1.111 (1/3) | −0.716 (1/3) |
| `shrinking` − `committed` | −0.095 (1/3) | +2.941 (3/3) | +1.666 (3/3) |
| `committed` − `onestep` | +0.470 (2/3) | −4.444 (0/3) | −2.322 (0/3) |

### 10.3 `R2` 에서 `C3 > 0` 의 원인

절대값이 두 해석을 구별한다.

```text
committed  R1 20.491  →  R2(64) 1.086,  R2(128) −0.402     붕괴
shrinking  R1 20.395  →  R2(64) 2.757,  R2(128)  3.185
```

`R2` 에서 `committed` 는 `best_static` 보다도 낮고 `onestep` 보다 `2.3 ~ 4.4 nat`
뒤진다. 거절률이 결정적이다.

Figure 3. `committed` 거절률: R1 `0.00` → batch 128 `0.79` → batch 64 `0.66`.

초기 상태에서 세운 계획의 상당 부분이 이후 batch 에서 거절된다. 즉 `C3 > 0` 은
`shrinking` 이 좋아진 것이 아니라 **낡은 계획을 고수하는 것이 손해**라는 뜻이다.

동시에 `R2` 에서 planner 는 튜닝된 상수보다도, 값싼 1-step 제어보다도 낮다. 상태
조건 제어 자체는 `R2` 에서도 도움이 되지만(`onestep 3.468` vs `best_static 3.029`),
그것은 이미 1-step greedy 가 하고 있고 탐색 비용이 planner 의 `1/146` 이다
(batch 64 regime 의 spec 별 median: `onestep 1,879 GE`, `shrinking 275,286 GE`).

### 10.4 한계

```text
regime 당 n=3. CI 와 p-value 를 인용하지 않는다
batch size 3 점의 전이가 단조가 아니다. batch 128 이 batch 64 보다 낮다
regime 간 절대값 비교는 FLOP 정규화가 아니다 (§2.2)
모델 1종, 데이터 1종
```

`R1` 의 `C2 = +0.547` 은 `§7` 의 `+0.456`(`n=40`, `p<0.0001`)과 방향이 같지만, 대체
수락 기준에서는 `−0.080` 이 됐다. `n=40` 결과가 강한 쪽이며 `n=3` 이 그것을 뒤집지
않는다. **불일치 자체를 결과로 보고한다.**

---

## 11. Acceptance criterion ablation (exploratory)

### 11.1 동기

`§10` 의 `R2` 결과에는 두 효과가 섞여 있다.

```text
[1] minibatch 가 바뀌어 초기 계획이 낡아지는 현상
[2] 잡음 있는 목적함수에서 단조 감소를 요구해 step 이 거절되는 현상
```

기본 수락 규칙은 현재 minibatch 목적함수의 단조 감소를 요구한다. 참 목적함수를
개선하는 step 도 표본 잡음 때문에 거절될 수 있다.

### 11.2 설계

> We replaced minibatch-local monotonic acceptance with a fixed-evaluation-objective
> criterion.

gradient 와 HVP 는 계속 minibatch 에서 계산한다. 바뀌는 것은 수락 판정뿐이다. 고정
평가 목적함수 forward 는 `n_samples / batch_size` 배 비싸고, 그 배수를 GE 회계에
포함했다.

### 11.3 결과

| `C3` | control | fixed-evaluation |
|---|---|---|
| R1 full-batch | −0.095 (1/3) | −0.949 (0/3) |
| R2 batch 128 | +2.941 (3/3) | +1.110 (3/3) |
| R2 batch 64 | +1.666 (3/3) | +0.973 (3/3) |

| `C2` | control | fixed-evaluation |
|---|---|---|
| R1 full-batch | +0.547 (2/3) | −0.080 (1/3) |
| R2 batch 128 | −1.111 (1/3) | −0.118 (1/3) |
| R2 batch 64 | −0.716 (1/3) | +0.104 (2/3) |

| `A2` | control | fixed-evaluation |
|---|---|---|
| R1 full-batch | +15.176 (3/3) | +14.065 (3/3) |
| R2 batch 128 | −0.900 (1/3) | −0.598 (0/3) |
| R2 batch 64 | −0.277 (1/3) | −0.093 (0/3) |

<!-- CLAIM: C13 -->
> This alternative criterion reduced, but did not eliminate, the apparent advantage of
> replanning over a committed stale plan. The estimated `C3` magnitude decreased
> substantially under the alternative acceptance criterion, indicating that the
> original magnitude was partly acceptance-dependent.

`R2` 에서 `shrinking` 은 여전히 `best_static` 을 이기지 못하고(`0/3` 양수)
`onestep` 을 비기는 데 그친다.

### 11.4 대체 기준은 완화가 아니라 엄격화였다

<!-- CLAIM: C14 -->
거절률이 올라갔다.

| controller / regime | control | fixed-evaluation |
|---|---|---|
| `committed` batch 128 | 0.79 | 0.89 |
| `committed` batch 64 | 0.66 | 0.92 |
| `shrinking` batch 128 | 0.04 | 0.42 |
| `shrinking` batch 64 | 0.00 | 0.36 |
| `onestep` batch 64 | 0.00 | 0.57 |

기본 규칙에서는 `loss_before` 와 후보 loss 가 **같은 minibatch** 위에 있다. 방금
gradient 를 계산한 batch 의 loss 를 줄이는 것은 쉽다. 대체 기준은 전체 데이터 loss 의
감소를 요구하므로 "현재 batch 에 과적합하는 step" 을 막는다.

### 11.5 이 ablation 의 한계

**단일 요인 변경이 아니다.** 수락 기준과 유효 step 수가 함께 바뀐다. 평가 비용을
회계에서 빼면 비용을 숨기는 것이 되므로 넣었고, 그 대가로 두 효과가 섞였다. 따라서
두 열의 **절대값 비교로 수락 규칙의 우열을 주장하지 않는다.**

<!-- CLAIM: C19 -->
`R1` 열은 의미 비교가 아니다. `full-batch` 에서는 고정 평가 목적함수와 곡률 목적함수가
같은 값이므로, 대체 기준이 바꾸는 것은 step 마다 forward 1회를 더 청구하는 것뿐이다
(`best_static` `5.2199 → 5.1094`). 예산 교란이다.

---

## 12. Why we did not train a policy

이 절은 **두 가지를 분리해** 서술한다. 하나는 데이터가 지지하는 과학적 결론이고,
다른 하나는 자원 투입에 관한 프로젝트 결정이다.

### 12.1 Scientific conclusion

<!-- CLAIM: C03 -->
1. 다단계 planning 은 held-out 에서 1-step 제어보다 개선됐다
   (`+0.456 nat`, CI `+0.254 ~ +0.720`, 35/40).

<!-- CLAIM: C04 -->
2. 실행 중 재계획은 committed planning 보다 의미 있게 개선되지 않았다.
   `+0.010 nat`, 21/40.

> The confidence interval was narrow and included zero, and we did not observe a
> practically large feedback benefit.

`equivalent` 나 `statistically the same` 을 쓰지 않는다. 사전 equivalence margin 이
없다 [@schuirmann1987tost; @lakens2017equivalence].

<!-- CLAIM: C12 -->
3. stochastic model mismatch 에서 committed plan 은 취약했다. 거절률이
   `0.00 → 0.66~0.79` 로 올랐고 터미널 개선이 튜닝 상수보다 낮아졌다.

<!-- CLAIM: C16 -->
4. 재계획은 그 취약성을 줄였으나 값싼 1-step 제어를 안정적으로 이기지 못했다.
   `replanning − one-step` 이 두 minibatch 조건에서 각각 `−1.111`, `−0.716`
   (`n=3` per regime, exploratory).

### 12.2 Project decision

> PPO training was a conditional next stage, not a required component of the study. We
> predeclared empirical gates to determine whether the added implementation and
> computational cost was justified. Because feedback replanning did not consistently
> outperform committed planning or inexpensive one-step control, we did not proceed to
> PPO.

`decision-search` 비용이 배포 예산의 `1,294배` 라는 점도 이 판단에 들어간다.

명확히 구분한다.

```text
PPO 가 실패한 것이 아니다
PPO 를 실험하지 않았다
강화학습 전반을 부정하는 것이 아니다
현재 환경에서 상태 의존 학습 feedback 정책을 정당화할 헤드룸을 확보하지 못했다
```

### 12.3 게이트의 지위

게이트 임계값(`C3 ≥ 0.3 nat` 등)은 **이론적 보편 기준이 아니다.** 다음 성격을
명시한다.

```text
정책 학습은 별도의 큰 실험 단계를 요구한다
따라서 사전에 최소 효과 크기, 부호 일관성, 다단계 사용 여부, 전이 조건을 등록했다
게이트는 통계적 진리 판정이 아니라 scope-control 장치다
결과를 본 뒤 임계값을 바꾸지 않았다
```

임계값은 **결과를 보기 전에 고정했고 이후 바꾸지 않았다.** 그 구체적인 값은
scope-control 선택이며 정확성 주장이 아니다.

### 12.4 후속 방향

`C2` 가 held-out 에서 GO 라는 것은 좋은 다단계 시퀀스가 존재한다는 뜻이고, `C3` 가
작다는 것은 그 시퀀스를 초기 상태에서 정할 수 있다는 뜻이다. 이 조합은 step 단위
feedback 정책보다 **초기 문제 특징에서 스케줄을 예측하는 amortized 접근**을 시사한다
[@amos2023amortized; @andrychowicz2016l2l; @bae2022apo]. 우리는 그것을 구현하지 않았고
후속 방향으로만 언급한다.

---

## 13. Discussion

### 13.1 Planning is not feedback

사다리의 각 구간이 서로 다른 능력의 가치를 잰다.

```text
상수 -> 1-step        상태별 즉시 선택의 가치
1-step -> committed   다단계 lookahead 의 가치
committed -> 재계획    실행 중 feedback 수정의 가치
```

<!-- AVOID: C26 -->
**이 값들을 산술적으로 빼서 분해하지 않는다.** 쌍별 차이의 median 은 선형이 아니다.
실측에서 튜닝 상수 대비 `committed` 가 `+2.090`, `재계획` 이 `+1.690` 인데 직접 측정한
`재계획 − committed` 는 `+0.010` 이다. spec 별로는 `+0.472 / −0.019 / +0.009 / +0.000`
로 두 planner 가 사실상 동률이고, pooled median 이 서로 다른 인스턴스에 떨어져
marginal 값 차이가 생긴다. **비교는 직접 쌍별 통계로만 한다** (`§8.1`, Figure 2).

`Q1`(좋은 시퀀스가 존재하는가)은 지지되고 `Q2`(실행 중 수정할 가치가 있는가)는 이
조건에서 지지되지 않는다. 결정론적 목적함수에서는 planner 의 내부 모델이 정확하므로
초기 상태에서 세운 계획이 이미 최적 예측이고 재계획이 새 정보를 얻지 못한다.

### 13.2 모델 오차가 장기 계획의 위험을 키운다

<!-- CLAIM: C16 -->
탐색적 micro-neural 결과는 이렇게 정리된다.

> 장기 계획은 내부 예측 모델이 정확할 때 가치가 있고, 모델이 틀리면 stale plan 의
> 위험이 커진다.

minibatch regime 에서 committed plan 의 거절률이 `0.00 → 0.66~0.79` 로 올랐고 터미널
개선이 튜닝 상수보다 낮아졌다. 재계획은 그 손해를 줄였다.

**그러나 같은 조건에서 재계획은 1-step 제어보다 나쁘다.** `재계획 − 1-step` 이 두
minibatch 조건에서 각각 `−1.111`, `−0.716` 이고 `재계획 − 튜닝 상수` 도 각각
`−0.900`, `−0.277` 이다. 즉 `committed → 재계획` 의 양수 값만 보고 학습 feedback
정책의 필요성을 읽으면 안 된다. **그 양수는 stale plan 회피이지 값싼 myopic 제어에
대한 우위가 아니다.**

`n=3` per regime 이므로 크기를 주장하지 않는다.

### 13.3 Benchmark audit 이 왜 필요했는가

> Our benchmark audit was necessary because nominal numerical ceilings and task labels
> did not reliably identify whether a problem could distinguish controllers.

이 프로젝트에서 실제로 결론을 뒤집은 문제들이다.

```text
수치 하한에 의한 joint saturation           초기 dev subset 2/3 spec
one-sided saturation                        쌍 삭제가 좋은 결과를 제거
Rosenbrock 의 도달 가능한 국소최소점         적격 판정을 통과했다
같은 seed 가 같은 인스턴스를 만든 문제        n=3 이 실제로 n=1
전역 optimum 기준과 시작점 basin 기준 ceiling 의 차이
단일 reference solver 의 실패 가능성          k=1e6 에서 L-BFGS 미수렴
```

`task 이름으로 포화를 판정한다` 는 초기 규칙과 `수치 하한으로 상한을 계산한다` 는
초기 규칙이 모두 틀렸다. 그래서 참조 solver panel 과 도달 가능 상한을 도입했다.

**범용 benchmark framework 를 제안하는 것이 아니다.** 이 프로젝트의 benchmark 를
교정한 경험적 절차이며, 유사한 controller 비교를 설계할 때 확인할 항목 목록으로
읽히기를 의도한다.

### 13.4 값싼 방법이 놓치는 것

`onestep` 은 탐색 비용이 예산의 `7.9배` 로 planner 의 `1/164` 이면서 튜닝 상수 대비
`+1.155 nat` 를 얻는다. planner 는 `+1.690 nat` 를 얻지만 탐색 비용이 `1,294배` 다.
직접 측정한 증분 `+0.456 nat` 는 통계적으로 견고하나 그 대가가 이 비용 차이다.
실용적 관점에서 이것이 현재 결과의 가장 직접적인 함의다.

---

## 14. Limitations

### 우리가 주장하지 않는 것

```text
"feedback 은 쓸모없다" / "효과가 0 임을 증명했다"
  equivalence margin 을 사전 등록하지 않았다. §7.4 의 표현만 쓴다

"C3 의 정확히 절반이 acceptance artifact 였다"
  감소 비율이 regime 마다 다르고 (62%, 42%) n=3 이다. §11.3 의 표현만 쓴다

"강화학습이 optimizer 제어에 실패한다"
  정책을 학습하지 않았다

"결정론적 환경이 stochastic 환경보다 N배 좋다"
  GE 가 batch 크기별 FLOP 을 맞추지 않는다 (§2.2)

"헤드룸이 condition number 에 따라 감소한다"
  비단조를 관측했을 뿐이다. 어느 방향의 단조성도 주장하지 않는다

"비선형 문제에서 방법이 실패했다"
  두 Rosenbrock 변형은 benchmark 결함으로 사용 불가 판정을 받았다

"micro-neural 결과가 quadratic 결과를 뒤집는다"
  n=3 exploratory 대 n=40 held-out 이다

쌍별 median 을 더하거나 빼서 만든 값
  median 은 선형이 아니다. 비교는 직접 쌍별 측정으로만 한다 (§8.1)
```

### 프로토콜 이탈

```text
E1  설정 선택 규칙을 1 인스턴스 dry run 의 게이트 표를 본 뒤 확정했다 (§6.4)
E2  (quad κ=10⁵, seed 2) 가 초기 dev audit 과 설정 선택에 중복된다 (§4.2)
E3  일부 실행 시 커밋되지 않은 변경이 있었다. code_dirty 로 기록된다
E4  held-out 실행 중 테스트를 병행했다. wall-clock 기록이 부정확하나 수치 결과와
    판정에는 영향이 없다 (단일 스레드 결정론적 연산)
E5  summary 수준 provenance 의 commit 필드 누락 (아래)
E6  수락 기준 ablation 이 단일 요인 변경이 아니다 (§11.5)
E7  R1 의 대체 기준 열은 예산 교란이다 (§11.5)
E8  게이트 C1 은 seed 1개 진단 baseline 이므로 판정에 쓰지 않았다
E9  3층 보고가 게이트 A1/B 에는 미적용이며 단일 통계를 쓴다
E10 median 규약 불일치 (아래)
E11 결과 식별자에 해당 run 이 쓰지 않는 설정이 포함되어 있었다 (아래)
E12 동결 태그를 동결 시점이 아니라 Stage 2 종료 후에 만들었다 (아래)
```

**E5.**

> Summary-level provenance omitted the commit field because of a reporting bug; the
> corresponding per-run raw records retained the commit, allowing deterministic
> reconstruction.

**E10.**

> An earlier diagnostic script used an upper-middle convention for even sample counts,
> whereas the main paired analysis used the conventional arithmetic median. All
> reporting code was unified before manuscript generation; the selected configuration
> and qualitative conclusions were unchanged.

**E11.**

> Run identifiers initially embedded the full target table rather than only the targets
> used by the corresponding task family. Adding a target entry for an unrelated task
> family therefore invalidated the cost-to-target runs of the quadratic suite. We
> restricted the identifier to the targets actually used and re-executed those runs; the
> fixed-budget comparisons were unaffected, and the cost-to-target statistic is a
> re-aggregation of the same measurements.

**E12.**

> The protocol specified that a tag be created at the configuration-freeze point. The
> tag exists in the private repository, but it was created after the confirmatory runs
> had completed and it points at the final Stage 2 commit rather than at the freeze
> commit. It therefore cannot attest that the configuration was fixed before the
> held-out runs; that ordering is attested by the dated decision record and by the
> freeze commit itself. We did not move the tag retroactively. The tag was not
> published, because the commit it marks precedes the removal of device names from
> committed artifacts.

### 범위와 교란

<!-- CLAIM: C17 -->
```text
[L1] 핵심 confirmatory 결과는 synthetic ill-conditioned SPD quadratic 에 한정된다
     d=100 고정, κ 네 값, held-out seed 10개

[L2] micro-neural 결과는 regime 당 n=3 의 exploratory evidence 다
     CI 와 p-value 를 인용하지 않는다

[L3] acceptance ablation 은 평가 forward 비용까지 포함하므로 단일 요인 변화가 아니다
     수락 기준과 유효 step 수가 함께 바뀐다

[L4] GE 는 regime 내부의 oracle-call matching 이며 batch size 간 FLOP matching 이
     아니다. regime 간 절대값 비교는 descriptive 다

[L5] fixed-evaluation criterion 은 "완화" 가 아니다. 실측에서 거절률이 올라간
     더 엄격한 기준이다

[L6] beam search 비용이 object-level 예산보다 매우 크다
     decision-search 194,095 GE / 150 GE = 1,294배

[L7] D21 규칙은 전체 실행 전에 확정했지만 비용 측정용 dry-run 게이트 표를 본 뒤
     확정했다. protocol deviation 으로 기록한다 (E1)

[L8] PPO 를 직접 실행하지 않았으므로 learned policy 성능에 대한 결론을 낼 수 없다

[L9] Hessian-free Newton 계열 전체나 일반 neural optimization 으로 일반화할 수 없다
```

Rosenbrock 의 benchmark 결함과 median 규약 수정은 `§15` 와 위 이탈 목록에 있다.
여기서 반복하지 않는다.

---

## 15. Reproducibility

### 15.1 실행 환경

> All Stage 2 optimization and planning experiments were executed on CPU. The
> GPU-derived cost-model files in the repository originate from an earlier Stage 1
> calibration and were not used in Stage 2; Stage 2 GE accounting used HVP-equivalent
> oracle counts.

> Search GE measures simulated oracle work rather than wall-clock GPU compute.

단일 스레드로 고정했다. wall-clock 은 어떤 판정에도 쓰지 않는다.

### 15.2 산출물

```text
재현 명령        docs/reproduce.md
자동 생성 결과표  docs/results_stage2.md
raw checksum     paper/evidence_map.md  (SHA-256)
주장 원장         paper/claim_ledger.md
인용 검증         paper/CITATIONS.md
프로토콜          docs/experiment_protocol.md  (결정 D1~D32)
공개 저장소       https://github.com/bu11ymaguire/When_Does_Feedback_Help
공개 릴리스 태그   stage2-report-v1
테스트           490개 (구현 458 + 공개 재현 패키지 32). CPU 수집 기준
```

테스트 개수는 **장비 의존적이다.** `tests/test_cg.py` 와 `tests/test_hvp.py` 의 수치
정확도 테스트 7개가 사용 가능한 device 로 파라미터화돼 있어, CUDA 가 있는 장비는 497개를
수집한다. Stage 2 는 CPU 에서 수행됐고 재현 경로가 지정하는 환경도 CPU 이므로 CPU 기준
값을 적는다. 이전 판이 CUDA 장비 기준 497 만 적어서 CPU 에서 clone 한 사람에게는 틀린
값으로 보였다.

공개 저장소는 검증된 private 소스 트리의 allowlist export 이며 **Git 이력을 공유하지
않는다.** export 와 함께 나가는 manifest 가 private 소스 커밋과 파일별 SHA-256 을
기록하므로, private 이력을 공개하지 않고도 한 방향으로 두 트리를 대조할 수 있다.

export 에는 여기 보고한 모든 쌍별 통계를 다시 계산할 수 있는 행 단위 결과가 들어간다.
**고정된 소프트웨어 환경에서** 쌍별 분석은 결정론적이다(부트스트랩 10,000회, 고정
시드). 따라서 보고한 median 과 CI 와 `p` 를 공개된 행 단위 결과에서 다시 계산할 수 있다.

**라이브러리 버전이나 플랫폼이 다를 때의 비트 수준 일치는 주장하지 않는다.**

다만 위험의 출처를 정확히 적는다. 이 부트스트랩은 **Python 표준 라이브러리의
`random.Random(seed)`** 를 쓴다. NumPy 도 SciPy 도 쓰지 않으므로, 리샘플 인덱스는 그
스택의 판본과 무관하다 (`metrics.py` 의 `_bootstrap_median_ci`). 남는 위험은 두
가지다.

```text
부동소수점 누적 순서   median 과 정렬에서. 이 경로에서는 작다
matplotlib 판본        그림의 PNG 바이트에 직접 영향을 준다
```

그래서 바이트 단위 동일성은 고정된 환경으로 한정해 말한다.

`scripts/check_latex.py --check-remote` 가 원고가 인용한 URL 과 릴리스 태그를 공개
원격에서 조회한다. **공개 저장소를 갱신할 때마다 통과해야 한다.** 채움 표시 검사보다
강하다. 이름을 채우는 것만으로는 그 artifact 가 존재하는지 알 수 없기 때문이다.

결과 저장소는 완료된 run 을 건너뛴다. 정체성 3층 분리 덕에 집계 코드나 문서를
고쳐도 optimizer 가 재실행되지 않는다.

### 15.3 개인정보 정리

> Hostnames in committed public artifacts were replaced with stable pseudonymous labels
> before public release. This sanitization did not modify experimental configurations,
> results, checksums of raw numerical records, or scientific conclusions.

**두 저장소를 구별해야 한다.** 이전 판은 "Git ancestry 에 과거 hostname 이 남아 있다"
고만 적어, 새 공개 저장소에도 그것이 남아 있다는 뜻으로 읽힐 수 있었다.

> Earlier hostnames remain in the private source repository's history. The public
> reproducibility repository is an allowlist export with independent Git history and
> contains only pseudonymous host identifiers.

뒤 문장은 주장이 아니라 검사 결과다. `scripts/export_public_repo.py` 가 공개 대상 파일
중 하나라도 장치 이름을 담으면 실행을 거부한다. 현재 export 에는 별칭 `host-a` 만
있다 (summary JSON 9곳, cost-model YAML 2곳).
