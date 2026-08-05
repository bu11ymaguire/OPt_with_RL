# 외부 검토 요청 발췌

원고 전체가 아니라 **네 부분만** 담았다. 요청 범위를 좁혀야 검토가 실제로 돌아온다.

원본은 `paper/draft.md`(내용 source of truth)와 `paper/main.tex`(제출용)이다. 여기 수치는
`docs/results_stage2.md` 에서 오고, 그 출처는 `paper/evidence_map.md` 의 SHA-256 으로
고정된다.

---

## 검토를 부탁하는 것

> 연구 질문의 framing 이 타당한지, held-out quadratic 결과를 과대해석한 문장이 있는지,
> PPO 를 진행하지 않은 결정이 논리적으로 설명되는지

세 가지만 봐주시면 된다. 구현 정확성, 코드 품질, 문장 다듬기는 요청 범위가 아니다.

영문 요청문:

> Please comment on three things only: (1) whether the framing of the research
> question is sound, (2) whether any sentence overstates the held-out quadratic
> result, and (3) whether the decision not to proceed to PPO is explained
> logically. We are not asking for a review of implementation correctness or
> writing style.

---

## 1. Abstract

We study the allocation of damping and conjugate-gradient (CG) effort in
Hessian-free Newton--CG as a sequential decision problem, and ask whether the
resulting benefit comes from multi-step planning or from execution-time state
feedback.

Under a fixed budget of 150 gradient-equivalent units (GE) we compare, on the same
ladder, a tuned constant setting, a resource-clock open-loop schedule, a one-step
greedy controller, a planner that commits to a single plan formed at the initial
state, and a planner that replans at every step. The configuration was fixed once
on development seeds and measured on disjoint held-out seeds.

On 40 instances of ill-conditioned symmetric positive definite quadratics
(`d = 100`, `κ ∈ {1e3, 1e4, 1e5, 1e6}`, ten held-out seeds each), multi-step
lookahead improved over one-step greedy control by `+0.456 nat`
(95% CI `+0.254` to `+0.720`, `p < 0.0001`, 35/40 instances). On the same 40
instances, replanning during execution differed from the committed planner by
`+0.010 nat` (95% CI `−0.033` to `+0.053`, `p = 0.97`, 21/40). The confidence
interval was narrow and included zero, and **we did not observe a practically
large feedback benefit.**

As an exploratory extension we compared a micro-neural problem in a deterministic
full-batch regime and in minibatch regimes, changing only the samples the optimizer
observes for the same model and data (`n = 3` per regime). In the minibatch
regimes, holding a stale plan was costly and replanning reduced that cost, but
**the planner did not consistently outperform inexpensive one-step control.**
Policy learning was a predeclared conditional next stage; on this evidence we did
not proceed to it.

---

## 2. Contributions

1. We operationalize the control of damping and CG effort as a sequential decision
   problem.
2. We present a controller ladder that separates multi-step planning from
   execution-time feedback.
3. On held-out instances of ill-conditioned SPD quadratics, we confirm an
   additional value for multi-step planning (`+0.456 nat`, CI `+0.254` to
   `+0.720`, `p < 0.0001`, 35/40) and do not observe a practically large
   additional value for feedback (`+0.010 nat`, CI `−0.033` to `+0.053`).
4. We document and apply an audit procedure required to make our controller
   comparisons identifiable.

**의도적으로 하지 않은 표현.** `1` 을 "새로운 MDP formulation" 으로 쓰지 않았다. `4` 를
"general benchmark framework" 라고 쓰지 않았고 보조 기여로 배치했다. PPO 를 진행하지
않은 결정은 기여 목록에 넣지 않았다.

---

## 3. 핵심 결과표

### 3.1 사다리 각 단계 (held-out, `n = 40`)

**이 표의 값을 서로 빼면 안 된다.** 쌍별 차이의 median 은 선형이 아니다. 아래 두 블록은
각각 독립적으로 측정한 통계다.

튜닝된 상수 설정 대비:

| treatment | median | 95% CI | p | 양수 |
|---|---|---|---|---|
| `best_open_loop` | +0.395 | [+0.350, +0.476] | <0.0001 | 40/40 |
| `onestep_narrow` | +1.155 | [+1.092, +1.811] | <0.0001 | 40/40 |
| `committed_Q4_narrow` | +2.090 | [+1.532, +2.407] | <0.0001 | 40/40 |
| `shrinking_Q4_narrow` (A2) | +1.690 | [+1.462, +2.368] | <0.0001 | 40/40 |

직접 측정한 증분:

| 비교 | median | 95% CI | p | 양수 |
|---|---|---|---|---|
| `shrinking` − `onestep` (C2) | +0.456 | [+0.254, +0.720] | <0.0001 | 35/40 |
| `shrinking` − `committed` (C3) | +0.010 | [−0.033, +0.053] | 0.97 | 21/40 |

`committed` 의 상수 대비 값(`+2.090`)이 `shrinking`(`+1.690`)보다 크지만, 직접 측정한
`shrinking − committed` 는 `+0.010` 이다. spec 별로는 `+0.472 / −0.019 / +0.009 /
+0.000` 로 두 planner 가 사실상 동률이고, pooled median 이 서로 다른 인스턴스에 떨어져
marginal 값 차이가 생긴다.

초판 초안이 `A2 − C2 = 1.233` 을 `onestep` 의 상수 대비 값으로 계산했다. **틀렸고**
직접 측정값은 `+1.155` 다. 제출 전에 잡았다.

### 3.2 탐색 비용 (held-out median)

| controller | decision-search GE | 배포 예산 대비 |
|---|---|---|
| `onestep_narrow` | 1,186 | 7.9× |
| `committed_Q4_narrow` | 69,401 | 462.7× |
| `shrinking_Q4_narrow` | 194,095 | 1,294.0× |

배포 예산은 `150 GE` 다. `+0.456 nat` 의 증분은 통계적으로 견고하나 그 대가가 이 비용
차이다.

> Search GE measures simulated oracle work rather than wall-clock GPU compute.

### 3.3 모델 불일치 (exploratory, regime 당 `n = 3`)

`n = 3` 이므로 CI 와 p 를 인용하지 않는다. 괄호는 양수 쌍의 개수다.

| 비교 | R1 full-batch | R2 batch 128 | R2 batch 64 |
|---|---|---|---|
| `shrinking` − `best_static` | +15.176 (3/3) | −0.900 (1/3) | −0.277 (1/3) |
| `shrinking` − `onestep` | +0.547 (2/3) | −1.111 (1/3) | −0.716 (1/3) |
| `shrinking` − `committed` | −0.095 (1/3) | +2.941 (3/3) | +1.666 (3/3) |
| `committed` − `onestep` | +0.470 (2/3) | −4.444 (0/3) | −2.322 (0/3) |

`committed` 절대값이 R1 `20.491` 에서 R2 `1.086 / −0.402` 로 붕괴한다. 거절률이
`0.00 → 0.66~0.79` 로 오른다. 즉 `R2` 에서 `C3 > 0` 은 `shrinking` 이 좋아진 것이
아니라 **낡은 계획을 고수하는 것이 손해**라는 뜻이다.

동시에 `R2` 에서 planner 는 튜닝 상수보다도, 값싼 1-step 제어보다도 낮다.

### 3.4 사전 등록 게이트 판정

```text
A1 = GO   A2 = GO   B = 재설계   C1 = 판정불가   C2 = GO   C3 = 재설계   D = GO
```

임계값(`C2 ≥ 0.3`, `C3 ≥ 0.3` 등)은 결과를 보기 전에 고정했고 이후 바꾸지 않았다.
**이 임계값이 옳다고 주장하지 않는다.** 사전에 고정했고 바꾸지 않았다는 것만 주장한다.

### 3.5 실행 환경

> All Stage 2 optimization and planning experiments were executed on CPU. The
> GPU-derived cost-model files in the repository originate from an earlier Stage 1
> calibration and were not used in Stage 2; Stage 2 GE accounting used
> HVP-equivalent oracle counts.

---

## 4. Discussion 과 Limitations

### 4.1 Planning is not feedback

```text
상수 -> 1-step        상태별 즉시 선택의 가치      +1.155
1-step -> committed   다단계 lookahead 의 가치      +0.456  (직접 측정)
committed -> 재계획    실행 중 feedback 의 가치      +0.010  (직접 측정)
```

`Q1`(좋은 시퀀스가 존재하는가)은 지지되고 `Q2`(실행 중 수정할 가치가 있는가)는 이
조건에서 지지되지 않는다. 결정론적 목적함수에서는 planner 의 내부 모델이 정확하므로
초기 상태에서 세운 계획이 이미 최적 예측이고 재계획이 새 정보를 얻지 못한다.

### 4.2 모델 오차가 장기 계획의 위험을 키운다

> 장기 계획은 내부 예측 모델이 정확할 때 가치가 있고, 모델이 틀리면 stale plan 의
> 위험이 커진다.

그러나 같은 조건에서 재계획은 1-step 제어보다 나쁘다. `committed → 재계획` 의 양수 값만
보고 학습 feedback 정책의 필요성을 읽으면 안 된다. **그 양수는 stale plan 회피이지 값싼
myopic 제어에 대한 우위가 아니다.**

### 4.3 PPO 를 진행하지 않은 이유

이 절은 두 가지를 분리한다.

**과학적 결론.** 다단계 planning 은 held-out 에서 1-step 을 개선했다. 실행 중 재계획은
committed planning 보다 의미 있게 개선되지 않았다(CI 가 좁게 0 을 포함). stochastic
regime 에서 committed plan 은 취약했으나 재계획이 값싼 1-step 제어를 안정적으로 이기지
못했다.

**프로젝트 결정.**

> PPO training was a conditional next stage, not a required component of the study.
> We predeclared empirical gates to determine whether the added implementation and
> computational cost was justified. Because feedback replanning did not consistently
> outperform committed planning or inexpensive one-step control, we did not proceed
> to PPO.

명확히 구분한다.

```text
PPO 가 실패한 것이 아니다
PPO 를 실험하지 않았다
강화학습 전반을 부정하는 것이 아니다
현재 환경에서 상태 의존 학습 feedback 정책을 정당화할 헤드룸을 확보하지 못했다
```

후속 방향으로는 step 단위 feedback 정책보다 초기 문제 특징에서 스케줄을 예측하는
amortized 접근을 언급하되, 구현하지 않았음을 밝힌다.

### 4.4 우리가 주장하지 않는 것

```text
"feedback 은 쓸모없다" / "효과가 0 임을 증명했다"
  equivalence margin 을 사전 등록하지 않았다. TOST 는 사전 경계를 요구한다

"C3 의 정확히 절반이 acceptance artifact 였다"
  감소 비율이 regime 마다 다르고 (62%, 42%) n=3 이다

"강화학습이 optimizer 제어에 실패한다"
  정책을 학습하지 않았다

"결정론적 환경이 stochastic 환경보다 N배 좋다"
  GE 가 batch 크기별 FLOP 을 맞추지 않는다

"헤드룸이 condition number 에 따라 감소한다"
  비단조를 관측했을 뿐이다. 어느 방향의 단조성도 주장하지 않는다

"비선형 문제에서 방법이 실패했다"
  두 Rosenbrock 변형은 benchmark 결함으로 사용 불가 판정을 받았다

"micro-neural 결과가 quadratic 결과를 뒤집는다"
  n=3 exploratory 대 n=40 held-out 이다

쌍별 median 을 더하거나 빼서 만든 값
  median 은 선형이 아니다
```

### 4.5 범위

```text
[L1] 핵심 confirmatory 결과는 synthetic ill-conditioned SPD quadratic 에 한정된다
     d=100 고정, κ 네 값, held-out seed 10개
[L2] micro-neural 결과는 regime 당 n=3 의 exploratory evidence 다
[L3] acceptance ablation 은 단일 요인 변화가 아니다
[L4] GE 는 regime 내부의 oracle-call matching 이며 batch size 간 FLOP matching 이 아니다
[L5] fixed-evaluation criterion 은 완화가 아니라 거절률이 오른 더 엄격한 기준이다
[L6] beam search 비용이 object-level 예산의 1,294배다
[L7] 선택 규칙을 비용 측정용 dry-run 게이트 표를 본 뒤 확정했다 (protocol deviation)
[L8] PPO 를 직접 실행하지 않았으므로 learned policy 성능에 대한 결론을 낼 수 없다
[L9] Hessian-free Newton 계열 전체나 일반 neural optimization 으로 일반화할 수 없다
```

프로토콜 이탈 `E1~E11` 은 `paper/draft.md §14` 에 전부 나열돼 있다.

---

## 참고

```text
전체 원고        paper/draft.md   (내용 source of truth)
제출용 LaTeX     paper/main.tex, paper/sections/
주장 원장        paper/claim_ledger.md   26 claim 의 상태와 허용 문장
인용 검증        paper/CITATIONS.md      각 인용의 내용 일치와 오인용 위험
자동 생성 결과표  docs/results_stage2.md
raw checksum     paper/evidence_map.md   SHA-256
프로토콜          docs/experiment_protocol.md   결정 D1~D32
```
