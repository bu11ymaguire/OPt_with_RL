# Outline

`paper/claim_ledger.md` 에 등록된 주장만 쓴다. 각 절에 어떤 claim 이 들어가는지
미리 고정한다.

## 리뷰어가 결정할 항목 (여기서 정하지 않는다)

```text
중심 연구 질문의 최종 문구
contribution 의 강도
부정적 결과를 얼마나 전면에 둘지
목표 형식 (논문 / 기술보고서 / 포트폴리오)
Abstract, Introduction, Discussion, Conclusion 의 주장 강도
```

아래 구조는 **현재 데이터가 지지하는 흐름**을 제시한 것이고 최종 결정이 아니다.

## 연구 질문

> Hessian-free Newton-CG 에서 상태 의존적 계산 자원 제어의 이득은 다단계 planning
> 에서 발생하는가, 실행 중 feedback 에서 발생하는가?

## 절 구성

### 1. Introduction

```text
목적    문제 설정과 질문 분해
claim   없음 (배경). C7 의 사다리를 예고만 한다
citation dembo1982inexact steihaug1983cg nash1984lanczos nash2000survey
         martens2010hessianfree martens2011rnn pearlmutter1994hvp
         levenberg1944 marquardt1963 conn2000trustregion
         andrychowicz2016l2l metz2019pathologies metz2020effective bae2022apo
         schulman2017ppo bertsekas2017dp
분량    1 페이지
```

주의. "RL optimizer 가 성공했다" 로 읽히는 문장을 넣지 않는다. 질문은 **분해**다.

### 2. Problem setup and cost accounting

```text
claim   C6 (탐색 비용), C17 (GE 의 범위)
내용    Newton-CG step, damping / CG budget action 공간
        GE 정의와 그 한계를 여기서 미리 밝힌다
citation hestenes1952cg steihaug1983cg pearlmutter1994hvp
```

`GE normalizes gradient-equivalent oracle calls within a regime, not total
floating-point operations across different batch sizes.` (C17 허용 문장)

### 3. Controllers and oracles

```text
내용    사다리를 정의한다
          best_static          튜닝된 상수 action
          best_open_loop       자원 시계 기반 4구간 스케줄 (튜닝 결과, D16)
          heuristic            규칙 기반
          onestep (C0)         1-step 효율 최대화
          committed            초기 상태에서 한 번 계획하고 맹목 실행
          shrinking            매 step 재계획 (shrinking quota MPC)
        committed 가 open_loop 와 다른 이유를 명시한다 (초기 상태에 조건화된 oracle)
citation rawlings2017mpc bertsekas2017dp
```

### 4. Experimental protocol

```text
claim   C25 (설정을 held-out 으로 고르지 않았다)
내용    seed 역할 분리
          CALIBRATION_SEEDS  0, 1        challenge spec 선정에만
          SELECTION_SEEDS    2, 3, 4     설정 선택에만
          HELD_OUT_SEEDS     100 ~ 109   최종 평가에만
        3계층 정체성 (run_semantics / sweep / aggregation)
        paired design, Wilcoxon signed-rank, bootstrap CI
        사전 등록된 게이트 A1/A2/B/C1/C2/C3/D 와 임계값
citation wilcoxon1945 efron1979bootstrap
         paired design 자체는 인용하지 않는다 (우리 실행기의 설계 사실)
```

### 5. Benchmark eligibility (방법론 기여)

```text
claim   C8, C9, C10
내용    수치 하한 기준 ceiling 의 실패 사례 (rosen_d5)
        achievable_ceiling = log(L0) − log(max(L_ref, L_floor))
        참조 solver panel. 단일 solver 금지 근거 (C9)
        다른 초기화는 진단이며 상한을 올리지 않는다 (C10)
        seed 복제 자동 검출
citation shang2006rosenbrock kok2009rosenbrock
         **변종을 본문에 명시해야 인용이 성립한다** (paper/CITATIONS.md §5)
```

이 절은 **부정적 결과에서 나온 방법론 기여**다. 원고에서 이것을 약점이 아니라
기여로 배치할지는 리뷰어가 결정한다.

### 6. Configuration selection (dev)

```text
역할    configuration selection 근거만. primary 효과 추정이 아니다
내용    사전 등록한 승격 대상 Q in {2,4} x {narrow, wide}
        선택 통계와 tie-break 사다리 (D21)
        선택 결과 shrinking_Q4_narrow (median logΔ 10.5306, 단독)
        Q4_wide 가 Q4_narrow 보다 낮다 (탐색 가능 집합 포함관계 != 실현 성능 비감소)
이탈    E1 을 여기서 공개한다
```

### 7. Held-out confirmation (primary)

```text
claim   C1, C2, C3, C4, C5, C7
내용    Table 1  사다리별 절대 median logΔ
        Table 2  paired delta (A2, C2, C3, B, open_loop, heuristic)
        Table 3  spec 별 A2 (C18 로 연결)
        Figure 1 사다리 막대 그래프
        Figure 2 C3 의 개별 40 delta 분포
```

`C4` 는 허용 문장만 쓴다. `C11` 을 절대 쓰지 않는다.

### 8. Where does the headroom come from? (분해)

```text
claim   C7
내용    상수 대비 사다리 (+0.395 / +1.155 / +2.090 / +1.690) 와
        직접 측정한 증분 (+0.456 다단계, +0.010 재계획) 을 **별도 표로** 낸다
        **합으로 분해하지 않는다.** 쌍별 median 은 선형이 아니다 (C26)
        C2 가 dev 조건부 -> held-out GO 로 승격된 것을 명시
        P3 두 조건 충족 (depth>1 채택률 0.84, cap 0.00)
```

### 9. Conditioning (exploratory)

```text
claim   C18
내용    κ=1e3 +6.983 / 1e4 +2.317 / 1e5 +1.399 / 1e6 +1.253
        비단조. 단조 증가도 감소도 주장하지 않는다 (C22 금지)
        수치는 docs/results_stage2.md 에서만 가져온다 (E10 정정 후)
```

### 10. Model mismatch: micro-neural (exploratory)

```text
claim   C12, C16, C17
내용    두 regime 설계. 핵심 질문은 "비선형인가" 가 아니라
        "초기 계획 시점에 미래 상태를 예측할 수 없는가"
        같은 모델과 데이터, optimizer 표본만 다름
        Table 4  regime 별 절대 median logΔ
        Table 5  regime 별 paired delta
        Figure 3 committed 거절률 vs batch size
이탈    n=3, 전이 비단조, GE 정규화 범위를 이 절에서 반복 명시
```

`C24` 금지. micro-neural 이 held-out quadratic 을 뒤집지 않는다.

### 11. Acceptance criterion ablation (exploratory)

```text
claim   C13, C14, C19
내용    "We replaced minibatch-local monotonic acceptance with a
         fixed-evaluation-objective criterion."
        결과: reduced, but did not eliminate
        거절률이 올라갔다 -> 완화가 아니라 엄격화
        단일 요인 변경이 아님 (E6)
        full_batch 열은 예산 교란 (E7, C19)
```

`C15` 금지. "정확히 절반" 을 쓰지 않는다.

### 12. Why we did not train a policy

```text
claim   C20 의 대체 문장
내용    사전 등록한 게이트를 데이터에 적용한 결과
          결정론적 환경: feedback headroom 관측되지 않음 (C4)
          stochastic 환경: committed 붕괴 완화에 그치고 C0 를 이기지 못함 (C12, C16)
          decision-search 비용이 예산의 1,294배 (C6)
        따라서 상태 의존 정책이 학습할 유용한 신호를 확인하지 못했다
        후속 방향으로 amortized schedule selector 를 언급 (구현하지 않았다)
citation schulman2017ppo amos2023amortized andrychowicz2016l2l bae2022apo
         **PPO 를 실행하지 않았다.** 인용이 실행으로 읽히지 않게 쓴다
```

**이 절의 논리 검증은 리뷰어 몫이다.**

### 13. Limitations

```text
내용    claim_ledger 의 이탈 목록 E1~E9 전부
        NOT SUPPORTED 목록에서 "우리가 주장하지 않는 것" 을 명시적으로 적는다
        C11, C15, C20, C21, C22, C23, C24 를 반전 문장으로 나열
```

### 14. Reproducibility

```text
내용    docs/reproduce.md 참조
        paper/evidence_map.md 의 SHA-256
        테스트 465개
        공개 저장소 URL 과 릴리스 태그 (아직 없다. 제출 전 채운다)
        정체성 3계층이 재실행을 막는 방식
```

## 표와 그림 목록

```text
Table 1  held-out 사다리 절대 median logΔ            C1,C2,C3,C4
Table 2  held-out paired delta 전체                  C1~C5
Table 3  held-out spec 별 A2                         C18
Table 4  micro-neural regime 별 절대값                C12,C16
Table 5  micro-neural regime 별 paired delta          C13,C16
Table 6  탐색 비용과 거절률                            C6,C14
Table 7  eligibility calibration 결과                 C8,C9,C10
Figure 1 사다리 막대 (held-out)                       C7
Figure 2 C3 개별 40 delta 분포                        C4
Figure 3 committed 거절률 vs batch size               C12,C14
Figure 4 spec 별 A2 vs log10 κ                        C18
```

모든 표는 `docs/results_stage2.md` 에서 온다. 손으로 숫자를 옮기지 않는다.
출처는 `paper/evidence_map.md` 의 SHA-256 으로 고정한다.

## 제목 후보 (리뷰어 결정)

```text
Planning Is Not Feedback: Dissecting Adaptive Compute Allocation in
  Hessian-Free Newton Optimization

When Does Feedback Help? Planning and Model Mismatch in
  Hessian-Free Newton Optimization
```

두 번째가 덜 단정적이다. `C11` 을 피하려면 두 번째가 안전하다.
