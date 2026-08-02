# 실험 프로토콜

> `README.md` 는 **무엇을 만드는가**(명세)를 정의한다.
> 이 문서는 **어떻게 실행하고 무엇으로 판정하는가**(실행 계획)를 정의한다.
>
> 판정 기준과 목표치는 실험 실행 **전에** 확정한다. 결과를 본 뒤 임계값을 조정하면
> 원하는 결론이 나오므로, 변경이 필요하면 §9 변경 이력에 이유와 함께 기록한다.

---

## 1. 기준 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11, PowerShell |
| GPU | NVIDIA RTX 3060 Ti, 8 GB, compute capability 8.6 (Ampere) |
| 드라이버 | 610.88 |
| Python | 3.12.12 (uv managed) |
| PyTorch | 2.13.0+cu130 (CUDA runtime 13.0) |
| Gymnasium | 1.3.0 |
| Stable-Baselines3 | 2.9.0 |
| 의존성 고정 | `uv.lock` (해시 포함, 커밋 대상) |
| venv 위치 | `C:\Users\<user>\.venvs\opt-with-rl` (OneDrive 동기화 회피, `UV_PROJECT_ENVIRONMENT`) |

사전 확인 완료 사항:

- `torch.autograd.grad(..., create_graph=True)` 기반 double-backward HVP가
  CUDA에서 정상 동작. SPD quadratic(d=64) 대조 상대오차 `5.8e-8`.

8 GB VRAM은 이 프로젝트의 상한을 규정한다. Phase 1·2 모델은 모두 수십만~수백만
파라미터 규모로 제한하고, 대형 모델 실험은 범위에 넣지 않는다.

---

## 2. 확정된 설계 결정

README가 열어둔 항목 중 결과 해석에 직접 영향을 주는 것들을 여기서 고정한다.

### D1. 주 비용 지표는 grad-equivalent(GE), wall-clock은 보조 지표

**문제.** MNIST MLP 784-128-10은 약 101,770 파라미터다. 이 규모에서 GPU 시간은
FLOP이 아니라 커널 런치와 파이썬 오버헤드가 지배한다. HVP 1회의 실제 연산이
수십 µs인데 런치 비용이 수백 µs라면, wall-clock으로 측정되는 것은 최적화 효율이
아니라 구현 오버헤드다. 이 상태에서 "RL이 wall-clock을 15% 줄였다"는 주장은
방어할 수 없다.

**결정.** 1 GE = gradient batch 1회 forward+backward. 모든 비용을 GE로 환산한다.

```text
gradient (create_graph=False)  = 1.0 GE          (정의)
gradient (create_graph=True)   ≈ c_grad_graph GE
HVP 1회 (그래프 재사용)         ≈ c_hvp  GE
step acceptance forward        ≈ c_fwd  GE
Newton-CG step (k iters)       = c_grad_graph + k·c_hvp + c_fwd  GE
```

계수는 이론값(`c_hvp ≈ 2.5`)을 쓰지 않고 **대상 하드웨어에서 실측**한다
(`scripts/measure_cost_model.py`). 산출값은 `configs/cost_model.<model>.yaml` 로
저장하고 config·commit hash와 함께 기록한다.

- 주 지표: **GE** — 하드웨어 독립적, 재현 가능. 단 GE를 무엇에 쓰는지는
  트랙에 따라 다르다 (D9 참조)
- 보조 지표: wall-clock — "실제로도 이득이 남는가" 확인용, 오버헤드 지배 구간임을 명시하여 보고
- Stage 5에서 최소 하나의 task는 FLOP 지배 규모(수백만 파라미터급 CNN)로 두어
  wall-clock 결론을 별도 검증한다

**주장의 범위를 넘지 않는다.** 아래 실측에서 도출할 수 있는 결론은

> 소규모 GPU workload에서는 kernel-launch overhead와 낮은 utilization 때문에
> wall-clock이 계산량을 제대로 반영하지 않을 수 있다.

까지다. "wall-clock 기반 optimizer 논문을 신뢰할 수 없다"로 일반화하려면 다른
GPU, CPU 실험, 여러 모델 크기와 batch size, GPU utilization 또는 profiler 근거가
필요하다. 현재 근거는 단일 GPU에서의 **calibration finding** 이며 독립적인
주요 기여로 선언하지 않는다.

#### 실측 결과 (2026-08-01, RTX 3060 Ti, torch 2.13.0+cu130)

| 구성 | 파라미터 | `t_grad` | `c_grad_graph` | `c_hvp` | `c_fwd` | GE(k=10) | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| MNIST MLP 784-128-10, B=512 | 101,770 | 0.68 ms | 0.93 | 1.59 | 0.26 | 17.0 | launch-bound |
| small CNN (CIFAR 규모), B=128 | 2,193,226 | 5.64 ms | 1.00 | 3.12 | 0.37 | 32.5 | flop-bound |

D1의 가정이 실측으로 확인되었다. MNIST MLP에서 **배치를 256에서 4096으로 16배
늘려도 gradient 시간이 0.657 ms → 0.706 ms로 거의 변하지 않는다.** 연산량과
무관하게 고정 오버헤드가 시간을 지배한다는 직접적인 증거다. 같은 구성에서
`c_hvp` 측정값도 반복 간 1.30 ~ 2.53으로 흔들려 이론값(약 2~3)에서 벗어난다.

small CNN에서는 배치 2배에 시간이 1.6~2.1배로 비례하고 `c_hvp` 가 3.12로
이론 기대치에 부합한다. 즉 **이 GPU에서 FLOP 지배 구간은 도달 가능하며,
Stage 5의 wall-clock 검증은 실현 가능하다.**

**이 결정에서 따라오는 냉정한 계산.** 실측 기준 MNIST MLP에서 `k = 10` 이면
Newton-CG 1 step ≈ **17 GE**. 200 step = 3,400 GE 로, `B=512` 기준 약 29 에폭에
해당한다. AdamW는 5에폭(약 600 GE) 안에 98%에 도달한다. 따라서 **동등 예산에서
경쟁하려면 Newton-CG는 약 35 step 안에 목표에 도달해야 한다.** 이 수치를 알고 시작한다.

### D2. `B_curv / B_grad` 비율은 1급 실험 변수

D1의 비용식에서 이 비율 하나가 step 비용을 4배 좌우한다. RL 컨트롤러가 목표로 하는
이득(10~30%)보다 영향이 크다. 따라서 grid에 포함한다.

```text
B_curv / B_grad ∈ {1/4, 1/2, 1}
```

**gradient batch와 curvature batch의 동일성.** `g`와 `H`를 다른 배치에서 뽑으면
선형계 `(H_A + λI)p = -g_B` 가 불일치해져 Newton 방향이 편향된다. 기본값은
**동일 배치**(`B_c = B_g`)로 두고, 분리 방식(Martens 2010 스타일)은 ablation으로 돌린다.
CG solve 1회 내부에서 curvature batch를 바꾸지 않는다는 README 원칙은 유지한다.

### D3. 보상은 트랙마다 다르다. per-step ratio 보상은 쓰지 않는다

초판은 단일 보상을 썼다.

```text
r_t = (log L_t − log L_{t+1}) − β · (cost_t / GE_ref) − γ · I_failure      [폐기]
```

**Stage 2 파일럿에서 이 설계의 결함이 드러났다.** 같은 형태의 목적
(`Δlog L / cost`)으로 매 step 최선을 고르는 컨트롤러가 고정 설정보다
cost-to-target에서 **나빴다** (비율 0.967x). 국소 효율 최대화가 총비용 최소화와
다른 문제이기 때문이다.

```text
행동 A:  3 GE 로 loss 10% 감소     → 순간 효율 높음
행동 B: 20 GE 로 loss 60% 감소     → 목표까지 총비용은 더 적을 수 있음
```

per-step ratio 보상을 쓰면 정책이 `k=3` 같은 싸고 작은 행동만 반복할 유인이
생긴다. 그래서 보상을 트랙별로 분리한다 (D9).

#### Track E 보상 (고정 GE 예산)

```text
r_t = (log L_t − log L_{t+1}) − γ · I_failure
에피소드 종료: 누적 GE ≥ B
```

리턴이 텔레스코핑되어 `log(L_0 / L_B) − γ·(총 실패)` 가 된다. **트랙 E의 목적과
정확히 일치한다.** 행동별 비용 차이는 보상을 나누는 대신 **남은 예산에서
차감**하는 방식으로 반영한다. 비용이 큰 행동은 예산을 더 많이 먹으므로 자연히
에피소드가 짧아진다.

#### Track T 보상 (목표 도달 총비용)

```text
r_t = −c_t / GE_ref − γ · I_failure
에피소드 종료: L ≤ τ (도달) 또는 누적 GE ≥ B (절단)
```

stochastic shortest-path 형태다. 리턴이 `−(총 GE)/GE_ref` 가 되어 트랙 T의
목적과 정확히 일치한다. 미도달 절단에 큰 임의 벌점을 주지 않고 D6의 절단 규칙을
유지한다. 학습이 불안정하면 potential-based shaping을 더한다.

```text
r'_t = r_t + γ_disc · Φ(s_{t+1}) − Φ(s_t),   Φ(s) = −max(0, log L − log τ)
```

potential-based shaping은 최적 정책을 바꾸지 않는다는 것이 알려져 있으므로
목적을 훼손하지 않는다.

#### 공통 사항

- `γ = 1.0` (실패 패널티), per-step reward clip `[-5, 5]`
- `log L` 이 정의되지 않는 경우를 막기 위해 quadratic 계열은 `L* = 0` 으로
  구성하고 `log(max(L, 1e-30))` 를 쓴다
- indefinite quadratic은 아래로 유계가 아니므로 두 트랙 모두에서 제외한다
- README의 상대 감소량 형태와 폐기된 ratio 형태는 ablation으로 보존한다

### D4. baseline에 open-loop schedule과 best-of-36 static을 추가

README의 4종(AdamW, SGD+momentum, Fixed, Heuristic)만으로는 결론을 낼 수 없다.

| baseline | 정의 | 왜 필요한가 |
|---|---|---|
| `best_static` | 36개 action 조합을 각각 고정해 전부 실행, meta-train 성능 최고를 선택 | "동일 탐색 예산" 원칙의 정직한 구현 |
| `open_loop` | `progress`(step/총 step)만 입력으로 받는 학습된 스케줄, 랜덤 서치 36회 | **RL이 "적응 제어"인지 "튜닝된 스케줄"인지 구분** |

`open_loop`이 없으면, RL이 Fixed를 이겼을 때 그것이 상태 기반 적응 때문인지
단순히 시간에 따른 스케줄 때문인지 알 수 없다. 이는 논문의 주장 자체가 달라지는
문제이므로 ablation이 아니라 독립 baseline으로 세운다.

전체 비교군:

```text
adamw · sgd_momentum · fixed_newton_cg · heuristic_newton_cg
best_static · open_loop · rl_newton_cg
```

### D5. 탐색 예산을 명시적으로 회계 처리

결과 표에 "optimizer를 얻는 데 든 비용" 열을 넣는다. RL은 meta-training 비용이
있고 Fixed는 없다. 이를 숨기면 learned-optimizer 문헌의 흔한 함정에 빠진다.

| Optimizer | 튜닝/학습 비용 (GE) | 튜닝 run 수 | Cost-to-target (GE) | Wall-clock (s) | Final Acc | Failure Rate |
|---|---:|---:|---:|---:|---:|---:|

**탐색 예산은 모든 컨트롤러에 동일해야 한다.** `best_static` 을 200개 설정에서
찾고 `open_loop` 은 50개만 평가하면 static 쪽에 유리하다. 반대도 마찬가지다.
파일럿에서 이 문제가 실제로 발생했다. static 12개 조합 전수 탐색 대 open_loop
랜덤 서치 12회였는데, open_loop 우승자가 static과 **완전히 동일**한 결과를 냈다
(비율 1.000x, CI 1.000–1.000). 12회로는 스케줄 공간을 사실상 탐색하지 못한다.

```text
탐색 예산 N_tune = 각 컨트롤러가 평가받는 설정 후보 수. 모두 같게 맞춘다.
  best_static   행동 공간 전수 (부족하면 N_tune 까지 반복 없이 확장)
  open_loop     스케줄 파라미터 랜덤 서치 N_tune 회
  heuristic     rho_low / rho_high / 배수 랜덤 서치 N_tune 회
  adamw / sgd   learning rate + weight decay 랜덤 서치 N_tune 회
  rl            PPO 하이퍼파라미터 탐색 횟수를 기록하고 meta-training GE 합산
```

행동 공간이 `N_tune` 보다 작으면 `N_tune` 을 행동 공간 크기로 내리거나, static에
초기 damping 축을 추가해 후보를 늘린다. **어느 쪽이든 실제 사용한 횟수를
결과 표에 기록한다.**

**선택은 dev task/seed 에서만** 하고, 선택된 설정을 held-out task/seed 에 그대로
적용한다 (D6의 pilot / confirmatory 구분과 동일한 분할을 쓴다).

planner의 분석 비용도 별도 열로 기록한다. one-step efficiency controller는 step당
행동 공간 전수 sweep, H-step MPC planner는 그 위에 beam 확장 비용이 든다. 이
비용은 배포 비용(deployment GE)과 합치지 않지만 반드시 보고한다.

### D6. 목표치 다단계 사전 등록, pilot / confirmatory 분리, 절단 규칙

#### target은 난이도별로 여러 개 둔다

target 하나만 잡으면 그 값 선정에 따라 결론이 흔들린다. Track T는 난이도
3단계로 본다.

```text
easy    L / L_0 ≤ 1e-2
medium  L / L_0 ≤ 1e-4
hard    L / L_0 ≤ 1e-6
```

Rosenbrock은 `L* = 0` 이므로 절대값으로 `{1e-1, 1e-2, 1e-4}` 를 쓴다.
신경망 task는 Stage 3에서 확정한다.

#### pilot과 confirmatory를 분리한다

예산과 target을 결과를 본 뒤에 고치면 사후적으로 유리한 프로토콜을 고른 것이
된다. 그래서 두 국면으로 나눈다.

| 국면 | task / seed | 용도 |
|---|---|---|
| **pilot** | dev seed `{0, 1, 2}`, 초기 condition number 집합 | GE 예산과 target 난이도 **선정**. 프로토콜 결정에만 사용 |
| **confirmatory** | held-out seed `{100..109}`, 새 condition number와 초기점 | 최종 결론. 선정된 예산/target을 그대로 적용 |

실행 순서는 세 국면으로 나눈다.

```text
C1  pilot            budget / target / timeout / beam / horizon / N_tune 결정
C2  protocol freeze  config 고정 + 태그. 이후 변경 금지
C3  confirmatory     held-out seed 에서 게이트 판정
```

**C1 pilot 절차:**

1. 여러 GE 예산을 시험한다
2. 방법 대부분이 너무 쉽게 성공하지도, 전부 실패하지도 않는 예산을 고른다
   (도달률이 20~80% 구간에 오도록)
3. easy / medium / hard target을 pilot 분포를 보고 확정한다
4. beam width를 위 calibration 규칙으로 확정한다
5. pilot 결과는 **최종 효과 크기 계산에 섞지 않는다**

**C2 protocol freeze:**

config를 고정하고 태그를 남긴다.

```text
git tag protocol-freeze-stage2-v1
```

함께 저장할 것: 최종 config 파일, pilot 결과 요약, 각 파라미터를 그 값으로
선택한 이유, 이후 변경 금지 항목, 예외적으로 변경 가능한 오류 조건.

**confirmatory 중 버그를 발견하면 조용히 고치고 계속하지 않는다.**

```text
중단 → 버그 범위 기록 → 영향받은 결과 폐기 → 버전 증가(v2) → 전체 재실행
```

**C3 confirmatory:** held-out seed `{100..109}` 에서 게이트 A1·A2·B·C·D를 평가한다.
Track E와 Track T를 분리해 보고한다.

| 트랙 | 보고 항목 |
|---|---|
| Track E | 고정 GE에서 terminal log-loss improvement, paired difference, CI, 행동공간별 헤드룸 |
| Track T | target별 도달률, cost-to-target, restricted mean, 절단 run 수, success-conditioned cost와 전체 성과를 구분 |

**2026-08-01 시점의 파일럿 결과는 예산 300 GE, seed {0,1}에서 도달률 67% 였다.
이 결과는 pilot으로만 분류하며 어떤 결론에도 쓰지 않는다.**

#### 절단(censoring) 규칙

예산 내 미도달 run은 삭제하거나 최댓값으로 대입하지 않는다.

- `success_rate` = 도달한 run 비율 (별도 보고)
- `cost_to_target` = **도달한 run만의 중앙값** (평균이 아님)
- `restricted_mean` = 미도달을 예산값으로 절단한 제한 평균 (보조 지표)
- 위 세 지표를 항상 함께 보고한다. 하나만 보면 왜곡된다.
- 실패 run도 `results/raw/` 에 보존하고 실패 원인 태그(`nan`, `budget_exhausted`,
  `divergence`, `cg_breakdown`, `oom`)를 기록한다.

파일럿에서 이 규칙이 실제로 작동했다. heuristic은 cost-to-target 중앙값이
best_static보다 68% 나빴지만 도달률은 더 높았다(75% vs 67%). 중앙값만 봤다면
"느리지만 더 자주 도달한다"는 다른 성격을 놓쳤을 것이다.

### D7. Paired design과 통계 프로토콜

**Paired design.** seed는 난수 시드가 아니라 **실험 조건 식별자**로 쓴다.
`seed=s` 이면 모든 optimizer가 동일한:

- task 인스턴스 (quadratic의 `A`, 모델 초기화)
- minibatch 순서
- train/val split

을 본다. 구현은 `benchmark/paired.py` 에서 `seed → (task_instance, batch_order)` 를
결정론적으로 매핑한다. 비용이 들지 않는 순수 분산 감소이므로 반드시 적용한다.

**통계.** `mean ± std` 는 보고하지 않는다(n=5에서 무의미하고 정규성 가정도 없다).

- 쌍별 비교: `(task, seed)` 쌍에 대한 **Wilcoxon signed-rank test**
- 효과 크기: `cost_to_target` **비율의 기하평균**과 부트스트랩 95% CI (`n_boot = 10000`)
- seed 수: 최소 5, 주장 근거가 되는 비교는 10
- 다중 비교: 주 가설(RL vs fixed, RL vs heuristic, RL vs open_loop) 3개에 대해
  Holm 보정

### D9. 실험을 두 트랙으로 분리한다

Stage 2 파일럿에서 드러난 것은 지표 불일치가 아니라 **서로 다른 두 최적화 문제를
한 실험에 섞고 있었다**는 사실이다. 분리한다.

#### Track E — 고정 예산에서 얼마나 개선하는가

> 동일한 GE 예산 `B` 를 받았을 때 어떤 컨트롤러가 loss를 가장 많이 낮추는가?

```text
목적:  max  log(L_0 / L_B)      s.t.  Σ c_t ≤ B
지표:  J_E = log L_0 − log L_B  (B가 모두 같으므로 사실상 최종 loss 비교)
```

#### Track T — 목표까지 얼마나 싸게 도달하는가

> 사전 지정한 target loss `τ` 에 도달하는 데 필요한 총 GE는 얼마인가?

```text
목적:  min  Σ_{t≤T_τ} c_t       s.t.  L ≤ τ
지표:  J_T = GE-to-target,  도달률,  제한 예산 내 restricted mean,  절단 run 수
```

#### 두 트랙은 같은 답을 주지 않는다

파일럿이 이미 보여줬다. 국소 효율 컨트롤러는 고정 예산에서는 쓸 만하지만
cost-to-target에서는 고정 설정보다 나빴다. 다음도 충분히 가능하다.

```text
고정 예산에서는 adaptive 가 좋다
하지만 특정 target 까지는 best_static 이 더 싸다
```

**이 불일치 자체가 연구 결과다.** 그래서 둘 다 보고한다.

#### 헤드룸도 트랙별로 정의한다

```text
H_E = J_E(planner) − J_E(best_static)              [nat, 클수록 여지 큼]
H_T = C_τ(best_static) / C_τ(planner)              [배수, 클수록 여지 큼]
```

하나의 "헤드룸"으로 묶으면 같은 혼동이 재발한다.

#### 상한이라고 부르지 않는다

이름과 해석을 정정한다.

| 초판 이름 | 정정된 이름 | 이유 |
|---|---|---|
| `greedy_oracle` | **one-step efficiency controller** | 전역 상한이 아니다. 매 step 즉시 효율이 가장 좋은 후보를 고르는 컨트롤러일 뿐이며, 실제로 고정 설정보다 나쁠 수 있음이 확인됐다 |
| `lookahead_oracle` | **H-step MPC planner** | 유한 horizon과 beam 폭에 제한된 근사다. 전역 최적해가 아니다 |

문서와 표에서 `oracle` 이라는 단어는 도달성 제약이 없는 `absolute` 행동 공간을
쓰는 planner에 한해서만, 그리고 "one-step" / "H-step" 을 함께 붙여서 쓴다.

### D8. Truncated horizon의 근시안 편향 대응

에피소드를 50 step에서 끊으면 정책이 "지금 loss를 최대한 줄이는" 행동을 학습한다.
그런데 우리가 원하는 것은 장기 수렴 효율이다. 목적 불일치가 발생한다.

대응:

- horizon 랜덤화: `H ~ Uniform{30, 50, 80, 120}`
- SB3에서 `TimeLimit.truncated` 시 value bootstrapping이 켜지도록 환경을 구성
  (termination과 truncation을 구분해 반환)
- `gamma = 0.995`, `gae_lambda = 0.95`
- 상태에 `progress` 를 포함하되, 이것이 open_loop baseline과 겹치므로
  `progress` 제거 ablation을 반드시 수행

### D10. Planner 목적함수를 비율에서 고정 GE 쿼터로 교체한다

Track E planner의 초기 목적함수는 다음이었다.

```text
U = (log L_start − log L_terminal) / cumulative_cost
```

dry run에서 이것이 **장기 계획을 구조적으로 검출하지 못한다**는 것이 드러났다.

#### 무엇이 관측됐는가

`quadratic`, seed 0, beam 3, 150 GE 예산:

```text
SPD κ=1e2    H=1/3/5 전부 logΔ=59.8636, depth 히스토그램 {1: 8}
ill κ=1e5    H=1 → 10.4998 {1:10} / H=3 → 10.5116 {1:9,2:1} / H=5 → 동일
wide κ=1e5   H=1 → 10.3315 {1:9}  / H=5 → 10.3564 {1:7,2:2}
```

- `H=3` 과 `H=5` 가 모든 조건에서 완전히 동일했다
- `depth ≥ 3` 은 `H=5` 에서도 채택 0회
- 게이트 C 효과 크기 0.025 nat (GO 0.3, 재설계 0.05)
- `H=5` 의 search 비용은 본문의 약 100배 (16,848 GE vs 166 GE)

#### 왜 그런가

`U` 는 step별 rate의 **비용 가중 평균**이다. mediant 부등식에 의해

```text
min(r₁, r₂) ≤ (g₁+g₂)/(c₁+c₂) ≤ max(r₁, r₂)
```

depth 1에서 이미 최대 rate `R*` 를 골랐으면, depth 2가 이기려면 `r₂ > R*`,
즉 두 번째 step이 **지금 당장 가능한 모든 행동보다** 효율적이어야 한다.
수익 체감이 일반적인 환경에서는 드물다. mediant 부등식 때문에 깊은 계획이
수학적으로 절대 불가능한 것은 아니고, 두 번째 상태에서 더 효율적인 행동이 열리면
이길 수 있다. 실제로 ill-conditioned 문제에서 depth 2가 간헐적으로 채택됐다.
그러나 **장기 투자 행동을 검출하는 목적함수로는 부적합**하다.

#### 결정

이 결과를 "lookahead가 불필요하다"는 근거로 **쓰지 않는다.** 증명되는 것은
다음뿐이다.

> 누적 평균 효율을 최대화하는 목적에서는 짧은 계획이 유리하다.

Track E의 실제 연구 질문은 고정 예산 문제다.

```text
max  log L_t − log L_{t+m}     s.t.  Σ_{i=t}^{t+m−1} c_i ≤ Q
```

여기에 비용으로 나누는 비율은 들어가지 않는다. 따라서 게이트 C의 주 컨트롤러를
`BudgetedMPCController` 로 교체한다. 후보마다 동일한 미래 GE 쿼터 `Q` 를 주고
그 안에서 도달한 terminal loss를 비교한다.

#### 기존 planner는 버리지 않고 이름을 바꿔 보존한다

```text
기존:  HorizonPlannerController   (게이트 C 주 컨트롤러로 오해될 이름)
수정:  AverageRateEfficiencyPlanner  (진단 baseline)
```

버그가 아니었다. 푸는 문제가 달랐을 뿐이다. 이 발견은 별도 결과로 보고한다.

> `Δlog L / GE` 의 누적 평균을 최적화하면 planner가 거의 항상 depth 1을
> 선택했으며, 이는 cost-to-target이나 fixed-budget terminal performance를
> 개선하지 못했다.

이는 **RL 보상을 ratio로 설계할 때 생기는 실제 함정**을 보여준다. D3의 보상
설계 근거를 강화한다.

#### Beam pruning도 비율을 쓰지 않는다

후보를 `Δlog L / c` 스칼라 하나로 정렬하면 같은 문제가 가지치기 안에서 재발한다.
비싼 장기 계획이 싼 단기 계획과 섞여 조기에 탈락한다. 대신 `(used_GE,
terminal_loss)` 의 **Pareto frontier** 를 유지한다.

```text
A 가 B 보다 GE 를 같거나 적게 쓰고 terminal loss 도 같거나 낮으면 B 를 제거
```

Pareto frontier는 크기 상한이 없으므로 계산량 제한을 위해 GE cost bucket을
함께 쓴다. 구간마다 terminal loss가 좋은 후보를 `beam_width` 개 남긴다.

Pareto는 **terminal loss 최소 후보를 절대 지우지 않으므로** incumbent
carry-over를 대체한다. depth 1 최선은 더 나은 계획에 의해서만 밀려난다.

#### Track T planner는 lexicographic

임의의 큰 실패 벌점이나 비율을 넣지 않는다.

```text
1. target 에 도달한 sequence 가 있으면 누적 GE 가 가장 작은 것
2. 아무도 도달하지 못하면 같은 쿼터에서 terminal loss 가 가장 낮은 것
3. 동률이면 더 적은 GE, 그다음 더 짧은 sequence
```

### D11. Track E는 예산을 넘지 않는 prefix에서 평가한다

optimizer 루프는 `spent >= budget` 에서 종료한다. 즉 **마지막 step이 예산을
초과한다.** 초과량은 컨트롤러가 고른 action 크기에 비례하므로, 고정 예산
비교에서 큰 step을 고르는 컨트롤러가 공짜로 이득을 본다.

```text
C0  (평균 k=17.9)   150 GE 예산에 실제 171 GE 소모     <- 큰 step 하나가 공짜
Q=4 (평균 k=3.3)    150 GE 예산에 실제 154 GE 소모
```

약 11% 예산 차이다. 그런데 게이트 C의 쿼터 사다리는 정확히 "쿼터를 키우면
planner가 싼 action을 고른다"는 현상을 다루므로, 이 편향이 결론과 **같은 방향**
으로 섞인다. 즉 편향을 제거하지 않으면 "planning이 나쁘다"는 결론의 일부가
회계 인공물이 된다.

따라서 집계 시 **누적비용이 예산을 넘지 않는 마지막 prefix** 에서 평가한다
(`budget_respecting_prefix`). optimizer의 동역학은 바꾸지 않고, 절단된 step은
raw trace에 남는다. 이렇게 하면 모든 컨트롤러의 `total_cost_ge ≤ budget` 이므로
planner의 쿼터 회계와 의미가 일치한다.

Track T 지표(`cost_to_target_ge`, `reached`)는 목표 도달 시점으로 정의되므로
영향받지 않는다. Track E를 공정하게 만드는 수정이 Track T의 정의를 바꾸면 안 된다.

동일한 버그가 진단 스크립트에도 있었다. `cost_budget_ge=Q` 로 돌린 one-step
참조가 Q=30에서 실제 49.9 GE를 썼고(1.66배), 그 상태로 beam search를 비교해
"탐색 손실"이라고 잘못 판정했다. 동일 비용으로 고치니 18개 조건 중 17개가
`B ≤ A` 였다. **고정 예산 비교에서는 "예산"과 "실제 소모량"을 항상 함께
확인한다.**

### D12. 계획의 가치와 실행 방식을 분리한다

D10 쿼터 사다리에서 "쿼터를 키우면 성능이 나빠진다"가 관측됐다. 그런데 planner가
찾은 **계획 자체**는 동일 비용의 greedy 궤적보다 좋았다. 따라서 문제는 목적함수도
탐색도 아니라 **실행 방식**이었다.

세 방식을 동일 planner, 동일 쿼터, 동일 GE 예산에서 비교한다. 실행 방식만 다르므로
차이가 탐색 품질 차이와 섞이지 않는다.

```text
committed        계획을 끝까지 실행. 재계획 없음. 소진되면 새 window
fresh-quota      매 step 미래 예산 Q 를 새로 지급 (D10 초판 방식)
shrinking-quota  쓴 비용을 차감. horizon 을 새로 연장하지 않음
```

#### 먼저 확인할 불변조건

```text
J_predicted_plan  ≈  J_committed_execution
```

synthetic task는 결정적이므로 planner가 예측한 terminal loss와 그 계획을 끝까지
실행한 결과가 같아야 한다. **이게 맞지 않으면 이후 비교는 의미가 없다.** 12개
조건 전부에서 상대오차 `< 1e-9` 로 일치했다.

#### shrinking은 이전 계획의 suffix를 보장 후보로 포함한다

결정적 환경에서 재계획이 더 나은 것을 못 찾아도 이전 suffix는 유지할 수 있어야
한다. 그렇지 않으면 beam 근사 때문에 재계획 자체가 성능을 떨어뜨리고, 그것이
"피드백이 해롭다"로 오해된다.

단, **탐색에서 살아남는 것은 보장되지만 채택이 보장되는 것은 아니다.** 목적함수가
남은 쿼터 안에서 더 낮은 terminal loss를 찾으면 계획을 버린다. 그 이탈이
국소적으로는 개선이어도 episode 전체로는 손해일 수 있다.

#### 쿼터 차감은 직전 step의 실제 비용으로 한다

예측 비용을 쓰면 CG가 조기 수렴한 만큼 쿼터가 과도하게 줄어들어 window가 일찍
닫힌다. 실측에서 이 차이가 컸다.

```text
예측 비용 차감:  SPD beam8 Q=4  shrinking 48.90   (셋 중 최악)
실제 비용 차감:  SPD beam8 Q=4  shrinking 56.73   (committed 와 동일)
```

`context.previous.cost_ge` 를 쓴다.

#### 실측 결과 (quadratic, seed 0, narrow, 150 GE, max_depth 24)

```text
SPD κ=1e2   C0 = 52.137 nat
  beam 8, Q=4×c_max   committed +4.59   shrinking +4.59   fresh +3.67
  beam 4, Q=4×c_max   committed −3.38   shrinking −4.21   fresh −13.26
  beam 4, Q=2×c_max   committed −2.03   shrinking −1.91   fresh −4.82

ill κ=1e5   C0 = 9.558 nat
  beam 8, Q=4×c_max   committed +0.91   shrinking +0.89   fresh −0.23
  beam 4, Q=4×c_max   committed +0.35   shrinking +0.87   fresh −0.16
  beam 4, Q=2×c_max   committed +0.57   shrinking +0.65   fresh +0.33
```

해석: **`committed > C0`, `shrinking ≈ committed`, `fresh < committed`.**
원인은 쿼터 초기화에 의한 시간 불일치다. fresh-quota는 매 step 미래 예산을 새로
지급하므로 "나중에 이득을 얻을 준비 행동"을 계속 고르면서 payoff를 뒤로 미룬다.
horizon을 연장하지 않으면(shrinking) 피드백 재계획은 committed 대비 손실이 없다.

`beam 4 → 8` 에서 `Q=4` 결과가 SPD에서 8 nat 이상 움직인다. **`Q=4` 조건은 아직
탐색 한계에 걸려 있으므로, 그 수치를 planning 가치의 하한으로만 읽는다.**

탐색 비용은 실제 최적화 비용의 500~2,600배다 (committed 80,311 GE vs fresh
389,245 GE, 본문 150 GE). 이들은 오라클이며 실용 optimizer가 아니다.

---

## 3. 로깅과 provenance

step 단위 JSONL(README §13 스키마)에 다음을 추가한다.

```json
{
  "cost_ge": 26.4,
  "cost_model_id": "rtx3060ti_cu130_b512",
  "git_commit": "a1b2c3d",
  "config_hash": "9f8e7d6c",
  "task_instance_id": "spd_d100_kappa1e3_seed0",
  "failure_tag": null
}
```

원칙:

- 모든 run은 config 스냅샷 + git commit hash를 함께 저장한다. dirty working tree면
  경고를 남기고 diff도 저장한다.
- wall-clock 측정은 warm-up 후 `torch.cuda.synchronize()` 를 앞뒤로 호출한다.
- peak VRAM은 `torch.cuda.max_memory_allocated()` 로 기록한다.
- 실패한 run을 삭제하지 않는다.

---

## 4. 실행 계획

각 Stage 끝에 **게이트**가 있다. 게이트를 통과하지 못하면 다음 Stage로 가지 않고
설계를 수정한다.

### Stage 0 — 환경과 scaffold  (완료)

- uv + venv + pyproject + `uv.lock`
- 패키지 구조, 인터페이스 정의(`types.py`), 동작하는 유틸(flatten/seed/logging/provenance)
- `benchmark/cost_model.py` + `scripts/measure_cost_model.py` 및 실측 산출물
- 이 문서
- 첫 커밋

게이트: `uv run pytest -q` 통과, `torch.cuda.is_available() == True`,
double-backward HVP 상대오차 < `1e-5`. **통과** (71 tests, ruff clean,
GPU HVP 상대오차 `5.8e-8`)

### Stage 1 — 수치 커널  (완료)

비용 모델(`benchmark/cost_model.py`)과 실측은 Stage 0에서 완료했다.
`configs/cost_model.mnist_mlp.yaml`, `configs/cost_model.small_cnn.yaml` 참조.

1. `curvature/hvp.py` — `HvpGraph`. 그래프를 한 번만 만들고 k회 재사용한다.
   그 결과 "한 CG solve 안에서 동일한 curvature batch"(README §15)가
   규율이 아니라 **구조로** 보장된다. 다른 배치를 쓰려면 새 그래프가 필요하다.
2. `curvature/operators.py` — `DampedHessianOperator`, preconditioner 2종.
   damping은 그래프 재사용 중에도 바꿀 수 있다 (step 거절 후 재풀이 시 절약).
3. `solvers/conjugate_gradient.py` — truncated PCG, `CGResult` 반환
4. `tasks/quadratics.py`, `tasks/rosenbrock.py`
5. `benchmark/paired.py` — 결정론적 `seed → task instance` 매핑
6. `scripts/verify_numerics.py` — 게이트를 수치로 보고

게이트: **전체 통과** (CPU / CUDA 양쪽, 183 tests, ruff clean)

| 항목 | 임계값 | 실측 (FP32) |
|---|---|---|
| HVP vs explicit Hessian (κ ≤ 1e3) | < 1e-5 | 4.1e-8 ~ 9.4e-8 |
| HVP, ill-conditioned κ=1e5 | < 1e-4 | 6.1e-8 |
| Newton-CG 방향 vs explicit solve, κ=1e1 | < 1e-3 | 3.7e-7 (27 iters) |
| Newton-CG 방향, κ=1e4 | < 1e-3 | 1.3e-4 (307 iters) |
| damping 증가 → CG 수렴률 | 단조 비감소 | 단조, 최대 damping에서 1.00 |
| indefinite negative curvature 탐지 | 탐지 + damping으로 복구 | 양쪽 확인 |

#### Stage 1에서 발견한 것

세 가지가 초기 가정과 달랐고, 모두 이후 단계에 영향이 있다.

**1. negative curvature 판정은 상대 기준이어야 한다.** `p^T A p <= eps` 처럼
절대 임계값을 쓰면 `p^T A p ∝ ||p||²` 이므로 수렴이 진행되어 `p` 가 작아질 때
양정 행렬에서도 조건이 성립해 **오탐**이 난다. 곡률이 아니라 스케일을 재는 셈이다.
`p^T A p <= eps * ||p||²` 로 바꿨다. RL 상태 특징에 `negative_curvature` 가
들어가므로, 이 오탐은 정책 학습을 직접 오염시킬 수 있었다.

**2. Rosenbrock 표준 시작점은 Hessian이 양정이다.** `det H = 8s²(x² − y) + 4s`
이므로 negative curvature는 `y > x² + 1/(2s)`, 즉 골짜기 **위쪽**에서만 발생한다.
표준 시작점 `(-1.2, 1.0)` 은 `y = 1.0 < x² = 1.44` 로 아래쪽이고 고유값이
23.6, 1506이다. "비볼록 문제이니 시작부터 음의 곡률"이라는 가정은 틀렸다.

**3. `tasks/quadratics` 에서는 Jacobi preconditioner가 원리적으로 무력하다.**
`A = Q diag(λ) Qᵀ` 를 랜덤 직교기저로 만들면 `A` 의 대각이 거의 상수가 된다
(실측 분산 < 10배). Stage 5에서 diagonal preconditioner의 이득이 없다고 나오면
구현 결함이 아니라 문제 구조 때문이다. 대각이 퍼진 계에서 별도로 평가해야 한다.

**4. `κ=1e6` 문제는 damping을 `1e6` 수준까지 올려야 예산 20회 안에 풀린다.**
`1e2` 정도로는 damped 조건수가 여전히 ~1e4다. Stage 2 헤드룸 측정에서
action space의 damping 배수 `{0.3, 1.0, 3.0}` 만으로는 극단적 ill-conditioned
구간에 도달하는 데 여러 step이 걸린다는 뜻이다. 이 점이 헤드룸의 크기에
영향을 줄 수 있으므로, 초기 damping 설정과 배수 범위를 Stage 2에서 함께 본다.

### Stage 2 — 헤드룸 측정  ← 이 프로젝트의 분기점

**목적.** RL 스택을 만들기 전에 적응 제어의 여지가 얼마나 있는지 측정한다.
README 순서대로 가면 RL이 돌아가기까지 2~3주가 걸리고 그때서야 "애초에 이득이
있었나"를 알게 된다.

**초판 설계는 파일럿에서 실패했다.** 단일 `greedy_oracle` 을 상한으로 쓰려 했으나,
그것의 목적(`Δlog L / cost`)과 평가 지표(cost-to-target)가 다른 문제여서 오라클이
고정 설정보다 나쁜 결과를 냈다. D9에 따라 두 트랙으로 분리하고 게이트를 재정의한다.

#### 비교군

```text
best_static           행동 공간 전수 고정 → 최고 선택            (N_tune 회)
best_open_loop        progress 만 보는 스케줄, 랜덤 서치         (N_tune 회)
heuristic             trust ratio 규칙                          (N_tune 회)
one_step_efficiency   매 step 전수 sweep, 즉시 효율 최대 선택     (게이트 C의 C0)
budgeted_Q1/Q2/Q4     동일 미래 GE 쿼터 안의 계획 비교            (게이트 C의 C1~C3)
avgrate_H3/H5         누적 평균 효율 planner                     (진단 baseline, D10)
lagrangian_b*         `Δlog L − β·Σc` planner                    (보조 민감도, D10)
```

`one_step_efficiency` 는 **상한이 아니다** (D9). `mpc_*` 도 유한 horizon 근사다.
행동 공간은 `narrow` / `wide` / `absolute` 세 가지를 쓰며, 세 공간의 **로그
해상도를 맞춘다**. `absolute` 가 범위만 넓고 해상도가 거칠면 게이트 B가 도달성
손실과 해상도 손실을 섞는다 (파일럿에서 실제로 발생).

기본 조건은 **step_size 고정 1.0** 이다. damping이 큰 구간에서 update가
`-(α/λ)g` 로 근사되어 `(λ, α)` 와 `(10λ, 10α)` 가 aliasing되므로, 먼저
`damping × CG budget` 만 분리해 본다. 헤드룸이 확인되면 step_size 축을 추가한다.

#### 계산 자원

**Stage 2는 GPU를 쓰지 않는다.** 대상이 quadratic(d=32~100)과 Rosenbrock(d=2~10)
뿐이므로 CPU가 더 빠르다. Stage 0 실측에서 10만 파라미터 MNIST MLP조차 GPU
런치 오버헤드 지배(0.68 ms/gradient)였으므로 d=100 matvec을 GPU로 보내면 순손실이다.

| 단계 | 디바이스 | VRAM |
|---|---|---|
| Stage 1~2 | CPU (실측) | 0 |
| Stage 2.5 micro-neural | CPU 가능 | ~0 |
| Stage 3 MNIST MLP (102k, B=512) | GPU 권장 | 계획값 < 100 MB |
| Stage 4 PPO (DummyVecEnv) | GPU 또는 CPU | 계획값 수백 MB |
| Stage 5 small CNN (2.19M, B=128) | GPU | 계획값 ~1 GB |

**Stage 3 이후의 VRAM 숫자는 확정값이 아니라 계획값이다.** HVP 메모리는
`create_graph` 유지 기간, 후보를 병렬 평가하는지, mixed precision 여부,
curvature batch 크기, activation 크기에 따라 달라진다. 각 Stage 진입 시
`torch.cuda.max_memory_allocated()` 와 `max_memory_reserved()` 를 다시 실측한다.
PPO controller 자체는 작아서 GPU가 반드시 필요한 것도 아니다.

#### 실행은 재개 가능해야 한다

Stage 2는 컨트롤러 × 행동공간 × horizon × task × seed × target 조합이라 수백~수천
run이 된다. 프로세스가 끊겨도(셸 중단, timeout) 계산한 결과를 잃지 않아야 한다.

- run 하나가 끝나는 즉시 `results/raw/headroom_<tag>.jsonl` 에 append
- 재실행 시 완료된 `(controller, task_instance, seed, target)` 조합은 건너뜀
- 상태를 `completed` / `failed` 로 구분. 미완료는 파일에 없으므로 자동 재시도
- 각 run에 GE, HVP, wall-clock, action 빈도, `chosen_depth` 분포, git commit,
  config hash를 함께 기록
- **raw와 집계를 분리한다.** 집계 로직을 바꿔도 실험을 다시 돌리지 않는다

#### Beam width와 쿼터는 추측이 아니라 측정으로 정한다

beam search는 정확한 planner가 아니므로 폭을 줄이면 계획 품질이 떨어질 수 있고,
그것이 게이트 C의 결론을 바꿀 수 있다. 따라서 **pilot calibration parameter**로
취급한다. D10 이후에는 planning 쿼터와 beam을 **함께** calibration한다.

측정 범위:

```text
beam   ∈ {1, 2, 4}
quota  ∈ {1, 4} × c_max        사다리의 양 끝
space  ∈ {narrow, wide}
seed   dev seed 만 (held-out 100~109 사용 금지)
```

**선택 규칙 (사전 정의. 결과를 본 뒤 바꾸지 않는다):**

1. 최대 beam(4)을 기준으로 삼는다
2. 모든 `(space, quota)` 조합에서 다음 **둘을 모두** 만족하는 beam 중
   **가장 작은 것**을 고른다

   ```text
   |J_b − J_ref| / max(|J_ref|, ε) < 0.02          ε = 1e-6
   |d_b − d_ref| ≤ 0.05                            d = chosen_depth > 1 비율
   ```

3. tie-break: 가장 작은 beam → wall-clock이 짧은 것 → 그래도 같으면 beam 2

수치는 코드 상수로 고정되어 있다 (`UTILITY_TOLERANCE = 0.02`,
`UTILITY_EPSILON = 1e-6`, `DEEP_FRACTION_TOLERANCE = 0.05`). "크게 다르면 제외"를
결과를 본 뒤 판단하면 사후 선택이 되므로 수치로 못박는다.

분모가 `|J_ref| + ε` 이 아니라 `max(|J_ref|, ε)` 인 이유는 `J_ref` 가 0 근처일 때
상대 오차가 폭발하기 때문이다.

`chosen_depth` 를 함께 보는 이유는 효용이 비슷해도 깊은 계획을 쓰는 빈도가
달라지면 게이트 C의 해석이 바뀌기 때문이다. 큰 쿼터를 줬는데도 거의 항상 1이면
장기 planning의 실질적 가치가 낮다는 직접적 증거이고, 이는 PPO 착수 판단에
직결된다.

`depth_cap_hit` 도 함께 기록한다. 0이 아니면 계산 상한 때문에 쿼터를 다 쓰지
못한 step이 있다는 뜻이므로, 그 calibration 행은 쿼터 사다리 비교가 훼손된
것으로 표시한다.

#### wall-clock tie-break를 쓰므로 CPU를 고정한다

GE가 주 지표이므로 핵심 결론은 CPU 경쟁에 영향받지 않는다. 그러나 tie-break에
wall-clock을 쓰므로 스레드를 고정하고 환경을 기록한다.

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

기록 항목: CPU 모델, 코어 수, torch/interop 스레드 수, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, 동시 실행 프로세스 수. **실행 환경이 달라지면 calibration을
재사용하지 않는다.**

#### 재개 판단은 전체 config 해시로 한다

`(controller, task, seed, target)` 만으로 완료를 판단하면 위험하다. beam,
horizon, GE 예산, action space, CG budget, damping 격자 중 어느 하나가 바뀌어도
같은 조합으로 보고 **낡은 결과를 새 결과로 착각**한다. 재개 기능이 오히려 실험을
오염시킨다.

```text
experiment_id = hash(canonicalized_full_config)
run_key       = experiment_id | controller | task_instance | seed | target
```

`experiment_id` payload에 포함되는 것: `protocol_version`, phase, device,
GE 예산, `max_steps`, damping 초기값·경계, CG tolerance, `pap_eps`,
`max_loss_increase_ratio`, `safe_fallback`, horizons, beam, `tuning_budget`,
스케줄 구간 수, tuning seed, target 정의, task spec 목록, seed 목록,
**행동 공간 정의 전체**(damping 값, CG budget, step size), 그리고 `code_dirty`.

git commit은 provenance로 기록하지만 정체성으로는 불충분하다. 커밋되지 않은
변경이 있을 수 있으므로 `code_dirty` 를 함께 넣는다.

`chosen_depth`를 반드시 본다. `H=5`인데 거의 항상 1이면 장기 planning의 실질적
가치가 낮다는 직접적 증거이고, 이는 PPO 착수 판단에 직결된다.

#### 게이트

**Gate A1 — Instantaneous absolute-action headroom (Track E)**

> 현재 상태에서 좋은 damping이 존재하는가?

도달성 제약을 완전히 없앤 `absolute` 행동 공간에서 **H=1**로 측정한다.

```text
H_E(A1) = J_E(absolute, H=1) − J_E(best_static)      [nat]
```

| H_E | 판단 |
|---|---|
| ≥ 1.0 nat (약 2.7배 loss) | GO |
| 0.3 ~ 1.0 nat | 조건부. 주 주장을 강건성으로 이동 |
| < 0.3 nat | 적응 제어 연구를 중단하거나 음성 결과로 정리 |

**"순간적" 헤드룸이다.** 전역 상한이나 장기 헤드룸이라고 부르지 않는다.
`absolute`는 H=3, 5 planning에 쓰지 않는다. 현재 damping과 무관하게 순간
이동하므로 damping ramp-up과 temporal credit assignment 자체를 제거하고,
따라서 장기 계획의 필요성을 묻는 게이트 C의 질문과 무관하다. 비용도 감당할
수 없다 (33 damping × 4 budget = 132 action이면 H=3 beam=4에서 실제 step당
약 1,200회 시뮬레이션).

**Gate A2 — Reachable sequential headroom (Track E)**

> 현실적인 multiplier action으로 그 이득에 접근할 수 있는가?

```text
H_E(A2) = max over {narrow, wide} of J_E(space, H=5) − J_E(best_static)
```

A1 대비 크게 낮으면 행동 공간 도달성이 병목이다. GO 기준 0.7 nat, 재설계 0.2 nat.

**Gate B — Action-space restriction**

```text
absolute  vs  wide multiplier  vs  narrow multiplier      (모두 H=1)
```

세 공간이 **같은 `3^e` 격자** 위에 있어야 이 비교가 성립한다. 해상도가 다르면
도달성 손실과 해상도 손실이 섞인다. 실제로 초기 구성(2 decade 간격)에서
`absolute`가 범위가 4배 넓은데도 `narrow`보다 나쁜 결과를 냈다.

```text
NARROW    {3^-1 .. 3^1}      3점
WIDE      {3^-3 .. 3^3}      7점
ABSOLUTE  {3^-16 .. 3^16}   33점
```

`ABSOLUTE` 범위 `[2.3e-8, 4.3e7]`은 optimizer 경계 `[1e-8, 1e8]` 안에 있다.
경계에서 클립되면 서로 다른 action이 같은 damping으로 붕괴한다.

**CG budget `{3, 5, 10, 20}`은 축소하지 않는다.** 부차적 하이퍼파라미터가 아니라
이 프로젝트의 핵심인 inexactness control 축이다. 특히 파일럿 발견이 "국소 효율
목적이 `k=3`을 과도하게 선호한다"는 것이므로, `k=3`을 삭제하면 문제를 해결하는
게 아니라 관찰된 현상을 action space 밖으로 숨기는 것이 된다.

계산량이 문제라면 순서는 이렇다.

```text
1. absolute 를 H=1 로 제한                    (적용됨)
2. planner 는 narrow / wide 만                 (적용됨)
3. beam width 민감도 측정 후 축소               (calibration)
4. 마지막 수단으로 CG budget 축 축소            (하지 않음)
```

**Gate C — Temporal planning value (미래 GE 쿼터 사다리)**

D10에 따라 재정의됐다. `H = 1, 3, 5` 비교는 폐기했다. 이유는 D10 참조.

`c_max` 를 단일 action 최대 비용이라 할 때, planner마다 동일한 미래 GE 쿼터
`Q` 를 준다. `narrow` / `wide` 만 쓰고 `absolute` 는 제외한다 (A1 참조).

```text
C0  one_step_efficiency        비율 baseline
C1  budgeted MPC  Q = 1 × c_max
C2  budgeted MPC  Q = 2 × c_max
C3  budgeted MPC  Q = 4 × c_max
```

각 planner는 쿼터 안에서 여러 action을 선택할 수 있다. `Q = 1 × c_max` 는
"비싼 action 한 번"과 "싼 action 여러 번"을 같은 예산에서 겨루게 한다.

| 결과 | 판단 |
|---|---|
| `J_E(C3) − J_E(C1) ≥ 0.3 nat` **그리고** `depth > 1` 이 실제로 채택됨 | 순차적 의사결정에 가치가 있다. RL 진행 근거 |
| < 0.05 nat | contextual bandit이나 heuristic이 적절하다. **PPO를 시작하지 않는다** |

**두 조건이 모두 필요하다.** 쿼터를 늘려 개선이 나왔지만 `chosen_depth` 가 계속
1이면, 기여한 것은 planning이 아니라 늘어난 탐색량이다. 그 경우 GO 판정을 내리지
않는다.

함께 보고할 것:

```text
실제 episode 의 고정 GE terminal loss
쿼터별 개선량 (C1 → C2 → C3)
chosen_depth 분포와 최대 채택 depth
quota_used_fraction   쿼터를 실제로 얼마나 썼는가
depth_cap_hit         계산 상한에 걸린 step 비율
narrow 와 wide 의 차이
planner 분석 비용 (search_cost_ge)
action sequence 분포
```

`depth_cap_hit` 이 0이 아니면 계산 상한 때문에 쿼터를 다 쓰지 못한 계획이
있으므로 **사다리 비교가 훼손된 것으로 보고한다.** 조용히 넘기면 게이트 C
결론이 계산 예산의 부산물이 된다.

`C0 → C1` 차이도 별도로 보고한다. 이것은 "목적함수를 비율에서 고정 예산으로
바꾼 효과"이고 "예산을 늘린 효과"와 다르다. 섞으면 어느 쪽이 기여했는지
알 수 없다.

**실현 성능의 단조성을 가정하지 않는다.** MPC는 매 step 재계획하므로 큰 쿼터가
항상 좋다는 보장이 없다. 쿼터가 커지면 탐색 가능 집합이 포함관계로 커지므로
**최대 채택 depth** 는 감소할 수 없고, 그 성질만 테스트로 검증한다.

```text
탐색 가능 집합의 포함관계   ≠   실제 episode 성능의 비감소
```

**보조 분석: Lagrangian planner.** `U_β = Δlog L − β·Σc` 를 최대화하는 변종을
사전 고정 β 격자 `{0, 0.01, 0.03, 0.1, 0.3, 1.0}` 전체에서 돌린다. 특정 β 하나를
골라 주 결과로 쓰지 않는다. discrete action에서 Lagrangian 완화는 고정 예산
문제와 정확히 같지 않고(duality gap이 0이라는 보장이 없음), β 선택에 따라 결론이
뒤집힐 수 있다. 탐색과 가지치기는 `BudgetedMPCController` 와 동일하고 최종 선택
규칙만 다르므로, 목적함수 차이가 탐색 품질 차이와 섞이지 않는다.

**Gate D — Cost-to-target headroom (Track T)**

target 난이도별로 `best_static` 과 MPC planner의 GE-to-target, 도달률,
restricted mean을 비교한다.

```text
H_T(τ) = C_τ(best_static) / C_τ(mpc)      [배수]
```

Gate A와 결론이 다를 수 있다. **그 불일치 자체를 결과로 보고한다.**

**Gate E — Micro-neural transfer (Stage 2.5)**

가장 큰 미지 위험은 synthetic → 신경망 전이다. PPO 전에 확인한다.

- 아주 작은 2-layer MLP, MNIST 일부 샘플
- 짧은 horizon, 고정 배치와 확률적 배치 각각
- absolute / multiplier planner 모두

| 결과 | 판단 |
|---|---|
| synthetic 헤드룸 큼, neural 거의 없음 | PPO 중단. 연구 질문을 synthetic 수치해석으로 축소 |
| absolute 헤드룸 큼, multiplier 만 낮음 | 행동 공간 재설계 |
| H1 이 이미 충분히 좋음 | contextual bandit 우선 |
| look-ahead 만 좋음 | sequential RL 진행 근거 확보 |

**Gate F — Learnability (Stage 4 이후)**

```text
Recovery = (J_learned − J_static) / (J_reachable_planner − J_static)
```

분모는 `absolute` 가 아니라 **정책과 같은 행동 공간을 쓰는 reachable planner**다.
absolute를 분모에 두면 정책이 구조적으로 도달할 수 없는 부분까지 요구하게 된다.

#### 부수 산출물

MPC planner의 trajectory가 behavior cloning 데이터셋이 된다.
README 위험 2번(PPO 불안정)의 대응책으로 적힌 warm start를 여기서 얻는다.
`results/raw/planner/` 에 `(state, action)` 쌍으로 저장한다.

#### 재현성 확인 항목

파일럿에서 나온 "높은 damping은 CG를 쉽게 만들지만 최적화를 망친다"는 결과는
단일 quadratic 조건에서 관측됐다. 기여로 올리기 전에 다음에서 재현되는지 본다.

- 여러 condition number
- 여러 eigenvalue 분포 (log-spaced 외에 clustered, two-cluster)
- Rosenbrock
- 작은 MLP
- step_size 고정과 line search 각각

### Stage 3 — baseline 정면 비교  (3~4일)

Stage 2를 통과하면 README Milestone 2·3에 해당하는 작업을 수행한다.

- `optimizers/fixed_newton_cg.py` — step acceptance, safe fallback
- `optimizers/heuristic_newton_cg.py` — trust ratio 기반 damping 제어
- `benchmark/runner.py` — paired design, JSONL 로깅, 비용 회계
- 대상: `adamw`, `sgd_momentum`, `fixed`, `heuristic`, `best_static`, `open_loop`
- task: SPD/ill-conditioned quadratic, Rosenbrock, MNIST MLP
- seed 5개 이상, `B_c/B_g ∈ {1/4, 1/2, 1}`

게이트:

- MNIST에서 NaN 없이 100 step 완주, loss 지속 감소
- step별 HVP / GE / wall-clock / peak VRAM 기록
- heuristic이 최소 하나의 ill-conditioned task에서 fixed 대비
  failure rate 또는 cost-to-target 개선
- D5 결과 표 자동 생성

이 단계에서 RL 없이도 발표 가능한 결과가 나온다. RL 결과가 놓일 좌표계다.

### Stage 4 — RL 컨트롤러  (1~2주)

- `rl/state_features.py` — README §5.1 특징, running mean/std 정규화, NaN 가드
- `rl/rewards.py` — D3 보상
- `rl/environment.py` — `MultiDiscrete([3, 4, 3])`, termination/truncation 구분
- `rl/train_policy.py` — behavior cloning warm start → PPO

구현 주의사항:

- **병렬화**: Windows + `SubprocVecEnv` + CUDA는 프로세스별 CUDA 컨텍스트로
  8 GB VRAM을 소진한다. 모델이 작으므로 `DummyVecEnv` 로 8~16 env를 한 GPU에 올린다.
  quadratic 단계는 CPU가 더 빠를 수 있으므로 측정해서 결정한다.
- **보상 해킹 감시**: step size를 최소로 깔고 버티기, 의도적 step rejection으로
  실패 패널티 회피. action 히스토그램과 entropy를 매 학습 구간 로깅한다.
- 커리큘럼: random SPD quadratic → ill-conditioned/indefinite → Rosenbrock → MNIST MLP
- 체크포인트와 observation normalization 통계를 함께 저장한다 (둘 중 하나만 저장하면 평가 불가)

게이트:

- `gymnasium.utils.env_checker.check_env` 통과
- random policy로 여러 에피소드 실행 시 crash 없음, observation에 NaN/Inf 없음
- random policy 대비 평균 리턴 향상
- unseen quadratic에서 `fixed` 와 비교 가능한 결과 산출
- deterministic 평가 스크립트로 동일 결과 재현

### Stage 5 — 일반화와 분석  (1주)

meta-train과 meta-test 분포를 수치로 분리한다.

```text
meta-train
  quadratic : κ ~ LogUniform(1e1, 1e4),  d ∈ {50, 100, 200}
  MLP       : width ∈ {64, 128, 256},  init scale ~ Uniform(0.5, 2.0)
  batch     : B_g ∈ {256, 512}

meta-test
  quadratic : κ = 1e5,  d = 500              (조건수·차원 외삽)
  Fashion-MNIST MLP                          (데이터셋 전이)
  더 깊은 MLP (784-256-256-128-10)            (깊이 외삽)
  CIFAR-10 small CNN, 수백만 파라미터급        (구조 전이 + wall-clock 검증)
  B_g = 1024                                 (batch 외삽)
```

CIFAR-10 CNN을 넣는 이유는 정확도가 아니라 **D1의 wall-clock 단서 확보**다.
FLOP 지배 구간에서 결론이 유지되는지 확인해야 한다.

ablation (README §9): 최소 1~8번 전부 수행. 우선순위는
`progress 제거` > `Hessian feature 제거` > `HVP penalty 제거` > `단일 축만 제어`.

---

## 5. 컴퓨트 예산

추정치이며 Stage 1의 실측으로 교정한다.

| 단계 | 추정 소요 (RTX 3060 Ti) |
|---|---|
| Stage 1 마이크로벤치 + 단위 테스트 | 분 단위 |
| Stage 2 헤드룸 (36 config × 20 task × 50 step) | 수십 분 |
| Stage 3 baseline (7 optimizer × 5 seed × 4 task × 3 batch ratio) | 1~3시간 |
| Stage 4 PPO meta-training 1회 (100k env step, MNIST) | 1.5~3시간 |
| Stage 5 meta-test 전체 + ablation | 4~8시간 |

지배 비용은 Stage 4의 **재실행 횟수**다. 3060 Ti에서 meta-training을 무한 반복할 수
없으므로, quadratic에서 하이퍼파라미터를 굳힌 뒤 신경망 task로 올라가는 순서를 지킨다.
Stage 4 재실행이 5회를 넘어가면 contextual bandit 또는 supervised policy imitation으로
축소하는 것을 검토한다 (README 위험 1).

---

## 6. 가설을 반증하는 조건

무엇이 나오면 가설이 틀린 것으로 볼지 미리 적어둔다.

1. Stage 2 헤드룸 < 1.10 → 상태 기반 적응 제어의 여지가 없다
2. RL이 `open_loop` 를 이기지 못한다 → 학습된 것은 적응 제어가 아니라 스케줄이다
3. RL이 `best_static` 를 이기지 못한다 → 동일 탐색 예산에서 이득이 없다
4. iteration 기준으로는 이기지만 GE 기준으로는 진다 → 계산 비용이 이득을 잠식한다
5. meta-test에서 이득이 사라진다 → 정책이 task를 암기했다
6. ablation에서 Hessian feature 제거 후에도 성능이 유지된다 → curvature 신호를 쓰지 않았다

4·5·6은 실패가 아니라 보고 가치가 있는 결과다 (README §17 기준 4). 1·2·3은 설계
변경을 요구하는 실패다.

---

## 7. 최소 성공 기준 (README §17 구체화)

다음 중 하나를 만족하면 의미 있는 결과로 본다. 모두 D7 통계 프로토콜로 검정한다.

1. RL이 `fixed` 대비 동일 target까지 GE를 **10% 이상** 감소
   (기하평균 비율 ≤ 0.90, Wilcoxon p < 0.05, Holm 보정 후)
2. RL이 `heuristic` 대비 meta-test에서 failure rate를 유의미하게 감소
3. RL이 동일 GE 예산에서 더 낮은 loss 또는 더 높은 accuracy 달성
4. 우위가 없더라도, 어떤 조건(조건수 / batch regime / target 수준)에서
   `fixed`·`heuristic`·`AdamW` 중 무엇이 우세한지 재현 가능한 분석 제공

"AdamW를 항상 이긴다"는 성공 기준이 아니다.

---

## 8. 보고 산출물

권장 figure (README §18):

- loss vs optimizer step
- loss vs **cumulative GE** ← 주 그림
- loss vs wall-clock (오버헤드 지배 여부 표기)
- damping over time / CG budget over time
- trust ratio histogram
- task별 cost-to-target (도달률 병기)
- policy action heatmap (state feature 축 기준)
- `greedy_oracle` vs `best_static` vs `rl` 3자 비교 (헤드룸 대비 달성률)

마지막 그림이 이 프로젝트의 핵심 주장을 한 장에 담는다.

---

## 9. 변경 이력

| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-08-01 | 초판. D1~D8 확정, Stage 0~5 정의, target 사전 등록 | — |
| 2026-08-01 | D1 예산 계산을 실측값으로 교정 (26 GE → 17 GE, 24 step → 35 step) | 이론 계수 대신 RTX 3060 Ti 실측 사용. MNIST MLP 오버헤드 지배 확인 |
| 2026-08-01 | 비용 모델 실측을 Stage 1 → Stage 0 으로 이동 | Stage 0에서 이미 완료했고, 이후 모든 지표가 여기에 의존 |
| 2026-08-01 | Stage 1 완료. negative curvature 판정을 상대 기준으로 변경 | 절대 임계값은 수렴 구간에서 오탐. RL 상태 특징을 오염시킬 수 있었다 |
| 2026-08-01 | indefinite quadratic을 진단 전용으로 명시 | 아래로 유계가 아니므로 cost-to-target 과 log 보상이 정의되지 않는다 |
| 2026-08-01 | `max_damping` 1e3 → 1e8, damping을 로그공간 지속 상태로 | Stage 1에서 κ=1e6이 damping ~1e6을 요구함이 확인됨. 이전 값은 그 자체로 병목 |
| 2026-08-01 | damping 배수를 정확한 역수쌍으로 (`0.3` → `1/3`) | `3 × 0.3 = 0.9` 라 배수를 번갈아 고르면 damping이 step당 10% 아래로 표류. `1/3` 이면 `3 × (1/3) = 1` 로 표류 없음 |
| 2026-08-01 | **D9 신설: 실험을 Track E / Track T로 분리** | 파일럿에서 `Δlog L / cost` 목적의 컨트롤러가 cost-to-target에서 best_static보다 나빴다(0.967x). 국소 효율 최대화와 총비용 최소화는 다른 문제다 |
| 2026-08-01 | **D10 신설: planner 목적함수를 비율 → 고정 GE 쿼터로 교체. 게이트 C를 `H=1/3/5` → 쿼터 사다리 `C0~C3` 로 재정의** | dry run에서 `H=3`과 `H=5`가 완전히 동일하고 `depth≥3`이 채택 0회였다. 누적 비율 효용은 step별 rate의 가중 평균이므로(mediant 부등식) depth 1 incumbent가 지나치게 강해져 "지금 손해, 나중에 이득"을 표현할 수 없다. 이 목적함수 아래의 음성 게이트 C는 증거가 아니라 항진명제에 가깝다 |
| 2026-08-01 | `HorizonPlannerController` → `AverageRateEfficiencyPlanner` 로 개명, 진단 baseline으로 보존 | 버그가 아니라 푸는 문제가 달랐다. "RL 보상을 ratio로 설계하면 생기는 함정"의 증거로 별도 보고 |
| 2026-08-01 | Beam pruning을 비율 스칼라 → `(used_GE, terminal_loss)` Pareto + GE cost bucket으로 교체 | 비율로 정렬하면 mediant 문제가 가지치기 안에서 재발한다. 비싼 장기 계획이 싼 단기 계획과 섞여 조기 탈락한다 |
| 2026-08-01 | Track T planner 선택 규칙을 cost-to-go 추정 → lexicographic(도달 여부 → 누적 GE)으로 교체 | 임의의 실패 벌점이나 비율 없이 "도달이 우선, 비용이 그다음"을 순서로 표현한다 |
| 2026-08-02 | **D11 신설: Track E를 예산 초과 step 절단 후 평가** | `spent >= budget` 종료 규칙 때문에 마지막 step이 예산을 넘고, 초과량이 action 크기에 비례한다. 150 GE 예산에서 C0는 171 GE, Q=4는 154 GE를 썼다. 큰 step을 고르는 컨트롤러가 공짜로 11% 예산을 더 쓰는 편향이 게이트 C 결론과 같은 방향으로 섞여 있었다 |
| 2026-08-02 | **D12 신설: 계획의 가치와 실행 방식을 분리. committed / fresh-quota / shrinking-quota 3종 비교** | D10 쿼터 사다리의 음성 결과가 목적함수나 탐색 때문이 아니라 **fresh-quota 실행 방식의 시간 불일치** 때문임이 확인됐다. `committed > C0`, `shrinking ≈ committed`, `fresh < committed`. 최선 조건에서 planning이 C0를 SPD +4.59 nat, ill +0.91 nat 앞선다 |
| 2026-08-02 | shrinking 쿼터 차감을 예측 비용 → `context.previous.cost_ge` 실제 비용으로 교체 | 예측 비용은 CG 조기 수렴을 반영하지 못해 window가 일찍 닫힌다. SPD beam8 Q=4에서 shrinking이 48.90 → 56.73으로 바뀌었다 |
| 2026-08-02 | 게이트 C 통계는 **아직 변경하지 않음**. 위 진단은 pilot 기록으로만 보존 | 사전 등록된 `C3 − C1` 을 결과를 본 뒤 바꾸면 사후 선택이 된다. protocol freeze 전에 공식 재정의한다 |
| 2026-08-01 | **D3 보상을 트랙별로 재정의. per-step ratio 보상 폐기** | ratio 보상은 정책이 `k=3` 같은 싸고 작은 행동만 반복하게 만든다. Track E는 additive log 감소, Track T는 `-cost` + target 종료 |
| 2026-08-01 | `greedy_oracle` → one-step efficiency controller, `lookahead_oracle` → H-step MPC planner | 전역 상한이 아니다. 실제로 고정 설정보다 나쁠 수 있음이 확인됐다 |
| 2026-08-01 | D6에 target 난이도 3단계와 pilot/confirmatory 분리 추가 | target 하나면 그 값 선정이 결론을 좌우한다. 결과를 본 뒤 예산을 고치면 사후 선택이 된다 |
| 2026-08-01 | D5에 탐색 예산 동일화 규칙 강화 | 파일럿에서 open_loop 랜덤 서치 12회가 static 전수 12개와 동일한 결과를 냈다. 스케줄 공간을 사실상 탐색하지 못했다 |
| 2026-08-01 | Stage 2 게이트를 A~F로 재정의, Stage 2.5(micro-neural)를 Gate E로 편입 | 단일 헤드룸 게이트로는 도달성·해상도·시간축 가치가 구분되지 않는다 |
| 2026-08-01 | D1의 wall-clock 주장 범위를 명시적으로 축소 | 단일 GPU 관측을 "wall-clock 논문 불신"으로 일반화할 수 없다. calibration finding으로 위치를 낮춤 |

### PPO 착수 조건 (명시)

다음이 모두 성립할 때만 Stage 4를 시작한다.

1. Gate C에서 `H` 증가에 따른 단조 개선이 확인된다 (순차 의사결정의 가치)
2. Gate E에서 micro-neural 헤드룸이 남아 있다 (synthetic 전용 현상이 아니다)
3. Gate B로 행동 공간이 병목이 아님을 확인했거나, 병목을 고친 공간을 확정했다

하나라도 실패하면 contextual bandit 또는 supervised policy imitation으로 축소하고,
그 판단 근거를 결과로 보고한다.
