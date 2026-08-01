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

- 주 지표: **cost-to-target (GE)** — 하드웨어 독립적, 재현 가능
- 보조 지표: wall-clock — "실제로도 이득이 남는가" 확인용, 오버헤드 지배 구간임을 명시하여 보고
- Stage 5에서 최소 하나의 task는 FLOP 지배 규모(수백만 파라미터급 CNN)로 두어
  wall-clock 결론을 별도 검증한다

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

### D3. 보상은 log-loss 감소 기반

```text
r_t = (log L_t − log L_{t+1}) − β · (cost_t / GE_ref) − γ · I_failure
```

에피소드 리턴이 텔레스코핑되어 다음이 된다.

```text
Return = log(L_0 / L_T) − β · (총 GE 비용) − γ · (총 실패 횟수)
```

- 스케일 프리. loss의 절대 크기와 무관하다.
- 리턴이 곧 "총 loss 감소 자릿수 − 비용"이므로, 정책이 최적화하는 목적과
  보고 지표(cost-to-target)가 같은 방향을 가리킨다.
- `β` 가 "연산 1 GE와 교환할 loss 감소량(nat)"이라는 해석을 갖는다.

초기값: `β = 0.02`, `γ = 1.0`, per-step reward clip `[-5, 5]`.
README의 상대 감소량 형태 `clip((L_t − L_{t+1}) / (|L_t| + ε), −1, 1)` 는 ablation으로 남긴다.

`log L` 이 정의되지 않는 경우(loss ≤ 0)는 quadratic task에서 발생할 수 있으므로,
task별로 loss에 하한 `L_min`을 두거나 `log(L − L*)` 형태(최적값 기지 시)를 쓴다.
결정: **quadratic 계열은 `L* = 0` 으로 구성하고 `log(max(L, 1e-30))` 를 쓴다.**

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

탐색 예산 규칙: 모든 baseline에 **동일한 36회 튜닝 run**을 부여한다.
- `fixed`: damping × step_size × cg_budget grid에서 36개
- `heuristic`: 규칙 임계값 랜덤 서치 36회
- `adamw` / `sgd`: learning rate + weight decay 랜덤 서치 36회
- `open_loop`: 스케줄 파라미터 랜덤 서치 36회
- `rl`: PPO 하이퍼파라미터 탐색 횟수를 기록하고 meta-training GE를 합산

선택은 **meta-train task에서만** 하고, 선택된 설정을 meta-test에 그대로 적용한다.

### D6. 목표치 사전 등록과 절단 규칙

`time-to-target` / `cost-to-target` 은 target을 먼저 정의해야 의미가 있다.

**사전 등록 target** (실험 전 확정, 변경 시 §9에 기록):

| Task | 주 target | 보조 target |
|---|---|---|
| SPD quadratic | `L / L_0 ≤ 1e-6` | `1e-3`, `1e-9` |
| ill-conditioned quadratic | `L / L_0 ≤ 1e-4` | `1e-2` |
| Rosenbrock (2D) | `L ≤ 1e-4` | `1e-2` |
| MNIST MLP | train loss ≤ 0.10 | val acc ≥ 97.0% |
| Fashion-MNIST MLP | train loss ≤ 0.30 | val acc ≥ 87.0% |
| CIFAR-10 small CNN | train loss ≤ 1.00 | val acc ≥ 60.0% |

**절단(censoring) 규칙.** 예산 내 미도달 run은 삭제하거나 최댓값으로 대입하지 않는다.

- `success_rate` = 도달한 run 비율 (별도 보고)
- `cost_to_target` = **도달한 run만의 중앙값** (평균이 아님)
- 두 지표를 항상 함께 보고한다. 하나만 보면 왜곡된다.
- 실패 run도 `results/raw/` 에 보존하고 실패 원인 태그(`nan`, `budget_exhausted`,
  `divergence`, `oom`)를 기록한다.

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

### Stage 1 — 수치 커널  (2~3일)

비용 모델(`benchmark/cost_model.py`)과 실측은 Stage 0에서 완료했다.
`configs/cost_model.mnist_mlp.yaml`, `configs/cost_model.small_cnn.yaml` 참조.

1. `curvature/hvp.py` — double-backward HVP, unused parameter 정책 명시
2. `curvature/operators.py` — `(H + λI)v` 연산자, GGN 확장 지점
3. `solvers/conjugate_gradient.py` — `CGResult` 반환, residual early stop,
   negative curvature 탐지, NaN/Inf 탐지, HVP 카운트
4. `tasks/quadratics.py`, `tasks/rosenbrock.py`
5. `benchmark/paired.py` — 결정론적 `seed → task instance` 매핑

게이트:

- `uv run pytest tests/test_hvp.py tests/test_cg.py -q` 전부 통과
- explicit Hessian 대조 상대오차 < `1e-5` (FP32 ill-conditioned는 `1e-4`)
- SPD quadratic에서 Newton-CG 해의 상대오차 < `1e-3`
- damping 증가 시 ill-conditioned 문제의 CG 실패 감소 확인
- indefinite 문제에서 negative curvature 탐지 정상 동작

### Stage 2 — 헤드룸 측정  (2~3일)  ← 이 프로젝트의 분기점

**목적.** RL 스택을 만들기 전에, 이 문제에 적응 제어의 여지가 얼마나 있는지 측정한다.
README 순서대로 가면 RL이 돌아가기까지 2~3주가 걸리고 그때서야 "애초에 이득이
있었나"를 알게 된다. 이 질문은 여기서 며칠이면 답할 수 있다.

대상: SPD quadratic, ill-conditioned quadratic, indefinite quadratic, Rosenbrock, MNIST MLP.

세 가지를 같은 paired 조건에서 비교한다.

```text
A. best_static          36개 action 조합 각각 고정 → 전부 실행 → 최고 선택
B. best_open_loop       progress 만 보는 스케줄, 랜덤 서치 36회
C. greedy_oracle        매 step 36개 action을 모두 시도해 실제 결과를 보고
                        (Δlog L / cost_GE) 최대인 것을 선택 후 진행
```

C는 매 step 36배 비용이 들지만 작은 task에서는 수 분이다. C가 A를 얼마나 앞서는지가
**어떤 컨트롤러도 실질적으로 넘기 어려운 상한의 대리 지표**다.
(엄밀한 상한은 아니다. greedy는 장기적으로 최적이 아니므로 C를 넘는 정책도 원리적으로
가능하다. 그러나 C가 A와 비슷하다면 상태 기반 제어의 여지가 작다는 강한 신호다.)

**게이트 (go / no-go).** 지표는 `cost_to_target(A) / cost_to_target(C)` 의 기하평균.

| 헤드룸 | 판단 |
|---|---|
| ≥ 1.30 | Stage 3 진행. 주 지표를 cost-to-target으로 유지 |
| 1.10 ~ 1.30 | Stage 3 진행하되 주 주장을 **실패율/강건성**으로 이동 (README §17 기준 2) |
| < 1.10 | **중단하고 재설계.** task 분포를 더 어렵게(조건수 범위 확대, indefinite 비중 증가) 하거나 action space를 확장 |

**부수 산출물.** C의 trajectory가 그대로 behavior cloning 데이터셋이 된다.
README 위험 2번(PPO 불안정)의 대응책으로 적힌 warm start를 여기서 공짜로 얻는다.
`results/raw/oracle/` 에 `(state, action)` 쌍으로 저장한다.

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
