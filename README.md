# RL-Controlled Hessian-Free Newton Optimization

> 강화학습 에이전트가 Newton–CG의 계산 예산과 안정화 파라미터를 조절하여, 고정된 2차 최적화 기법보다 더 나은 **time-to-quality**를 달성할 수 있는지 검증하는 실험 프로젝트

## 1. 프로젝트 개요

딥러닝에서 고전적인 Newton step은 다음 선형계를 푸는 문제로 표현된다.

\[
(H_t + \lambda_t I)p_t = -g_t,
\qquad
\theta_{t+1}=\theta_t+\alpha_t p_t
\]

- \(g_t=\nabla_\theta L(\theta_t)\): gradient
- \(H_t=\nabla^2_\theta L(\theta_t)\): Hessian
- \(\lambda_t\): damping
- \(p_t\): Newton direction
- \(\alpha_t\): step size

대규모 신경망에서는 Hessian을 직접 생성하거나 역행렬을 계산할 수 없다. 따라서 본 프로젝트는 Hessian-vector product(HVP)와 Conjugate Gradient(CG)를 이용하는 **matrix-free Newton–CG**를 기본 optimizer로 사용한다.

핵심 연구 아이디어는 RL 에이전트가 수백만 차원의 업데이트 방향을 직접 출력하도록 하는 것이 아니다. 대신 에이전트가 매 optimization step에서 다음 제어값을 선택하게 한다.

- damping 크기
- CG 최대 반복 횟수
- Newton direction의 step size
- 선택적으로 curvature 갱신 주기와 preconditioner

즉, 본 프로젝트는 다음 구조를 검증한다.

```text
Neural network training environment
            ↓ state
RL controller
            ↓ damping / CG budget / step size
Hessian-free Newton–CG solver
            ↓ parameter update
New loss, residual, compute cost
            └──────── reward ────────┘
```

---

## 2. 연구 주제

### 제안 제목

**Reinforcement Learning-Based Adaptive Control of Hessian-Free Newton Optimization**

한국어 제목:

**Hessian-Free Newton 최적화의 계산 예산과 감쇠 계수를 제어하는 강화학습 기반 적응형 Optimizer**

### 핵심 연구 질문

1. 고정된 damping과 CG 반복 횟수를 사용하는 Newton–CG보다 RL controller가 더 빠르게 목표 loss 또는 accuracy에 도달하는가?
2. RL controller가 계산량을 무조건 증가시키지 않고, 필요한 구간에서만 HVP와 CG 계산을 늘리는가?
3. 작은 synthetic problem과 MLP에서 학습한 정책이 다른 데이터셋, 모델 깊이, 초기화에 일반화되는가?
4. RL controller의 이득이 단순 heuristic trust-region controller보다 유의미한가?
5. iteration 수가 아니라 wall-clock, HVP 수, peak VRAM까지 고려해도 이득이 유지되는가?

### 중심 가설

> 학습 초반·곡률 변화가 큰 구간·평탄한 구간에서 요구되는 damping과 CG 정확도는 서로 다르다. 따라서 고정 hyperparameter보다 학습 상태를 관찰하는 정책이 계산 비용 대비 더 효율적인 Newton step을 선택할 수 있다.

---

## 3. 범위와 비범위

### 본 프로젝트가 수행하는 것

- PyTorch 기반 matrix-free Hessian-vector product 구현
- damped Newton–CG optimizer 구현
- 고정 hyperparameter baseline 구현
- trust-ratio 기반 heuristic controller 구현
- RL controller 구현 및 meta-training
- loss, accuracy, HVP 횟수, wall-clock, VRAM을 포함한 공정한 benchmark
- synthetic problem에서 시작하여 작은 신경망으로 확장

### 초기 버전에서 수행하지 않는 것

- 전체 Hessian 또는 Hessian inverse의 명시적 저장
- 수백만 개 파라미터의 update를 RL agent가 직접 출력
- 대형 LLM pretraining
- 처음부터 K-FAC, Shampoo, distributed training까지 모두 구현
- AdamW보다 모든 조건에서 우수한 범용 optimizer 주장

K-FAC preconditioning, low-rank Hessian, Tiny Transformer 실험은 기본 실험이 안정적으로 완료된 뒤 확장한다.

---

## 4. 실험에서 검증할 Optimizer

### 4.1 AdamW baseline

일반적인 1차 optimizer 기준선이다.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=...)
```

### 4.2 Fixed Newton–CG

항상 동일한 설정을 사용한다.

```text
damping = 1e-2
cg_max_iters = 10
cg_tolerance = 1e-3
step_size = 1.0
```

목적은 RL 방식이 아니라 Newton–CG 자체의 효과와 비용을 확인하는 것이다.

### 4.3 Heuristic Adaptive Newton–CG

실제 감소량과 quadratic model의 예측 감소량 비율을 사용한다.

\[
\rho_t =
\frac{L(\theta_t)-L(\theta_t+p_t)}
{-g_t^\top p_t-\frac12p_t^\top H_tp_t}
\]

예시 규칙:

```text
rho < 0.25       → damping × 3, step size 감소
0.25 ≤ rho < 0.75 → 현재 설정 유지
rho ≥ 0.75       → damping × 0.5, step size 증가 가능
```

이 baseline은 “RL이 단순 적응 규칙보다 실제로 필요한가?”를 검증하기 위해 필수다.

### 4.4 RL-Controlled Newton–CG

RL agent는 optimizer 자체가 아니라 Newton–CG solver의 controller다.

기본 action은 다음 세 가지로 제한한다.

```text
damping multiplier ∈ {0.3, 1.0, 3.0}
cg budget          ∈ {3, 5, 10, 20}
step size          ∈ {0.25, 0.5, 1.0}
```

총 36개의 조합을 직접 하나의 categorical action으로 표현하거나, Gymnasium의 `MultiDiscrete([3, 4, 3])`로 표현한다.

초기 구현에서는 Stable-Baselines3 PPO의 `MlpPolicy`와 `MultiDiscrete` action space를 권장한다.

---

## 5. 강화학습 환경 설계

### 5.1 상태 공간

모든 값은 running mean/std, log transform, clipping을 통해 정규화한다.

권장 state vector:

| Feature | 설명 |
|---|---|
| `log_loss` | 현재 loss의 로그값 |
| `relative_loss_change` | 직전 step 대비 loss 변화율 |
| `log_grad_norm` | gradient norm |
| `grad_cosine` | 이전 gradient와 현재 gradient의 cosine similarity |
| `cg_residual_ratio` | 최종 CG residual / 초기 residual |
| `cg_iters_used_ratio` | 사용한 CG 반복 수 / 최대 반복 수 |
| `directional_derivative` | \(g^\top p\) 정규화값 |
| `curvature` | \(p^\top Hp\) 정규화값 |
| `predicted_reduction` | quadratic model이 예측한 감소량 |
| `trust_ratio` | 실제 감소량 / 예측 감소량 |
| `log_damping` | 현재 damping |
| `previous_step_size` | 직전 step size |
| `previous_cg_budget` | 직전 CG 예산 |
| `step_accepted` | 직전 step 성공 여부 |
| `progress` | 현재 step / 전체 step budget |

NaN, Inf, CG breakdown 발생 여부도 binary feature로 추가할 수 있다.

### 5.2 행동 공간

#### 기본 버전

```python
spaces.MultiDiscrete([3, 4, 3])
```

각 action index는 다음으로 변환한다.

```python
DAMPING_MULTIPLIERS = [0.3, 1.0, 3.0]
CG_BUDGETS = [3, 5, 10, 20]
STEP_SIZES = [0.25, 0.5, 1.0]
```

#### 확장 버전

- preconditioner: none / diagonal Adam state / Hessian diagonal
- curvature update interval: 1 / 5 / 10 step
- Hessian 대신 GGN 사용 여부
- negative curvature 발견 시 fallback 방식

확장 action은 기본 실험이 완료된 뒤 추가한다.

### 5.3 보상 함수

RL agent가 CG 반복을 무한히 늘리지 못하도록 loss 감소와 계산 비용을 동시에 반영한다.

권장 기본 보상:

\[
r_t =
\operatorname{clip}
\left(
\frac{L_t-L_{t+1}}{|L_t|+\epsilon},
-1,1
\right)
-eta\frac{N_{HVP,t}}{N_{HVP,\max}}
-\\gamma I_{\text{failure}}
\]

초기값 예시:

```text
beta = 0.05
gamma = 1.0
```

추가 penalty 후보:

- NaN/Inf 발생
- loss 폭증
- line search 실패
- step rejection
- VRAM OOM

주의: RL 학습 중 wall-clock을 직접 reward로 사용하면 시스템 노이즈가 크다. 학습 reward에는 HVP 횟수와 forward/backward 횟수를 사용하고, 최종 평가는 실제 wall-clock으로 수행한다.

### 5.4 에피소드

하나의 에피소드는 하나의 모델 학습 run 또는 짧은 truncated training horizon이다.

권장 curriculum:

1. synthetic quadratic: 30–50 optimizer steps
2. Rosenbrock: 50–100 steps
3. MNIST tiny MLP: 50–200 mini-batch updates
4. Fashion-MNIST 또는 작은 CNN: 100–300 updates

긴 전체 학습을 매 에피소드 수행하면 meta-training 비용이 지나치게 커지므로 초기에는 짧은 horizon을 사용한다.

---

## 6. Newton–CG 구현 요구사항

### 6.1 Hessian-vector product

전체 Hessian을 만들지 않고 다음 항등식을 사용한다.

\[
Hv = \nabla_\theta (g^\top v)
\]

PyTorch 구현은 `torch.autograd.grad(..., create_graph=True)` 또는 `torch.func`를 사용한다.

필수 테스트:

- 작은 quadratic problem에서 explicit Hessian multiplication과 HVP 결과 일치
- 상대오차 `1e-5` 이하, FP32에서는 문제 조건수에 따라 `1e-4` 허용
- 파라미터 shape 보존
- unused parameter 처리 정책 명시

### 6.2 Conjugate Gradient

CG는 다음 시스템을 근사적으로 푼다.

\[
(H+\lambda I)p=-g
\]

필수 기능:

- residual 기반 early stopping
- maximum iteration budget
- negative curvature 또는 `pAp <= eps` 탐지
- NaN/Inf 탐지
- 사용한 HVP 횟수 반환
- 초기/최종 residual 반환
- optional preconditioner interface

권장 반환 형식:

```python
@dataclass
class CGResult:
    solution: Tensor
    iterations: int
    hvp_count: int
    initial_residual: float
    final_residual: float
    converged: bool
    negative_curvature: bool
    numerical_failure: bool
```

### 6.3 Step acceptance와 fallback

RL agent가 위험한 action을 선택해도 전체 학습이 즉시 붕괴하지 않도록 안전장치를 둔다.

1. candidate step의 loss를 평가한다.
2. loss가 finite하지 않거나 허용 범위 이상 증가하면 step을 거절한다.
3. damping을 증가시킨다.
4. 선택적으로 clipped gradient step으로 fallback한다.

예시:

```text
if candidate_loss is NaN/Inf:
    reject step
    damping *= 10
    apply no-op or safe gradient fallback

if candidate_loss > current_loss * 1.5:
    reject step
    damping *= 3
```

RL agent가 fallback을 악용하지 못하도록 실패 penalty를 reward에 반영한다.

---

## 7. 실험 단계

## Phase 0 — 수치 정확성 검증

### 문제

- SPD quadratic
- ill-conditioned SPD quadratic
- indefinite quadratic
- Rosenbrock function

### 목표

- HVP correctness 확인
- CG convergence 확인
- damping에 따른 condition number와 반복 수 변화 확인
- negative curvature 처리 확인
- explicit Newton solve와 Newton–CG 방향 비교

### 성공 조건

- SPD quadratic에서 Newton–CG solution 상대오차 `1e-3` 이하
- damping 증가 시 ill-conditioned problem의 CG failure 감소
- indefinite problem에서 negative curvature 또는 step rejection 정상 탐지

---

## Phase 1 — 고정·휴리스틱·RL controller 비교

### 데이터셋과 모델

| Dataset | Model | 목적 |
|---|---|---|
| MNIST | MLP 784-128-10 | 가장 저렴한 신경망 benchmark |
| Fashion-MNIST | MLP 또는 small CNN | MNIST보다 어려운 일반화 실험 |
| CIFAR-10 subset | small CNN | convolution 구조에서의 검증 |

8GB VRAM 기준 권장 시작 설정:

```text
MNIST batch size: 256–1024
Fashion-MNIST batch size: 256–512
CIFAR-10 batch size: 64–256
```

Newton–CG의 HVP 비용 때문에 전체 dataset full-batch보다 고정된 curvature mini-batch를 사용한다. 하나의 CG solve 내부에서는 반드시 동일한 curvature batch를 유지한다.

### 비교 대상

- AdamW
- SGD with momentum
- Fixed Newton–CG
- Heuristic Adaptive Newton–CG
- RL-Controlled Newton–CG

### 공정성 원칙

모든 optimizer는 다음 조건을 분리해 비교한다.

1. 동일한 sample budget
2. 동일한 optimizer step budget
3. 동일한 wall-clock budget
4. 동일한 hyperparameter search budget
5. 동일한 random seed 목록

Newton–CG는 한 step이 비싸기 때문에 step 수만 비교하면 안 된다.

---

## Phase 2 — 정책 일반화

훈련 분포와 평가 분포를 분리한다.

### Meta-train tasks

- random SPD quadratics
- Rosenbrock variants
- MNIST MLP with random width and initialization

### Meta-test tasks

- 학습에 없던 condition number의 quadratic
- Fashion-MNIST
- 더 깊거나 더 넓은 MLP
- 다른 batch size
- 다른 initialization scale
- CIFAR-10 small CNN

### 핵심 질문

- 정책이 특정 모델의 loss curve를 외운 것인가?
- curvature와 residual이라는 optimizer-level signal을 학습한 것인가?

---

## Phase 3 — 선택 확장

기본 실험이 성공한 경우에만 수행한다.

- diagonal preconditioned CG
- Adam second moment를 CG preconditioner로 사용
- Hutchinson Hessian diagonal
- GGN/Fisher-vector product
- low-rank Lanczos preconditioner
- Tiny Transformer character-level language modeling
- agent가 curvature update interval까지 선택

---

## 8. 평가 지표

### 최적화 성능

- train loss
- validation loss
- validation accuracy
- target loss 도달 성공률
- target accuracy 도달 성공률

### 계산 효율

- time-to-target loss
- time-to-target accuracy
- 총 wall-clock
- optimizer step당 시간
- 총 HVP 횟수
- 총 CG 반복 횟수
- forward/backward 호출 횟수
- peak GPU memory

### 수치 안정성

- NaN/Inf 발생 횟수
- step rejection 비율
- negative curvature 탐지 횟수
- CG convergence 비율
- 최종 residual ratio
- damping 분포
- trust ratio 분포

### RL 정책 분석

- 학습 단계별 action 빈도
- loss plateau에서 선택한 CG budget
- 큰 gradient 구간에서 damping 선택
- action entropy
- unseen task에서의 action 변화

---

## 9. Ablation Study

최소 ablation:

1. RL state에서 Hessian 관련 feature 제거
2. reward에서 HVP penalty 제거
3. damping만 제어
4. CG budget만 제어
5. step size만 제어
6. heuristic controller와 동일한 action space 사용
7. trust ratio feature 제거
8. policy meta-train task 종류 변경

추가 ablation:

- PPO vs DQN
- discrete action vs continuous action
- exact Hessian HVP vs GGN-vector product
- preconditioner 없음 vs diagonal preconditioner
- 긴 horizon vs 짧은 horizon

---

## 10. 권장 저장소 구조

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── quadratic.yaml
│   ├── mnist.yaml
│   ├── fashion_mnist.yaml
│   └── cifar10_small.yaml
├── docs/
│   ├── hessian_inverse_report.md
│   ├── experiment_protocol.md
│   └── results_template.md
├── src/
│   └── rl_newton/
│       ├── __init__.py
│       ├── curvature/
│       │   ├── hvp.py
│       │   ├── diagonal.py
│       │   └── operators.py
│       ├── solvers/
│       │   ├── conjugate_gradient.py
│       │   ├── preconditioners.py
│       │   └── line_search.py
│       ├── optimizers/
│       │   ├── fixed_newton_cg.py
│       │   ├── heuristic_newton_cg.py
│       │   └── controlled_newton_cg.py
│       ├── rl/
│       │   ├── environment.py
│       │   ├── state_features.py
│       │   ├── rewards.py
│       │   └── train_policy.py
│       ├── tasks/
│       │   ├── quadratics.py
│       │   ├── rosenbrock.py
│       │   ├── datasets.py
│       │   └── models.py
│       ├── benchmark/
│       │   ├── runner.py
│       │   ├── profiler.py
│       │   ├── metrics.py
│       │   └── plotting.py
│       └── utils/
│           ├── seed.py
│           ├── flatten.py
│           └── logging.py
├── scripts/
│   ├── verify_hvp.py
│   ├── run_quadratic_benchmark.py
│   ├── train_rl_controller.py
│   ├── evaluate_controller.py
│   └── run_all_baselines.py
├── tests/
│   ├── test_hvp.py
│   ├── test_cg.py
│   ├── test_newton_step.py
│   ├── test_environment.py
│   └── test_reproducibility.py
└── results/
    ├── raw/
    ├── checkpoints/
    ├── figures/
    └── summaries/
```

---

## 11. 설정 파일 예시

```yaml
seed: 42
device: cuda
precision: fp32

experiment:
  task: mnist
  model: mlp
  total_steps: 200
  eval_interval: 20
  curvature_batch_size: 512
  gradient_batch_size: 512

newton_cg:
  initial_damping: 0.01
  min_damping: 1.0e-6
  max_damping: 1.0e3
  cg_tolerance: 1.0e-3
  cg_max_iters: 20
  step_size: 1.0
  max_loss_increase_ratio: 1.5
  safe_fallback: gradient

controller:
  type: ppo
  state_normalization: true
  action_space: multidiscrete
  damping_multipliers: [0.3, 1.0, 3.0]
  cg_budgets: [3, 5, 10, 20]
  step_sizes: [0.25, 0.5, 1.0]

reward:
  hvp_penalty: 0.05
  failure_penalty: 1.0
  reward_clip: 1.0

logging:
  backend: tensorboard
  save_raw_metrics: true
```

---

## 12. 실행 인터페이스

예상 CLI:

```bash
# 수치 정확성 검증
python scripts/verify_hvp.py --device cuda
python scripts/run_quadratic_benchmark.py --config configs/quadratic.yaml

# RL controller 학습
python scripts/train_rl_controller.py \
  --config configs/mnist.yaml \
  --output results/checkpoints/ppo_controller

# baseline 전체 실행
python scripts/run_all_baselines.py \
  --config configs/mnist.yaml \
  --seeds 0 1 2 3 4

# 학습된 정책 평가
python scripts/evaluate_controller.py \
  --checkpoint results/checkpoints/ppo_controller/best_model.zip \
  --config configs/fashion_mnist.yaml \
  --seeds 0 1 2 3 4
```

---

## 13. 로그 스키마

각 optimizer step마다 최소 다음 정보를 JSONL 또는 CSV로 저장한다.

```json
{
  "run_id": "mnist_rl_seed0",
  "seed": 0,
  "optimizer": "rl_newton_cg",
  "step": 17,
  "train_loss_before": 0.842,
  "train_loss_after": 0.791,
  "validation_loss": 0.815,
  "validation_accuracy": 0.764,
  "grad_norm": 2.13,
  "damping": 0.03,
  "step_size": 0.5,
  "cg_budget": 10,
  "cg_iterations": 7,
  "hvp_count": 8,
  "initial_residual": 2.13,
  "final_residual": 0.004,
  "trust_ratio": 0.83,
  "predicted_reduction": 0.061,
  "actual_reduction": 0.051,
  "step_accepted": true,
  "negative_curvature": false,
  "numerical_failure": false,
  "step_wall_time_sec": 0.142,
  "peak_vram_mb": 1842
}
```

최종 summary에는 seed별 평균과 표준편차를 기록한다.

---

## 14. Coding Agent 작업 지시

Coding Agent는 다음 순서를 반드시 지킨다.

### Milestone 1 — 기반 수치 모듈

1. 프로젝트 scaffold 생성
2. tensor list flatten/unflatten utility 구현
3. HVP 구현
4. explicit Hessian과 비교하는 unit test 작성
5. CG solver 구현
6. SPD/ill-conditioned/indefinite quadratic test 작성

완료 조건:

```bash
pytest tests/test_hvp.py tests/test_cg.py -q
```

모든 테스트 통과.

### Milestone 2 — Newton–CG baseline

1. fixed Newton–CG 구현
2. step acceptance 구현
3. adaptive damping과 safe fallback 구현
4. Rosenbrock와 MNIST MLP 실행
5. AdamW와 기본 결과 비교

완료 조건:

- MNIST에서 loss가 지속적으로 감소
- NaN 없이 100 step 실행
- step당 HVP와 wall-clock 기록

### Milestone 3 — Heuristic controller

1. predicted reduction 계산
2. trust ratio 계산
3. trust-ratio 기반 damping 조절
4. fixed baseline과 동일한 benchmark 실행

완료 조건:

- heuristic이 적어도 하나의 ill-conditioned task에서 fixed controller보다 failure rate 또는 HVP-to-target을 개선

### Milestone 4 — Gymnasium 환경

1. optimizer state → observation 변환
2. action → Newton–CG 설정 변환
3. reward 구현
4. `reset`, `step`, termination 구현
5. `gymnasium.utils.env_checker.check_env` 통과

완료 조건:

- random policy로 여러 episode를 실행해도 환경이 crash하지 않음
- observation에 NaN/Inf가 없음

### Milestone 5 — RL meta-training

1. Stable-Baselines3 PPO 연결
2. synthetic quadratic curriculum에서 학습
3. checkpoint와 normalization statistics 저장
4. deterministic evaluation script 작성

완료 조건:

- random policy보다 평균 return 향상
- unseen quadratic에서 fixed controller와 비교 가능한 결과 생성

### Milestone 6 — 신경망 실험

1. MNIST MLP meta-train 또는 fine-tuning
2. Fashion-MNIST meta-test
3. seed 5개 이상 실행
4. summary table과 figure 자동 생성

완료 조건:

다음 표를 자동 생성할 수 있어야 한다.

| Optimizer | Target 도달 시간 | HVP 수 | Peak VRAM | Final Accuracy | Failure Rate |
|---|---:|---:|---:|---:|---:|

---

## 15. 구현 원칙

- `torch.linalg.inv`를 optimizer update에 사용하지 않는다.
- 전체 Hessian을 신경망 학습 중 생성하지 않는다.
- 한 CG solve 내부에서는 동일한 curvature batch를 사용한다.
- curvature 관련 누적과 residual 계산은 FP32를 기본으로 한다.
- mixed precision은 baseline이 안정화된 뒤 도입한다.
- Python loop보다 tensor operation을 우선하지만, 정확성 검증 전에는 premature optimization을 피한다.
- 모든 benchmark는 warm-up 이후 CUDA synchronization을 포함해 측정한다.
- wall-clock 측정 전후에 `torch.cuda.synchronize()`를 호출한다.
- GPU peak memory는 `torch.cuda.max_memory_allocated()`로 기록한다.
- 각 experiment는 config와 git commit hash를 함께 저장한다.
- 실패한 run도 삭제하지 않고 실패 원인을 기록한다.

---

## 16. 예상 위험과 대응

### 위험 1: RL meta-training 비용이 지나치게 큼

대응:

- synthetic task에서 먼저 학습
- horizon 단축
- model과 dataset cache
- 작은 MLP 사용
- 정책 학습과 최종 benchmark를 분리
- 필요하면 RL을 contextual bandit 또는 supervised policy imitation으로 축소

### 위험 2: PPO가 optimizer 환경에서 불안정함

대응:

- action discretization
- observation normalization
- reward clipping
- curriculum learning
- random policy 및 heuristic imitation으로 warm start
- DQN 또는 contextual bandit baseline 추가

### 위험 3: Newton–CG가 AdamW보다 wall-clock에서 불리함

이는 실패가 아니라 중요한 연구 결과다. 다음을 분석한다.

- iteration 절감과 wall-clock 증가의 trade-off
- agent가 CG budget을 줄이는지
- 어느 condition number와 batch regime에서 이득이 발생하는지
- target quality가 높아질수록 상대적 이점이 달라지는지

### 위험 4: Hessian의 음의 곡률로 CG가 불안정함

대응:

- damping
- truncated CG
- negative curvature detection
- GGN/Fisher 확장
- step rejection과 safe fallback

### 위험 5: 정책이 특정 task를 암기함

대응:

- task distribution randomization
- model width/depth randomization
- condition number randomization
- held-out dataset과 initialization 사용
- state feature ablation

---

## 17. 최소 성공 기준

본 프로젝트는 다음 중 하나를 만족하면 의미 있는 결과로 간주한다.

1. RL controller가 fixed Newton–CG보다 동일 target loss까지 필요한 HVP를 10% 이상 감소시킨다.
2. RL controller가 heuristic controller보다 unseen task에서 failure rate를 유의미하게 낮춘다.
3. RL controller가 동일 wall-clock에서 더 낮은 loss 또는 더 높은 accuracy를 달성한다.
4. RL의 우위가 없더라도, 어떤 조건에서 heuristic과 fixed controller가 더 나은지 재현 가능한 분석을 제공한다.

“AdamW를 항상 이긴다”는 성공 기준이 아니다.

---

## 18. 결과 보고 형식

최종 보고서는 다음 질문에 답해야 한다.

1. RL agent는 어떤 학습 구간에서 CG budget을 늘렸는가?
2. damping 정책은 gradient norm과 trust ratio에 어떻게 반응했는가?
3. iteration 기준과 wall-clock 기준 결론이 서로 달랐는가?
4. synthetic task에서 배운 정책이 신경망 task로 전이되었는가?
5. RL의 성능 향상이 계산 비용을 정당화했는가?
6. 가장 강한 baseline은 fixed, heuristic, AdamW 중 무엇이었는가?
7. 실패 사례는 curvature noise, negative curvature, policy generalization 중 무엇 때문이었는가?

권장 figure:

- loss vs optimizer step
- loss vs wall-clock
- loss vs cumulative HVP
- damping over time
- CG budget over time
- trust ratio histogram
- task별 target 도달 시간
- policy action heatmap

---

## 19. 참고 연구

- Pearlmutter, **Fast Exact Multiplication by the Hessian** — HVP의 고전적 기반
- Martens, **Deep Learning via Hessian-Free Optimization** — Hessian-free Newton–CG
- Martens and Grosse, **Optimizing Neural Networks with Kronecker-factored Approximate Curvature** — K-FAC
- Li and Malik, **Learning to Optimize** — optimizer를 RL policy로 표현
- Andrychowicz et al., **Learning to Learn by Gradient Descent by Gradient Descent** — learned optimizer
- Bello et al., **Neural Optimizer Search with Reinforcement Learning** — RL 기반 optimizer 수식 탐색
- Yao et al., **AdaHessian** — diagonal Hessian 추정 기반 optimizer
- Liu et al., **Sophia** — scalable stochastic second-order optimization

관련 조사 보고서는 다음 경로에 저장하는 것을 권장한다.

```text
docs/hessian_inverse_report.md
```

---

## 20. 한 문장 요약

> 본 실험은 RL이 Newton direction을 직접 생성하도록 하지 않고, Hessian-free Newton–CG가 언제 얼마나 정밀하게 계산할지를 제어하게 함으로써 2차 최적화의 계산비용과 수렴성 사이의 trade-off를 학습할 수 있는지 검증한다.
