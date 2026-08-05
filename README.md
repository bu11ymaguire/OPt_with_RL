# Planning and feedback in Hessian-free Newton control

Hessian-free Newton-CG 에서 **damping 과 CG 반복 예산을 최적화 도중 조절하는 것**의
이득을 사다리로 분해한 실험 코드다.

핵심 질문은 성능 향상 여부가 아니라 그 향상이 어디서 오는가다.

> 상태 의존적 계산 자원 제어의 이득은 다단계 planning 에서 발생하는가,
> 실행 중 feedback 에서 발생하는가?

## 결과 요약

고정 예산 `150 GE` 에서 `d=100` ill-conditioned quadratic 네 개
(`κ ∈ {10³, 10⁴, 10⁵, 10⁶}`)를 썼다. 설정을 dev seed 에서 한 번 고정하고
**분리된 held-out seed 40 인스턴스**에서 측정했다.

| 튜닝 상수 대비 (paired median, n=40) | median | 95% CI | p |
|---|---|---|---|
| 자원 시계 open-loop 스케줄 | +0.395 | [+0.350, +0.476] | <0.0001 |
| 1-step greedy 제어 | +1.155 | [+1.092, +1.811] | <0.0001 |
| 초기 상태 고정 계획 (committed) | +2.090 | [+1.532, +2.407] | <0.0001 |
| 매 step 재계획 (MPC) | +1.690 | [+1.462, +2.368] | <0.0001 |

| 직접 측정한 증분 | median | 95% CI | p | 양수 |
|---|---|---|---|---|
| 다단계 lookahead (`MPC − 1-step`) | +0.456 | [+0.254, +0.720] | <0.0001 | 35/40 |
| 실행 중 재계획 (`MPC − committed`) | +0.010 | [−0.033, +0.053] | 0.97 | 21/40 |

> 다단계 lookahead 의 추가 가치는 확인됐다. 실행 중 재계획의 **실용적으로 큰 이득은
> 관측되지 않았다.**

**위 표의 값을 서로 빼지 마라.** 쌍별 차이의 median 은 선형이 아니다. `committed` 의
상수 대비 값이 `MPC` 보다 크지만 직접 측정한 `MPC − committed` 는 `+0.010` 이다.

`docs/results_stage2.md` 가 원본 결과에서 자동 생성되는 전체 표다.

### 우리가 주장하지 않는 것

```text
"feedback 은 쓸모없다" / "효과가 0 임을 증명했다"
  equivalence margin 을 사전 등록하지 않았다

"강화학습이 optimizer 제어에 실패한다"
  정책을 학습하지 않았다. 측정한 것은 oracle planner 의 헤드룸 분해다

"헤드룸이 condition number 에 따라 증가/감소한다"
  비단조를 관측했다. 어느 방향의 단조성도 주장하지 않는다

"결정론적 환경이 stochastic 환경보다 N배 좋다"
  GE 는 regime 내부에서만 compute-matched 다
```

전체 목록은 `paper/claim_ledger.md` 의 `NOT SUPPORTED` 절에 있다.

## 왜 정책을 학습하지 않았는가

두 가지를 분리해 기록했다.

```text
Scientific conclusion
  실험은 feedback replanning 이 committed planning 이나 값싼 1-step 제어보다
  일관되게 낫다는 것을 확립하지 못했다

Project decision
  사전 등록한 go/no-go 기준에서 이 증거는 PPO 학습에 드는 추가 복잡도와 계산을
  정당화하기에 불충분했다
```

게이트 임계값은 **이론적 보편 기준이 아니라 scope-control 장치**다. 결과를 본 뒤
바꾸지 않았다. 자세한 내용은 `paper/draft.md` §12.

## 저장소 구조

```text
src/rl_newton/
  optimizers/     Newton-CG 루프, 행동 공간, 컨트롤러 사다리
  curvature/      HVP 그래프, damped Hessian 연산자
  solvers/        truncated CG
  tasks/          quadratic, Rosenbrock, micro-neural MLP
  benchmark/      정체성 3계층, 지표, paired 통계, eligibility audit
  utils/          파라미터 평탄화, 난수 스트림 분리

scripts/          실행과 진단 (아래)
docs/             프로토콜, 자동 생성 결과표, 재현 명령
paper/            claim ledger, evidence map, outline, draft, figures
tests/            455개
```

## 빠른 시작

```bash
python -m pip install -e .
python -m pytest tests/ -q
python -m ruff check .
```

held-out 결과를 다시 집계한다 (재실행 없음).

```bash
python scripts/select_configuration.py \
    results/raw/headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl
```

표와 그림을 다시 만든다.

```bash
python scripts/make_report.py   --out docs/results_stage2.md
python scripts/make_manifest.py --out paper/evidence_map.md
python scripts/make_figures.py  --out-dir paper/figures
```

전체 실험을 처음부터 재현하는 명령은 `docs/reproduce.md` 에 있다.

## 설계에서 중요한 결정들

프로토콜 결정 `D1~D32` 는 `docs/experiment_protocol.md` 에 전부 기록돼 있다. 실험
설계에 영향이 컸던 것들이다.

### 실험은 전부 CPU 에서 돌았다

> All Stage 2 optimization and planning experiments were executed on CPU. The
> GPU-derived cost-model files in the repository originate from an earlier Stage 1
> calibration and were not used in Stage 2; Stage 2 GE accounting used HVP-equivalent
> oracle counts.

`configs/cost_model.*.yaml` 에 `device: cuda:0` 과 GPU 모델명이 있다. **Stage 2 결과와
무관하다.** `scripts/run_headroom.py` 는 cost model 을 로드하지 않으므로
(`cost_model=None`) GE 가 HVP 등가 횟수로 계산됐고, 모든 실행이 `device=cpu` 로
기록돼 있다. GPU 계수는 향후 신경망 task 를 위해 보존한다.

탐색 비용을 읽을 때도 구분이 필요하다.

> Search GE measures simulated oracle work rather than wall-clock GPU compute.

`1,294배` 는 planner 가 수행한 **oracle 호출량**의 비율이며 GPU wall-clock 비율이 아니다.

### 비용은 GE 로 잰다

```text
1 GE = gradient batch 1회 forward + backward
```

wall-clock 을 주 지표로 쓰지 않는다. 이 규모에서는 GPU 시간이 FLOP 이 아니라 커널
런치와 인터프리터 오버헤드에 지배된다. **모든 비교는 동일 GE 예산에서 한다.**
Newton-CG 는 step 비용이 행동에 따라 6배 이상 달라지므로 step 수를 맞추면 비용이
다른 것들을 비교하게 된다.

한계도 명시한다. GE 는 **regime 내부에서만** compute-matched 다. batch 크기가 다르면
같은 GE 가 다른 FLOP 을 뜻한다 (D30).

### seed 는 난수 시드가 아니라 실험 조건의 이름이다

`seed = s` 일 때 모든 컨트롤러가 동일한 인스턴스, 초기점, minibatch 순서를 본다.
task 생성용 난수 스트림을 optimizer 실행 스트림과 완전히 분리했다.

역할도 분리했다.

```text
CALIBRATION_SEEDS   0, 1          benchmark spec 선정에만
SELECTION_SEEDS     2, 3, 4       설정 선택에만
HELD_OUT_SEEDS      100 ~ 109     최종 효과 추정에만
```

### 정체성을 3계층으로 나눈다

집계 코드나 문서를 고쳐도 optimizer 가 재실행되지 않아야 한다.

```text
run_semantics_id   이 컨트롤러가 실제 쓰는 optimizer 설정만
sweep_id           이번 실행이 요청한 run 집합
aggregation_id     집계 정책
```

`git_commit` 과 `code_dirty` 는 어떤 식별자에도 넣지 않는다.

이 원칙은 **payload 에 키를 추가할 때마다 확인해야 한다.** 실제로 관계없는 task 족의
target 을 추가한 것만으로 Track T 240 run 이 무효화된 적이 있다 (D32).

### benchmark 적격성은 도달 가능한 상한으로 판정한다

수치 하한 기준 ceiling 은 **전역최소점이 도달 가능하다고 가정한다.** Rosenbrock
(`d=5`, 표준 시작점)에서 그 가정이 깨졌다. 표준 시작점 basin 에 strict 국소최소점이
있어 모든 baseline 이 정확히 같은 값에 도달했는데, 수치 하한 기준으로는 여유가
`29.62 nat` 로 보였다. 실제 여유는 `0` 이었다.

```text
L_ref        = min over reference runs started from the task's own initial point
J_achievable = log L_0 − log max(L_ref, L_floor)
```

참조 solver panel 을 쓴다. 단일 solver 는 금지한다. `κ=10⁶` 에서 L-BFGS 가 수렴하지
못해 상한을 `28.3 nat` 로 과소평가할 뻔했다. 다른 초기화의 결과는 **진단이며 상한을
올리지 않는다.**

seed 복제도 자동 검출한다. `RosenbrockSpec(dimension=5)` 가 시작점을 무작위화하지
않아 세 seed 가 같은 인스턴스였던 것을 이 검사로 발견했다.

## 진단 도구

```text
scripts/select_configuration.py    D21 설정 선택 규칙 적용
scripts/analyze_regimes.py         regime 별 분해. 절대값과 paired delta 함께
scripts/calibrate_challenge.py     baseline-only eligibility audit
scripts/bridge_validate.py         정체성 변경이 궤적을 바꿨는지 bitwise 검증
scripts/reaggregate.py             재실행 없이 새 집계 정책 적용
scripts/diagnose_rosen.py          컨트롤러가 같은 값에 멈춘 원인 진단
scripts/probe_rosen_basin.py       임계점이 국소최소점인지 판정
scripts/probe_instance_variation.py  seed 복제와 SPD 여부 확인
scripts/probe_micro_neural.py      micro-neural 사전 점검
scripts/diagnose_excluded_pairs.py 조용한 dropna 방지
```

## 알려진 한계

```text
quadratic 4 spec (d=100 고정), micro-neural 1 모델
예산 150 GE 단일 지점. 예산 축을 스캔하지 않았다
micro-neural 은 regime 당 n=3. CI 와 p-value 를 인용하지 않는다
CPU 단일 스레드. GPU 결과가 아니다
planner 는 oracle 이며 배포 가능한 방법이 아니다
  decision-search 가 배포 예산의 1,294배
게이트 C1 (fresh) 은 seed 1개 진단 baseline 이므로 판정에 쓰지 않는다
```

프로토콜 이탈 `E1~E11` 은 `paper/claim_ledger.md` 에 전부 기록돼 있다.

## 상태와 태그의 의미

```text
실험    Stage 2 종료
원고    초안 (paper/draft.md). 주장 강도 검토 중
테스트  455개 통과
```

`protocol-freeze-stage2-v1` 태그는 **프로토콜과 설정이 고정된 시점**을 가리킨다.

```text
태그가 보장하는 것
  게이트 정의와 임계값, 설정 선택 규칙, challenge spec 과 seed 역할이 그 시점에
  확정되어 있고 이후 결과를 보고 바꾸지 않았다

태그 이후 커밋에 포함된 것
  원고 작성, 그림 생성, README 재작성
  보고 도구의 median 규약 통일 (E10)
  run_semantics_id 에서 무관한 설정 제거 (E11, D32)

태그 이후에 하지 않은 것
  게이트 임계값 변경
  설정 재선택
  새 실험 조건 추가
```

`git diff protocol-freeze-stage2-v1..HEAD` 로 확인할 수 있다. 실험 조건이 아닌 변경만
들어 있다.

## 주장 검증

원고 문장과 claim ledger 를 기계적으로 대조한다. 사람이 읽어 확인하면 누락이 생긴다.

```bash
python scripts/check_claims.py
```

```text
SUPPORTED / LIMITED claim 이 draft 에 최소 1회 인용됐는가
NOT SUPPORTED claim 이 주장으로 인용되지 않았는가   (AVOID: 표시는 허용)
금지 표현이 draft 에 없는가
숫자를 담은 claim 에 evidence source 가 있는가
draft 의 claim ID 가 ledger 에 존재하는가
draft 의 [@key] 가 references.bib 에 있고 서지정보가 TODO 가 아닌가
```

원고에는 `<!-- CLAIM: C03 -->` 형태의 주석을 붙인다. HTML 주석이므로 PDF 에는 나오지
않는다.

## 인용

서지정보 확인과 "그 논문이 실제로 우리 주장을 지지하는가" 는 다른 작업이다. 앞의 것은
`scripts/check_claims.py` 가 기계적으로 보고, 뒤의 것은 사람이 봐야 한다.

```text
paper/references.bib   서지정보. note 필드에 VERIFIED / TODO 를 남긴다
paper/CITATIONS.md     인용별 내용 일치 체크리스트와 오인용 위험
```

`paper/CITATIONS.md` 의 `RISK` 항목은 특히 주의한다. 예를 들어 확장 Rosenbrock 인용은
**본문에 어느 변종을 썼는지 명시해야** 성립한다.

## 제출용 LaTeX

`paper/draft.md` 가 내용의 source of truth 이고 LaTeX 는 표현 계층이다.

```text
paper/main.tex        문서 골격
paper/sections/*.tex  본문 16개 파일
paper/tables/*.tex    scripts/make_tables.py 가 raw 에서 생성한다
paper/figures/*.png   scripts/make_figures.py 가 raw 에서 생성한다
```

**LaTeX 본문에 표 숫자를 쓰지 않는다.** 표는 `docs/results_stage2.md` 와 같은 로더·같은
median 규약을 쓰는 스크립트가 만든다.

```bash
python scripts/make_tables.py --out-dir paper/tables
python scripts/check_latex.py
```

`check_latex.py` 는 TeX 없이 `\input` 대상, 그림 파일, 인용 키, `\ref`/`\label` 만
검사한다. **조판 오류는 잡지 못하므로** 제출 전에 TeX 환경에서 한 번 빌드해야 한다.

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
