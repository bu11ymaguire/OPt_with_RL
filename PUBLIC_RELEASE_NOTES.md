# Public release notes

공개 전에 수행한 정리 내역이다. **실험 프로토콜 이탈이 아니다.** 프로토콜 이탈은
`paper/claim_ledger.md` 의 `E1~E12` 에 별도로 기록돼 있다.

## `configs/cost_model.*.yaml` 의 GPU 정보에 대해

두 cost-model YAML 에 `device: cuda:0` 과 GPU 모델명이 있다. **Stage 2 결과와 무관하다.**

```text
Stage 1 에서 GE 환산 계수를 실측한 산출물이다
scripts/run_headroom.py 는 cost model 을 로드하지 않는다 (cost_model=None)
따라서 Stage 2 의 GE 는 HVP 등가 횟수로 계산됐다
모든 Stage 2 실행이 device=cpu 로 기록돼 있다
```

Stage 2 의 모든 측정은 CPU 단일 스레드에서 수행됐다. GPU 계수는 향후 신경망 task 에서
쓰기 위해 보존한다.

## Hostname sanitization

> Hostnames in committed public artifacts were replaced with stable pseudonymous labels
> before public release. This sanitization did not modify experimental configurations,
> results, checksums of raw numerical records, or scientific conclusions.

### 무엇을 바꿨는가

```text
바꿈     git 추적 대상 11개 파일의 장치 이름 -> `host-a`
           results/summaries/*.json   (9개)  키 `hostname`
           configs/cost_model.*.yaml  (2개)  키 `host`
안 바꿈  results/raw/*.jsonl  수치 기록
```

`paper/evidence_map.md` 의 SHA-256 은 **raw 파일**을 해시한다. raw 를 건드리지 않았으
므로 checksum 이 그대로 유효하다. summary JSON 자체의 checksum 은 바뀌지만 evidence
map 이 그것을 가리키지 않는다.

`scripts/sanitize_public_artifacts.py` 가 이 작업을 수행하고 재검사한다.

```bash
python scripts/sanitize_public_artifacts.py --check
```

### 앞으로의 기록 방식

`environment_fingerprint` 와 `measure_cost_model` 이 장치 실제 이름을 더 이상 읽지
않는다. `EXPERIMENT_HOST_ID` 환경변수의 별칭을 쓰고, 없으면
`host-unspecified` 를 기록한다.

```bash
EXPERIMENT_HOST_ID=host-a python scripts/run_headroom.py --mode challenge-heldout ...
```

hostname 해시는 쓰지 않는다. salt 가 없으면 사전 대입으로 복원되고, salt 를 관리하는
복잡도가 이득보다 크다.

### 재현에 실제로 필요한 정보

장치의 개인 이름은 여기 없다. 아래는 모두 보존돼 있다.

```text
OS 와 platform            execution_provenance.platform
Python / PyTorch 버전     docs/experiment_protocol.md, cost-model YAML
CPU 모델과 스레드 수       environment_fingerprint (torch_num_threads, cpu_count)
dtype                     프로토콜과 task 정의
코드 commit               execution_provenance.git_commit, raw 레코드의 per-run 값
configuration hash        run_semantics_id / sweep_id / aggregation_id
raw checksum              paper/evidence_map.md
```

## 이미 공개된 이력에 남은 노출

**정리 이전 커밋에는 원래 hostname 이 남아 있다.** 원격 `origin/main` 이 다음 커밋을
이미 포함한다.

```text
9679fe3  D22/D23
0e5a182  D26
a140ed8  D27
```

즉 이 정리는 **앞으로의 산출물과 현재 트리**를 깨끗하게 만들지만, 이미 push 된 Git
이력에서 값을 제거하지는 않는다. 제거하려면 이력 재작성과 force push 가 필요하고
그것은 파괴적 작업이다. 이 저장소는 그 작업을 수행하지 않았다.

## 태그의 의미

**이 절의 이전 판은 사실과 달랐다.** "공개 저장소는 비어 있다", "릴리스 태그는 아직
없다", "protocol-freeze 태그는 push 하지 않는다" 세 가지가 모두 틀렸다. 원격을 조회하지
않고 계획을 서술한 것이 원인이다. 아래는 `git ls-remote` 로 확인한 값이다.

```text
private 저장소  https://github.com/bu11ymaguire/OPt_with_RL   (private 유지)
  protocol-freeze-stage2-v1   16681b3 -> 40c84c3   **origin 에 이미 있다** (아래)
  public-release-stage2-v1    4b3e644 -> beee6f4   origin 에 있다
  stage2-private-archive-v1   만들지 않았다

공개 저장소     https://github.com/bu11ymaguire/When_Does_Feedback_Help   (public)
  main                        내용 push 완료 (2026-08-06)
  arxiv-submission-v1         **폐기 예정.** 아래 참조
  stage2-report-v1            현재 원고 §15 가 인용하는 태그
```

### `arxiv-submission-v1` 을 폐기하는 이유

이 태그는 arXiv 제출을 전제로 만들어졌다. **제출은 일어나지 않았다.** 2026-08-07
endorsement 요청이 거절됐고 그 이후 제출을 시도하지 않았다.

따라서 이 이름은 **일어나지 않은 사건을 가리킨다.** 태그 이름만 보고 "arXiv 에 올라간
버전" 으로 읽는 사람이 생기며, 그것이 이 저장소가 다른 곳에서 지키려는 기준과 정면으로
어긋난다. 존재하지 않는 artifact 를 실재하는 것처럼 제시하는 사고를 막기 위해 `\PLACEHOLDER`
와 `check_latex.py [9]` 를 도입했는데, 태그 이름 자체가 같은 종류의 허위 진술이었다.

`stage2-report-v1` 로 교체한다. Stage 2 를 정리한 미심사 technical report 의 코드·데이터
동결점이라는 뜻이며, 그 이상을 주장하지 않는다.

### `protocol-freeze-stage2-v1`

`docs/experiment_protocol.md` 의 `C2 protocol freeze` 절은 동결 시점에 이 태그를
남기고 이후 변경을 금지하도록 규정했다. 태그는 **존재한다.**

```text
annotated tag 16681b3 -> commit 40c84c3
created 2026-08-04 21:53:22 +0900
```

그러나 시점과 대상이 규정과 다르다.

```text
17:48  e58e5cd  D24/D25  shrinking_Q4_narrow 동결   <- 규정된 태깅 시점
19:07  0e5a182  D26      held-out confirmatory n=40
21:47  40c84c3  D30/D31
21:53  tag               protocol-freeze-stage2-v1 -> 40c84c3
```

동결 커밋보다 4시간 5분 뒤, confirmatory 실행 이후에 만들어졌다. 따라서 이 태그로는
"held-out 실행 전에 config 가 고정됐다" 를 확인할 수 없다. 그 순서는 결정 기록
`D24/D25` 와 동결 커밋 `e58e5cd` 가 근거다. `D1~D32` 에 무엇이 언제 고정됐는지 순서대로
남아 있다. **태그를 소급해서 옮기지 않는다.**

**이 태그는 origin 에 이미 올라가 있다.** 이 문서의 이전 판은 "원격에 올리지 않는다" 고
적었는데 사실이 아니었다.

```bash
git ls-remote --tags origin
# 16681b30c8a34e5cd574e7b521f47feb00add2eb  refs/tags/protocol-freeze-stage2-v1
# 40c84c326f6aa00ec45c45184c712c40222be607  refs/tags/protocol-freeze-stage2-v1^{}
```

가리키는 커밋 `40c84c3` 이 **장치 이름 정리 이전**이므로 실질적 함의가 있다.

```text
지금        origin 이 private 이므로 노출되지 않는다
전환하면    이 저장소를 public 으로 바꾸면 40c84c3 의 hostname 이 드러난다
```

따라서 **이 저장소는 private 로 유지한다.** 공개는 `scripts/export_public_repo.py` 가
`git init` 으로 새 이력을 만드는 별도 저장소로만 한다. 그 저장소는 이 태그를 포함하지
않는다.

이 저장소를 굳이 public 으로 바꾸려면 이력 재작성(`filter-repo`)과 force push 가 필요하고,
그것은 파괴적 작업이며 이미 배포된 커밋 해시를 모두 무효화한다. **수행하지 않았다.**

### 공개 갱신 게이트

`scripts/check_latex.py --strict` 의 채움 표시 검사 `[8]` 은 0 건을 반환한다. 이름을
채우는 것과 그 artifact 가 실재하는 것은 다른 문제이고, `[8]` 은 앞의 것만 본다. 실재
확인은 `[9]` 가 한다.

```bash
python scripts/check_latex.py --strict --check-remote
```

**공개 저장소를 갱신할 때마다 이 명령이 통과해야 한다.** 원고가 인용하는 태그를 바꿨으면
공개 저장소에 그 태그를 만든 뒤에 돌린다. 순서를 뒤집으면 `[9]` 가 실패한다.

```text
원고 §15 가 인용하는 것    stage2-report-v1
공개 저장소에 있어야 하는 것  같은 이름의 태그
```

## 공개 전 점검 목록

```bash
python -m pytest tests/ -q
python -m ruff check .
python scripts/check_claims.py
python scripts/check_latex.py --strict --check-remote
python scripts/sanitize_public_artifacts.py --check
python scripts/make_report.py   --out docs/results_stage2.md
python scripts/make_manifest.py --out paper/evidence_map.md
python scripts/make_figures.py  --out-dir paper/figures
python scripts/make_tables.py   --out-dir paper/tables
git diff public-release-stage2-v1..HEAD
```

```text
사용자명과 로컬 절대경로 노출 없음
API key 없음
figure 경로 유효
raw SHA-256 재생성됨
README 재현 명령 유효
```

## 인용

`paper/references.bib` 에 26 항목이 있고 `paper/draft.md` 의 `[CITATION NEEDED]` 는
남아 있지 않다. 서지정보는 원문 또는 출판사 페이지에서 확인했고, **어디서 확인했는지는
`paper/CITATIONS.md §9`** 에 있다. `.bib` 에는 출판용 서지정보만 두고 내부 검증 메모를
넣지 않는다. `plainnat` 이 `note` 필드를 참고문헌 목록에 인쇄하기 때문이다.

세 항목에 **부분 미확인** 필드가 있다. 본문 주장에는 영향이 없고 표기 정밀도 문제다.

```text
bertsekas2017dp        절 번호 미확인 -> 특정 절을 지목하지 않는다
amos2023amortized      FnT 권/호/페이지 미확인 -> arXiv ID 로 인용한다
lakens2017equivalence  권/호/페이지 미확인 -> doi 로 인용한다
```

서지정보 확인과 **내용 일치**는 다른 작업이므로 후자는 `paper/CITATIONS.md` 에
인용별로 기록했다. `RISK` 로 표시한 항목이 네 개 있다.

```text
kok2009rosenbrock  본문에 어느 Rosenbrock 변종인지 명시해야 성립한다
bertsekas2017dp    절 번호를 지목하면 안 된다
schulman2017ppo    PPO 를 실행하지 않았다. 인용이 실행으로 읽히면 안 된다
schuirmann1987tost / lakens2017equivalence
                   등가성 검정을 수행하지 않았다. margin 이 없다는 근거로만 쓴다
```

## 남은 항목

이 절의 이전 판은 세 항목을 미완으로 적었는데 두 개는 이미 해결됐다.

```text
해결  MiKTeX 로 pdflatex + bibtex 4-pass 빌드가 된다
        paper/main.pdf 는 .gitignore 에 있다. 재생성 가능한 산출물이므로 추적하지 않는다
해결  LICENSE (MIT) 와 CITATION.cff 가 public/ 에 있다
        export 시 RENAMES 규칙으로 공개 저장소 루트에 놓인다

미해결  외부 연구자 검토를 받지 않았다
```

### 외부 검토에 대해

이것은 도구로 닫을 수 없는 유일한 항목이고, 실제로 시도했으며 실패했다. 기록해 둔다.

```text
2026-08-06  arXiv cs.LG endorsement 요청 (main.pdf 첨부)
2026-08-07  거절. 사유: 원고가 AI 생성물로 보이며 endorse 할 수 없다
```

거절 사유를 반박하지 않는다. AI 보조 범위는 원고 `§15` 와 공개 README 에 적혀 있고,
그 서술이 축소된 것도 아니다. 다만 **"AI 가 관여했다" 와 "주장이 증거를 넘었다" 는 다른
문제**이므로, 후자에 대해서는 기계적으로 검사 가능한 형태를 남겼다 (`check_claims.py`,
`claim_ledger.md`, `D1~D32`). 전자에 대해서는 방어하지 않고 공개한다.

따라서 이 저장소는 **심사를 통과한 결과물이 아니라 검사 가능한 산출물**로 공개한다.
원고 표지와 공개 README 상단에 그 지위를 명시했다.
