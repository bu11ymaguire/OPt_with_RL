# Public release notes

공개 전에 수행한 정리 내역이다. **실험 프로토콜 이탈이 아니다.** 프로토콜 이탈은
`paper/claim_ledger.md` 의 `E1~E11` 에 별도로 기록돼 있다.

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

```text
public-release-stage2-v1    개인정보 정리와 원고를 포함한 공개점
arxiv-submission-v1         arXiv 제출 시점의 코드와 원고 (제출 시 생성)
```

`docs/experiment_protocol.md` 의 `C2 protocol freeze` 절은 동결 시점에
`git tag protocol-freeze-stage2-v1` 을 남기도록 규정했다. **그 태그는 실제로 만들어지지
않았다.** 이전 판의 이 문서는 그것을 "로컬 기록으로 보존한다" 고 적었는데 사실이
아니었다. 로컬에도 원격에도 없고 reflog 에도 흔적이 없다.

프로토콜 동결점의 내용은 `docs/experiment_protocol.md` 의 결정 `D1~D32` 와 변경
이력에 전부 남아 있으므로, 태그 없이도 무엇이 언제 고정됐는지 확인할 수 있다.
**없는 태그를 소급 생성하지 않는다.** 그 시점에 실제로 동결했다는 이력을 사후에
만들어내는 셈이기 때문이다.

## 공개 전 점검 목록

```bash
python -m pytest tests/ -q
python -m ruff check .
python scripts/check_claims.py
python scripts/check_latex.py
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

```text
이 환경에 TeX 배포가 없어 PDF 를 빌드하지 않았다
  paper/main.tex 와 sections/ 는 작성됐고 scripts/check_latex.py 로 구조만 검사했다
  제출 전에 TeX 환경에서 pdflatex + bibtex 를 한 번 실행해야 한다
LICENSE 와 CITATION.cff 가 없다
외부 연구자 검토를 받지 않았다
```
