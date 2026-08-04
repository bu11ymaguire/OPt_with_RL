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
protocol-freeze-stage2-v1   실험 프로토콜과 설정이 고정된 시점
                            hostname 정리 이전이다
public-release-stage2-v1    개인정보 정리와 원고를 포함한 공개점
```

`protocol-freeze-stage2-v1` 은 로컬 기록으로 보존하고 공개 저장소에는 push 하지
않는다. 공개 이력에 정리 이전 값을 추가로 노출할 이유가 없다.

프로토콜 동결점의 내용은 `docs/experiment_protocol.md` 의 결정 `D1~D32` 와 변경
이력에 전부 남아 있으므로, 태그 없이도 무엇이 언제 고정됐는지 확인할 수 있다.

## 공개 전 점검 목록

```bash
python -m pytest tests/ -q
python -m ruff check .
python scripts/check_claims.py
python scripts/sanitize_public_artifacts.py --check
python scripts/make_report.py   --out docs/results_stage2.md
python scripts/make_manifest.py --out paper/evidence_map.md
python scripts/make_figures.py  --out-dir paper/figures
git diff protocol-freeze-stage2-v1..HEAD
```

```text
사용자명과 로컬 절대경로 노출 없음
API key 없음
figure 경로 유효
raw SHA-256 재생성됨
README 재현 명령 유효
```

## 남은 항목

```text
LaTeX 소스가 없어 PDF 를 빌드하지 않았다. 원고는 Markdown 이다 (paper/draft.md)
인용은 [CITATION NEEDED] 로 표시돼 있고 아직 채우지 않았다
```
