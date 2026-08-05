r"""LaTeX 원고의 구조를 LaTeX 없이 검사한다.

이 환경에 TeX 배포가 없어 `pdflatex` 로 빌드할 수 없다. 컴파일 대신 **깨지기 쉬운
참조만** 기계적으로 확인한다. 실제 조판 오류는 잡지 못하므로, 제출 전에 TeX 가 있는
환경에서 한 번 빌드해야 한다.

```text
1  \input{...} 대상 파일이 존재하는가
2  \includegraphics{...} 그림 파일이 존재하는가
3  \cite / \citep / \citet 의 키가 references.bib 에 있는가
4  \ref / \eqref 가 가리키는 \label 이 존재하는가
5  정의했지만 참조하지 않은 \label 이 있는가 (경고)
6  생성 파일(tables/*.tex)을 사람이 고친 흔적이 있는가
7  금지 표현이 LaTeX 본문에 없는가 (check_claims 와 같은 목록)
```

`[7]` 은 `check_claims.py` 의 목록을 그대로 import 한다. 목록이 갈리면 markdown 은
통과하고 LaTeX 만 과대주장하는 상태가 생긴다.

LaTeX 인용부호 ``...'' 안의 문구는 검사에서 제외한다. `§14` 의 "우리가 주장하지 않는
것" 표는 금지 문구를 **인용해 부인하는** 자리이므로 그대로 두어야 한다.

사용법:
    python scripts/check_latex.py
    python scripts/check_latex.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from check_claims import ALLOWED_CONTEXT, FORBIDDEN, LEGITIMATE_COMPOUNDS

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:eqref|[a-zA-Z]*ref)\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)
GENERATED_MARK = "scripts/make_tables.py 가 생성한다"
# LaTeX 인용부호. 안쪽 문구는 부인하려고 옮겨 적은 것이므로 금지어 검사에서 뺀다.
LATEX_QUOTE_RE = re.compile(r"``.*?''", re.DOTALL)


def strip_comments(text: str) -> str:
    """주석을 지운다. `\\%` 는 남긴다."""
    out = []
    for line in text.splitlines():
        idx = 0
        while True:
            idx = line.find("%", idx)
            if idx < 0:
                out.append(line)
                break
            if idx > 0 and line[idx - 1] == "\\":
                idx += 1
                continue
            out.append(line[:idx])
            break
    return "\n".join(out)


def scan_forbidden(body: str) -> list[tuple[int, str, str]]:
    """금지 표현을 찾는다. `check_claims.scan_forbidden` 과 같은 규칙이다."""
    patterns = [(p, re.compile(rf"\b{p}\b", re.IGNORECASE)) for p in FORBIDDEN]
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(body.splitlines(), start=1):
        low = LATEX_QUOTE_RE.sub(" ", line).lower()
        if any(ctx.lower() in low for ctx in ALLOWED_CONTEXT):
            continue
        probe = low
        for compound in LEGITIMATE_COMPOUNDS:
            probe = probe.replace(compound, " ")
        for name, rx in patterns:
            if rx.search(probe):
                hits.append((i, name, line.strip()[:110]))
    return hits


def collect(root: Path, main: Path) -> tuple[dict[Path, str], list[str]]:
    """`main.tex` 에서 시작해 `\\input` 을 따라가며 본문을 모은다."""
    errors: list[str] = []
    bodies: dict[Path, str] = {}
    queue = [main]
    seen: set[Path] = set()
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            errors.append(f"\\input 대상이 없다: {path}")
            continue
        body = strip_comments(path.read_text(encoding="utf-8"))
        bodies[path] = body
        for target in INPUT_RE.findall(body):
            name = target if target.endswith(".tex") else f"{target}.tex"
            queue.append(root / name)
    return bodies, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("paper"))
    parser.add_argument("--main", type=Path, default=None)
    parser.add_argument("--bib", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="경고도 실패로 취급한다")
    args = parser.parse_args()

    root: Path = args.root
    main_tex: Path = args.main or root / "main.tex"
    bib_path: Path = args.bib or root / "references.bib"
    fig_dir: Path = args.figures or root / "figures"

    if not main_tex.exists():
        print(f"없음: {main_tex}")
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    bodies, input_errors = collect(root, main_tex)
    errors += input_errors

    print("=" * 88)
    print(f"LaTeX 구조 검사   main {main_tex}")
    print(f"  본문 파일 {len(bodies)}개")
    print("=" * 88)

    print()
    print("[1] \\input 대상 존재")
    for path in sorted(bodies):
        print(f"  {path.relative_to(root) if root in path.parents else path}")

    print()
    print("[2] 그림 파일 존재")
    figures = {f for body in bodies.values() for f in GRAPHICS_RE.findall(body)}
    for name in sorted(figures):
        target = fig_dir / name
        ok = target.exists()
        print(f"  {'있음' if ok else '**없음**'}  {target}")
        if not ok:
            errors.append(f"그림 파일이 없다: {target}")
    if not figures:
        print("  없음")

    print()
    print("[3] 인용 키")
    if not bib_path.exists():
        errors.append(f"bib 가 없다: {bib_path}")
        bib_keys: set[str] = set()
    else:
        bib_keys = set(BIB_ENTRY_RE.findall(bib_path.read_text(encoding="utf-8")))
    used: set[str] = set()
    for body in bodies.values():
        for group in CITE_RE.findall(body):
            used |= {k.strip() for k in group.split(",") if k.strip()}
    print(f"  bib 항목 {len(bib_keys)}개, 인용 키 {len(used)}종")
    for key in sorted(used - bib_keys):
        print(f"  **{key}** 가 {bib_path.name} 에 없다")
        errors.append(f"인용 키 {key} 가 bib 에 없다")
    unused = sorted(bib_keys - used)
    if unused:
        print(f"  LaTeX 미인용 {len(unused)}개: {', '.join(unused)}")
        warnings.append(f"bib 항목 {len(unused)}개가 LaTeX 본문에서 인용되지 않았다")

    print()
    print("[4] \\ref 대상 \\label 존재")
    labels: set[str] = set()
    refs: set[str] = set()
    for body in bodies.values():
        labels |= set(LABEL_RE.findall(body))
        refs |= set(REF_RE.findall(body))
    print(f"  label {len(labels)}개, ref {len(refs)}종")
    for name in sorted(refs - labels):
        print(f"  **{name}** 에 대응하는 \\label 이 없다")
        errors.append(f"\\ref{{{name}}} 의 label 이 없다")

    print()
    print("[5] 참조되지 않은 \\label (경고)")
    orphan = sorted(labels - refs)
    if orphan:
        print(f"  {len(orphan)}개: {', '.join(orphan)}")
        warnings.append(f"label {len(orphan)}개가 참조되지 않았다")
    else:
        print("  없음")

    print()
    print("[6] 생성 표를 손으로 고쳤는지")
    table_dir = root / "tables"
    generated = sorted(table_dir.glob("*.tex")) if table_dir.exists() else []
    for path in generated:
        text = path.read_text(encoding="utf-8")
        if GENERATED_MARK not in text:
            print(f"  **{path.name}** 에 생성 표시가 없다")
            errors.append(f"{path} 에 생성 표시가 없다. 손으로 만든 표인지 확인한다")
    print(f"  검사 완료 ({len(generated)}개)")

    print()
    print("[7] 금지 표현 검사 (check_claims 와 같은 목록)")
    total = 0
    for path in sorted(bodies):
        for line_no, word, text in scan_forbidden(bodies[path]):
            total += 1
            print(f"  {path.name}:{line_no}  '{word}'  {text}")
            errors.append(f"{path.name}:{line_no} 금지 표현 '{word}'")
    if total == 0:
        print("  없음")

    print()
    print("=" * 88)
    if errors:
        print(f"실패 {len(errors)}건")
        for e in errors:
            print(f"  {e}")
    if warnings:
        print(f"경고 {len(warnings)}건")
        for w in warnings:
            print(f"  {w}")
    if not errors and not warnings:
        print("통과: 모든 검사를 만족한다")
    elif not errors:
        print("통과 (경고 있음)")
    print()
    print("**주의.** 이 검사는 조판 오류를 잡지 못한다. 제출 전에 TeX 환경에서")
    print("  pdflatex main && bibtex main && pdflatex main && pdflatex main")
    print("을 한 번 실행해야 한다.")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
