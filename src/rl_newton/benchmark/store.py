"""재개 가능한 결과 저장소.

왜 필요한가
-----------
Stage 2 실험은 컨트롤러 x 행동공간 x horizon x task x seed x target 조합이라
수백~수천 run 이 된다. 프로세스가 중간에 끊기면(셸 중단, timeout, OOM) 이미
계산한 결과까지 잃는 것은 받아들일 수 없다.

그래서 run 단위로 즉시 append-only JSONL 에 기록하고, 재실행 시 이미 완료된
``(controller, task_instance, seed, target)`` 조합을 건너뛴다. 단순한 백그라운드
실행보다 이것이 우선이다.

상태를 구분한다
---------------
```text
completed    정상 종료. 재실행 시 건너뛴다.
failed       예외 발생. 원인을 기록하고 재실행 시 다시 시도한다.
interrupted  기록 전에 프로세스가 끊긴 경우. 파일에 남지 않으므로
             자동으로 재시도 대상이 된다.
```

미완료를 조용히 성공으로 취급하지 않는다. 프로토콜 D6의 절단 규칙과 같은
원칙이다 (실패한 run 도 삭제하지 않고 원인을 기록한다).

집계와 raw 를 분리한다
----------------------
이 저장소는 raw run 기록만 담는다. 집계(``GroupSummary``, 게이트 판정)는
``metrics`` / ``oracle`` 이 raw 를 읽어 계산한다. 그래서 집계 로직을 바꿔도
실험을 다시 돌릴 필요가 없다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rl_newton.benchmark.metrics import RunSummary
from rl_newton.utils.logging import sanitize_for_json

__all__ = ["RunKey", "RunRecord", "ResultStore", "RunStatus"]

RunStatus = Literal["completed", "failed"]


@dataclass(frozen=True, slots=True)
class RunKey:
    """run 하나를 유일하게 식별한다. 재개 판단의 기준이다."""

    controller: str
    task_instance_id: str
    seed: int
    target: str

    def as_str(self) -> str:
        return f"{self.controller}|{self.task_instance_id}|{self.seed}|{self.target}"

    @classmethod
    def from_summary(cls, summary: RunSummary) -> RunKey:
        return cls(
            controller=summary.controller,
            task_instance_id=summary.task_instance_id,
            seed=summary.seed,
            target=summary.target,
        )


@dataclass(slots=True)
class RunRecord:
    """저장되는 한 줄.

    Attributes:
        key: 식별자.
        status: ``completed`` 또는 ``failed``.
        summary: 정상 종료 시의 ``RunSummary``. 실패면 ``None``.
        wall_clock_sec: 실제 소요 시간. GE 와 별개로 기록한다 (프로토콜 D1).
        action_counts: 선택한 action 의 빈도. 정책 분석용 (README §8).
        chosen_depths: planner 가 채택한 계획 길이의 빈도. 게이트 C 해석용.
        error: 실패 원인.
        recorded_at: 기록 시각.
        git_commit: 코드 버전.
        config_hash: 설정 해시.
    """

    key: RunKey
    status: RunStatus
    summary: RunSummary | None = None
    wall_clock_sec: float = float("nan")
    action_counts: dict[str, int] | None = None
    chosen_depths: dict[str, int] | None = None
    error: str | None = None
    recorded_at: str = ""
    git_commit: str = ""
    config_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": asdict(self.key),
            "key_str": self.key.as_str(),
            "status": self.status,
            "wall_clock_sec": self.wall_clock_sec,
            "action_counts": self.action_counts,
            "chosen_depths": self.chosen_depths,
            "error": self.error,
            "recorded_at": self.recorded_at or datetime.now(UTC).isoformat(),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "summary": asdict(self.summary) if self.summary is not None else None,
        }
        return sanitize_for_json(payload)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunRecord:
        summary_fields = {f.name for f in fields(RunSummary)}
        raw = payload.get("summary")
        summary = None
        if raw is not None:
            # JSON 에서 비유한값은 None 으로 저장되므로 float 필드를 복원한다.
            restored = {k: v for k, v in raw.items() if k in summary_fields}
            for name in summary_fields:
                if name not in restored:
                    restored[name] = None
            summary = _restore_summary(restored)
        return cls(
            key=RunKey(**payload["key"]),
            status=payload["status"],
            summary=summary,
            wall_clock_sec=_as_float(payload.get("wall_clock_sec")),
            action_counts=payload.get("action_counts"),
            chosen_depths=payload.get("chosen_depths"),
            error=payload.get("error"),
            recorded_at=payload.get("recorded_at", ""),
            git_commit=payload.get("git_commit", ""),
            config_hash=payload.get("config_hash", ""),
        )


def _as_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _restore_summary(raw: dict[str, Any]) -> RunSummary:
    """JSON 에서 ``RunSummary`` 를 복원한다.

    ``None`` 처리를 필드 타입에 따라 나눈다. 이것이 **절단 규칙의 핵심**이다.

    ```text
    float          NaN 이 "측정 불가" 를 뜻한다. JSON 의 null -> NaN
    float | None   None 이 "미도달" 을 뜻한다.   JSON 의 null -> None 유지
    ```

    ``cost_to_target_ge`` 를 NaN 으로 바꾸면 "목표에 도달하지 못했다"는 정보가
    "값을 모른다"로 바뀐다. 프로토콜 D6은 미도달 run 을 큰 값으로 대입하지도,
    삭제하지도 않는다고 규정하므로 이 구분을 보존해야 한다.
    """
    optional_fields = {
        name
        for name, annotation in (
            (f.name, str(f.type)) for f in fields(RunSummary)
        )
        if "None" in annotation
    }
    kwargs: dict[str, Any] = {}
    for field in fields(RunSummary):
        value = raw.get(field.name)
        if value is None and str(field.type) == "float":
            value = float("nan")
        elif value is None and field.name in optional_fields:
            value = None
        kwargs[field.name] = value
    # 필수 필드 타입 보정
    kwargs["reached"] = bool(kwargs.get("reached"))
    kwargs["seed"] = int(kwargs.get("seed") or 0)
    kwargs["n_steps"] = int(kwargs.get("n_steps") or 0)
    kwargs["total_hvp"] = int(kwargs.get("total_hvp") or 0)
    return RunSummary(**kwargs)


class ResultStore:
    """append-only JSONL 기반 재개 가능한 저장소.

    Args:
        path: JSONL 경로. 없으면 만든다.
        git_commit: 기록에 붙일 코드 버전.
        config_hash: 기록에 붙일 설정 해시.

    Example:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as d:
        ...     store = ResultStore(pathlib.Path(d) / "runs.jsonl")
        ...     key = RunKey("fixed", "quad_seed0", 0, "relative_loss<=1e-06")
        ...     store.is_completed(key)
        False
    """

    def __init__(
        self,
        path: str | Path,
        *,
        git_commit: str = "",
        config_hash: str = "",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.git_commit = git_commit
        self.config_hash = config_hash
        self._records: dict[str, RunRecord] = {}
        self._load()

    # --- 로드 -------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                record = RunRecord.from_json(payload)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                # 프로세스가 쓰는 중 끊기면 마지막 줄이 깨질 수 있다.
                # 그 줄만 버리고 나머지는 살린다.
                print(
                    f"경고: {self.path}:{line_number} 를 읽을 수 없어 건너뛴다 ({exc})",
                    flush=True,
                )
                continue
            # 같은 키가 여러 번 있으면 마지막 것이 유효하다 (재시도 결과).
            self._records[record.key.as_str()] = record

    # --- 조회 -------------------------------------------------------------

    def is_completed(self, key: RunKey) -> bool:
        """정상 완료된 run 인지. ``True`` 면 재실행에서 건너뛴다."""
        record = self._records.get(key.as_str())
        return record is not None and record.status == "completed"

    def get(self, key: RunKey) -> RunRecord | None:
        return self._records.get(key.as_str())

    def summaries(self, *, controller: str | None = None) -> list[RunSummary]:
        """완료된 run 의 ``RunSummary`` 목록. 집계 입력이다."""
        out: list[RunSummary] = []
        for record in self._records.values():
            if record.status != "completed" or record.summary is None:
                continue
            if controller is not None and record.key.controller != controller:
                continue
            out.append(record.summary)
        return out

    def failures(self) -> list[RunRecord]:
        """실패한 run 들. 삭제하지 않고 원인을 보존한다 (README §15)."""
        return [r for r in self._records.values() if r.status == "failed"]

    def controllers(self) -> list[str]:
        return sorted({r.key.controller for r in self._records.values()})

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._records.values())

    # --- 기록 -------------------------------------------------------------

    def put(self, record: RunRecord) -> None:
        """한 줄 추가하고 즉시 flush 한다.

        프로세스가 다음 run 에서 끊겨도 이 결과는 보존된다.
        """
        if not record.git_commit:
            record.git_commit = self.git_commit
        if not record.config_hash:
            record.config_hash = self.config_hash
        if not record.recorded_at:
            record.recorded_at = datetime.now(UTC).isoformat()

        line = json.dumps(record.to_json(), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
        self._records[record.key.as_str()] = record

    def record_success(
        self,
        summary: RunSummary,
        *,
        wall_clock_sec: float,
        action_counts: dict[str, int] | None = None,
        chosen_depths: dict[str, int] | None = None,
    ) -> None:
        self.put(
            RunRecord(
                key=RunKey.from_summary(summary),
                status="completed",
                summary=summary,
                wall_clock_sec=wall_clock_sec,
                action_counts=action_counts,
                chosen_depths=chosen_depths,
            )
        )

    def record_failure(
        self, key: RunKey, error: str, *, wall_clock_sec: float = float("nan")
    ) -> None:
        self.put(
            RunRecord(
                key=key,
                status="failed",
                error=error,
                wall_clock_sec=wall_clock_sec,
            )
        )

    def describe(self) -> str:
        completed = sum(1 for r in self._records.values() if r.status == "completed")
        failed = len(self._records) - completed
        return (
            f"ResultStore({self.path.name}): 완료 {completed}, 실패 {failed}, "
            f"컨트롤러 {len(self.controllers())}종"
        )
