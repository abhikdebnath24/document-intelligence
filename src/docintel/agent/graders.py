from __future__ import annotations

from collections.abc import Sequence

from docintel.core.interfaces import BaseGrader
from docintel.core.registry import Registry
from docintel.core.types import GradeResult, RetrievedChunk
from docintel.llm.factory import StructuredCaller
from docintel.llm.prompts import PromptBank
from docintel.llm.schemas import GradeBatchOut, GradeItemOut

GRADERS: Registry[BaseGrader] = Registry("grader")


def _clip(text: str, n: int = 800) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


@GRADERS.register("score_threshold")
class ScoreThresholdGrader(BaseGrader):
    def __init__(self, threshold: float = 0.0, **_: object) -> None:
        self.threshold = float(threshold)

    def grade(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[GradeResult]:
        _ = query
        return [
            GradeResult(
                chunk_id=row.chunk.chunk_id,
                relevant=row.score >= self.threshold,
                reason=f"score={row.score:.3f}",
            )
            for row in chunks
        ]


class LlmBatchGrader(BaseGrader):
    def __init__(self, caller: StructuredCaller, prompts: PromptBank) -> None:
        self.caller = caller
        self.prompts = prompts

    def grade(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[GradeResult]:
        if not chunks:
            return []
        blocks = "\n".join(
            f'<evidence id="{row.chunk.chunk_id}">\n{_clip(row.chunk.text)}\n</evidence>'
            for row in chunks
        )
        prompt = f"Question: {query}\n\n{blocks}"
        out = self.caller.structured(
            "grader", GradeBatchOut, prompt, system=self.prompts.get("grade_batch").text
        )
        by_id = {item.chunk_id: item for item in out.grades}
        return [_as_grade(by_id.get(row.chunk.chunk_id), row.chunk.chunk_id) for row in chunks]


class LlmPerChunkGrader(BaseGrader):
    def __init__(self, caller: StructuredCaller, prompts: PromptBank) -> None:
        self.caller = caller
        self.prompts = prompts

    def grade(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[GradeResult]:
        system = self.prompts.get("grade_one").text
        out: list[GradeResult] = []
        for row in chunks:
            prompt = (
                f'Question: {query}\n<evidence id="{row.chunk.chunk_id}">\n'
                f"{_clip(row.chunk.text)}\n</evidence>"
            )
            item = self.caller.structured("grader", GradeItemOut, prompt, system=system)
            out.append(
                GradeResult(chunk_id=row.chunk.chunk_id, relevant=item.relevant, reason=item.reason)
            )
        return out


def build_grader(
    name: str, caller: StructuredCaller, prompts: PromptBank, **params: object
) -> BaseGrader:
    if name == "llm_batch":
        return LlmBatchGrader(caller, prompts)
    if name == "llm_per_chunk":
        return LlmPerChunkGrader(caller, prompts)
    return GRADERS.create(name, **params)


def _as_grade(item: GradeItemOut | None, chunk_id: str) -> GradeResult:
    if item is None:
        return GradeResult(chunk_id=chunk_id, relevant=False, reason="missing grade")
    return GradeResult(chunk_id=chunk_id, relevant=item.relevant, reason=item.reason)
