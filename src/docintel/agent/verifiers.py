from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from docintel.core.interfaces import BaseGroundednessVerifier
from docintel.core.registry import Registry
from docintel.core.types import Citation, GroundednessResult, RetrievedChunk
from docintel.evaluation.gold import SpanMatcher, normalize_text
from docintel.llm.factory import StructuredCaller
from docintel.llm.prompts import PromptBank
from docintel.llm.schemas import VerifyOut

VERIFIERS: Registry[BaseGroundednessVerifier] = Registry("verifier")


def validate_citations(
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
    *,
    threshold: int = 90,
) -> list[str]:
    """Return reasons for invalid cites. Empty means all cites are valid."""
    by_id = {row.chunk.chunk_id: row.chunk for row in chunks}
    matcher = SpanMatcher(threshold)
    errors: list[str] = []
    for cite in citations:
        chunk = by_id.get(cite.chunk_id)
        if chunk is None:
            errors.append(f"unknown chunk_id {cite.chunk_id}")
            continue
        if cite.doc_id and cite.doc_id != chunk.doc_id:
            errors.append(f"doc_id mismatch for {cite.chunk_id}")
            continue
        if not cite.quote.strip() or not matcher.fragment_in_text(cite.quote, chunk.text):
            errors.append(f"quote not in chunk {cite.chunk_id}")
    return errors


@VERIFIERS.register("lexical_overlap")
class LexicalOverlapVerifier(BaseGroundednessVerifier):
    def __init__(self, threshold: float = 0.8, **_: object) -> None:
        self.threshold = float(threshold)

    def verify(self, answer: str, chunks: Sequence[RetrievedChunk]) -> GroundednessResult:
        hay = normalize_text(" ".join(row.chunk.text for row in chunks))
        tokens = [t for t in normalize_text(answer).split() if len(t) > 2]
        if not tokens:
            return GroundednessResult(grounded=False, score=0.0, unsupported_claims=["empty"])
        hits = sum(1 for tok in tokens if tok in hay)
        score = hits / len(tokens)
        unsupported = [] if score >= self.threshold else ["lexical overlap below threshold"]
        return GroundednessResult(
            grounded=not unsupported, score=score, unsupported_claims=unsupported
        )


class LlmClaimsVerifier(BaseGroundednessVerifier):
    def __init__(
        self, caller: StructuredCaller, prompts: PromptBank, threshold: float = 0.8
    ) -> None:
        self.caller = caller
        self.prompts = prompts
        self.threshold = float(threshold)

    def verify(self, answer: str, chunks: Sequence[RetrievedChunk]) -> GroundednessResult:
        blocks = "\n".join(
            f'<evidence id="{row.chunk.chunk_id}">\n{row.chunk.text}\n</evidence>' for row in chunks
        )
        prompt = f"Answer:\n{answer}\n\n{blocks}"
        out = self.caller.structured(
            "verifier", VerifyOut, prompt, system=self.prompts.get("verify").text
        )
        grounded = bool(out.grounded) and out.score >= self.threshold and not out.unsupported_claims
        return GroundednessResult(
            grounded=grounded, score=out.score, unsupported_claims=list(out.unsupported_claims)
        )


class NliCrossEncoderVerifier(BaseGroundednessVerifier):
    def __init__(
        self,
        model_id: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.8,
        **_: object,
    ) -> None:
        self.model_id = model_id
        self.threshold = float(threshold)
        self._model: Any = None

    def verify(self, answer: str, chunks: Sequence[RetrievedChunk]) -> GroundednessResult:
        premises = [row.chunk.text for row in chunks]
        if not premises or not answer.strip():
            return GroundednessResult(grounded=False, score=0.0, unsupported_claims=["empty"])
        model = self._load()
        pairs = [[text, answer] for text in premises]
        scores = model.predict(pairs, show_progress_bar=False)
        # sentence-transformers NLI models return entailment/neutral/contradiction scores
        # or a single score; take the max entailment-like value.
        best = _best_entailment(scores)
        unsupported = [] if best >= self.threshold else ["nli entailment below threshold"]
        return GroundednessResult(
            grounded=not unsupported, score=float(best), unsupported_claims=unsupported
        )

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from docintel.settings import hf_token

            kwargs: dict[str, object] = {}
            token = hf_token()
            if token:
                kwargs["token"] = token
            self._model = CrossEncoder(self.model_id, **kwargs)
        return self._model


def build_verifier(
    name: str, caller: StructuredCaller, prompts: PromptBank, **params: object
) -> BaseGroundednessVerifier:
    if name == "llm_claims":
        raw = params.get("threshold", 0.8)
        threshold = float(raw) if isinstance(raw, (int, float, str)) else 0.8
        return LlmClaimsVerifier(caller, prompts, threshold)
    return VERIFIERS.create(name, **params)


def _best_entailment(scores: object) -> float:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if not isinstance(scores, list) or not scores:
        return 0.0
    first = scores[0]
    if isinstance(first, (int, float)):
        return max(float(s) for s in scores)
    # [contradiction, entailment, neutral] is common; pick entailment index 1 when len=3
    vals: list[float] = []
    for row in scores:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            vals.append(float(row[1]))
        elif isinstance(row, (int, float)):
            vals.append(float(row))
    return max(vals) if vals else 0.0
