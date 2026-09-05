from __future__ import annotations

from pathlib import Path

from docintel.agent.verifiers import LexicalOverlapVerifier, validate_citations
from docintel.core.types import Chunk, Citation, RetrievedChunk

ROOT = Path(__file__).resolve().parents[2]


def _row(cid: str, text: str, doc_id: str = "acme") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc_id=doc_id, text=text, page_start=3, page_end=3, chunk_idx=0),
        score=1.0,
        source="dense",
        rank=1,
    )


def test_citation_validation() -> None:
    rows = [_row("c1", "This Agreement is governed by the laws of Delaware.")]
    ok = Citation(chunk_id="c1", doc_id="acme", page_no=3, bboxes=[], quote="laws of Delaware")
    assert validate_citations([ok], rows) == []
    bad_id = Citation(chunk_id="nope", doc_id="acme", page_no=1, bboxes=[], quote="Delaware")
    assert validate_citations([bad_id], rows)
    bad_quote = Citation(chunk_id="c1", doc_id="acme", page_no=3, bboxes=[], quote="laws of Nevada")
    assert validate_citations([bad_quote], rows)


def test_lexical_verifier_and_adversarial_text_is_just_data() -> None:
    attack = (ROOT / "tests" / "fixtures" / "adversarial_chunk.txt").read_text()
    rows = [
        _row("real", "Governing law: the laws of the State of Delaware."),
        _row("atk", attack),
    ]
    verifier = LexicalOverlapVerifier(threshold=0.2)
    hit = verifier.verify("The contract is governed by Delaware law.", rows)
    assert hit.grounded is True
    miss = verifier.verify("Photosynthesis converts sunlight into glucose.", rows)
    assert miss.grounded is False
