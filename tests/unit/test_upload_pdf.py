from __future__ import annotations

from pathlib import Path

import pytest

from docintel.service.ingest_service import MAX_UPLOAD_BYTES, UploadError, save_upload


def test_save_upload_rejects_non_pdf(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="not a PDF"):
        save_upload(b"hello", tmp_path)


def test_save_upload_rejects_oversize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import docintel.service.ingest_service as mod

    monkeypatch.setattr(mod, "MAX_UPLOAD_BYTES", 8)
    with pytest.raises(UploadError, match="exceeds"):
        save_upload(b"%PDF-1.4xxxx", tmp_path)
    assert MAX_UPLOAD_BYTES == 25 * 1024 * 1024


def test_save_upload_writes_uuid_pdf(tmp_path: Path) -> None:
    import pymupdf

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    doc.new_page()
    doc.save(src)
    doc.close()
    dest = tmp_path / "uploads"
    path = save_upload(src.read_bytes(), dest)
    assert path.parent == dest
    assert path.suffix == ".pdf"
    assert path.stem != "src"


def test_save_upload_keeps_safe_stem(tmp_path: Path) -> None:
    import pymupdf

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    doc.new_page()
    doc.save(src)
    doc.close()
    dest = tmp_path / "uploads"
    path = save_upload(
        src.read_bytes(),
        dest,
        filename="SECURIAN FUNDS MAINTENANCE AGREEMENT.PDF",
    )
    assert path.stem.lower().startswith("securian funds maintenance agreement__")
