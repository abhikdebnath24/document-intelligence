from __future__ import annotations

import uuid
from pathlib import Path

from docintel.config import AppConfig
from docintel.config.loader import find_repo_root
from docintel.ingestion.factory import build_ingest_components
from docintel.ingestion.pipeline import IngestReport

PDF_MAGIC = b"%PDF"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PAGES = 300


class UploadError(ValueError):
    pass


def save_upload(raw: bytes, dest_dir: Path) -> Path:
    """Ignore the client filename. uuid path, magic bytes, size/page caps."""
    # PDF 1.7 spec 7.5.2: the header may sit anywhere in the first 1024 bytes.
    if PDF_MAGIC not in raw[:1024]:
        raise UploadError("file is not a PDF (missing %PDF header)")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadError(f"PDF exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{uuid.uuid4()}.pdf"
    path.write_bytes(raw)
    try:
        import pymupdf

        pdf = pymupdf.open(path)  # type: ignore[no-untyped-call]
        try:
            pages = int(pdf.page_count)
        finally:
            pdf.close()  # type: ignore[no-untyped-call]
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise UploadError(f"PDF failed to parse: {exc}") from exc
    if pages < 1:
        path.unlink(missing_ok=True)
        raise UploadError("PDF has no pages")
    if pages > MAX_UPLOAD_PAGES:
        path.unlink(missing_ok=True)
        raise UploadError(f"PDF exceeds {MAX_UPLOAD_PAGES} pages")
    return path


class IngestService:
    def __init__(self, config: AppConfig, *, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or find_repo_root()

    def ingest_paths(self, paths: list[Path], *, only_changed: bool = True) -> IngestReport:
        pipeline = build_ingest_components(self.config, repo_root=self.repo_root)
        try:
            return pipeline.run(paths=paths, only_changed=only_changed)
        finally:
            pipeline.store.close()
            pipeline.registry.close()
