"""
OFD to PDF conversion adapter.
"""

from __future__ import annotations

import base64
import os
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import DocumentConversionError, ensure_output_dir


logger = logging.getLogger(__name__)


@contextmanager
def _temporary_cwd(work_dir: Path):
    previous_cwd = Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def _write_pdf_result(result: Any, output_pdf: Path) -> None:
    if result is None:
        return

    if isinstance(result, (bytes, bytearray)):
        output_pdf.write_bytes(bytes(result))
        return

    if isinstance(result, list) and len(result) == 1:
        _write_pdf_result(result[0], output_pdf)
        return

    if isinstance(result, str):
        result_path = Path(result)
        if result_path.exists():
            shutil.copyfile(result_path, output_pdf)
            return

    if isinstance(result, Path) and result.exists():
        shutil.copyfile(result, output_pdf)
        return

    raise DocumentConversionError(f"Unsupported easyofd PDF result type: {type(result)!r}")


def ofd_to_pdf(ofd_path: str, output_dir: str) -> str:
    ofd_path_obj = Path(ofd_path)
    if not ofd_path_obj.exists():
        raise FileNotFoundError(f"OFD input file not found: {ofd_path}")

    output_dir_obj = ensure_output_dir(output_dir)
    output_pdf = (output_dir_obj / f"{ofd_path_obj.stem}.pdf").resolve()
    work_dir = output_dir_obj / f"{ofd_path_obj.stem}_easyofd_work"

    try:
        import easyofd
    except ImportError as exc:
        raise DocumentConversionError(
            "easyofd is not installed. Install easyofd to convert OFD inputs."
        ) from exc

    converter = None
    try:
        with _temporary_cwd(work_dir):
            converter = easyofd.OFD()
            ofd_b64 = base64.b64encode(ofd_path_obj.read_bytes()).decode("utf-8")
            converter.read(ofd_b64)

            try:
                pdf_result = converter.to_pdf()
            except TypeError:
                pdf_result = converter.to_pdf(str(output_pdf))

            _write_pdf_result(pdf_result, output_pdf)
            if not output_pdf.exists() and pdf_result is None:
                fallback_result = converter.to_pdf(str(output_pdf))
                _write_pdf_result(fallback_result, output_pdf)

        if not output_pdf.exists():
            raise DocumentConversionError(
                f"easyofd conversion completed but output PDF was not created: {output_pdf}"
            )

        logger.info("Converted OFD to PDF: %s -> %s", ofd_path_obj, output_pdf)
        return str(output_pdf)
    except Exception as exc:
        if isinstance(exc, DocumentConversionError):
            raise
        raise DocumentConversionError(f"Failed to convert OFD to PDF: {ofd_path_obj}") from exc
    finally:
        if converter is not None and hasattr(converter, "del_data"):
            try:
                converter.del_data()
            except Exception:
                logger.debug("Failed to release easyofd converter state", exc_info=True)
