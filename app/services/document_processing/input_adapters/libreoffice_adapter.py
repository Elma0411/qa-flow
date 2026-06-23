"""
LibreOffice based conversion for DOC/DOCX files.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .common import DocumentConversionError, ensure_output_dir, find_pdf_for_stem


logger = logging.getLogger(__name__)


def _candidate_soffice_paths() -> list[str]:
    env_binary = os.getenv("SOFFICE_BINARY") or os.getenv("LIBREOFFICE_BINARY")
    candidates = [env_binary] if env_binary else []
    candidates.extend(["soffice", "libreoffice"])

    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]
        )

    return candidates


def _find_soffice_binary() -> str:
    for candidate in _candidate_soffice_paths():
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate

    raise DocumentConversionError(
        "LibreOffice executable not found. Install libreoffice/libreoffice-writer "
        "or set SOFFICE_BINARY."
    )


def convert_to_pdf(input_path: str, output_dir: str) -> str:
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Office input file not found: {input_path}")

    output_dir_obj = ensure_output_dir(output_dir)
    soffice = _find_soffice_binary()
    command = [
        soffice,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir_obj),
        str(input_path_obj),
    ]

    logger.info("Converting office document to PDF: %s", input_path_obj)
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        detail = "\n".join(
            part
            for part in [
                f"stdout: {exc.stdout.strip()}" if exc.stdout else "",
                f"stderr: {exc.stderr.strip()}" if exc.stderr else "",
            ]
            if part
        )
        raise DocumentConversionError(
            f"LibreOffice failed to convert {input_path_obj} to PDF."
            + (f"\n{detail}" if detail else "")
        ) from exc

    output_pdf = find_pdf_for_stem(output_dir_obj, input_path_obj.stem)
    if output_pdf is None:
        generated = ", ".join(path.name for path in output_dir_obj.glob("*.pdf")) or "none"
        raise DocumentConversionError(
            f"LibreOffice completed but no PDF matching {input_path_obj.stem!r} was found. "
            f"Generated PDFs: {generated}"
        )

    return str(output_pdf)

