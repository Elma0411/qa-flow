"""
Native DOCX parser that produces OCRResult-compatible output.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from app.services.document_processing.ocr_processor.ocr_models import ImageInfo, OCRResult


logger = logging.getLogger(__name__)

REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
COMPLEX_LAYOUT_PATTERNS = {
    "floating_pictures": "wp:anchor",
    "vml_or_pict": "w:pict",
    "charts": "c:chart",
    "ole_objects": "oleObject",
    "alternate_content": "AlternateContent",
    "text_boxes": "w:txbxContent",
    "vml_text_boxes": "v:textbox",
}
MAX_TEXT_BOXES_FOR_NATIVE = 0


class UnsupportedDocxNativeLayout(RuntimeError):
    """Raised when DOCX native parsing should fall back to PDF conversion."""


@dataclass
class _DocxImage:
    image_id: str
    filename: str
    div_tag: str
    width: int | None = None
    height: int | None = None
    context_before: str = ""
    context_after: str = ""


@dataclass
class _MarkdownBlock:
    kind: str
    text: str = ""
    image: _DocxImage | None = None


def detect_complex_docx_layout(docx_path: str) -> Dict[str, Any]:
    docx_path_obj = Path(docx_path)
    if not docx_path_obj.exists():
        raise FileNotFoundError(f"DOCX input file not found: {docx_path}")

    counts = {key: 0 for key in COMPLEX_LAYOUT_PATTERNS}
    inspected_files: list[str] = []

    with zipfile.ZipFile(docx_path_obj) as archive:
        for name in archive.namelist():
            if not name.startswith("word/") or not name.endswith(".xml"):
                continue
            inspected_files.append(name)
            xml_text = archive.read(name).decode("utf-8", errors="ignore")
            for key, pattern in COMPLEX_LAYOUT_PATTERNS.items():
                counts[key] += xml_text.count(pattern)

    unsupported_reasons = [
        f"{key}={count}"
        for key, count in counts.items()
        if count > 0 and (key != "text_boxes" or count > MAX_TEXT_BOXES_FOR_NATIVE)
    ]

    return {
        "is_complex": bool(unsupported_reasons),
        "unsupported_reasons": unsupported_reasons,
        "counts": counts,
        "inspected_files": inspected_files,
    }


def parse_docx_to_ocr_result(docx_path: str, output_dir: str) -> OCRResult:
    start_time = time.perf_counter()
    docx_path_obj = Path(docx_path)
    if not docx_path_obj.exists():
        raise FileNotFoundError(f"DOCX input file not found: {docx_path}")

    layout_info = detect_complex_docx_layout(str(docx_path_obj))
    if layout_info["is_complex"]:
        reasons = ", ".join(layout_info["unsupported_reasons"])
        raise UnsupportedDocxNativeLayout(
            f"DOCX native parsing does not support this layout: {reasons}"
        )

    docx2python_text = _extract_docx2python_text(docx_path_obj)

    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("DOCX native parsing requires python-docx") from exc

    run_output_dir = Path(output_dir)
    ocr_output_dir = run_output_dir / "ocr_output"
    images_dir = ocr_output_dir / "imgs"
    images_dir.mkdir(parents=True, exist_ok=True)

    document = Document(str(docx_path_obj))
    blocks: list[_MarkdownBlock] = []
    image_counter = 0

    for block in _iter_block_items(document):
        if _is_paragraph(block):
            image_counter = _append_paragraph_blocks(
                paragraph=block,
                blocks=blocks,
                images_dir=images_dir,
                image_counter=image_counter,
            )
        elif _is_table(block):
            table_markdown = _table_to_markdown(block)
            if table_markdown:
                blocks.append(_MarkdownBlock(kind="text", text=table_markdown))

    if not blocks and docx2python_text:
        blocks.append(_MarkdownBlock(kind="text", text=docx2python_text))

    _populate_image_contexts(blocks)
    markdown_content = _blocks_to_markdown(blocks)
    images_info = _images_to_info(blocks)

    result = OCRResult(
        pdf_name=docx_path_obj.stem,
        total_pages=1,
        markdown_content=markdown_content,
        images_info=images_info,
        figure_titles=[],
        processing_time=time.perf_counter() - start_time,
        output_dir=ocr_output_dir,
    )

    _save_ocr_artifacts(result, ocr_output_dir, layout_info)
    logger.info(
        "DOCX native parsing completed: %s text_chars=%s images=%s",
        docx_path_obj,
        len(markdown_content),
        len(images_info),
    )
    return result


def _extract_docx2python_text(docx_path: Path) -> str:
    try:
        from docx2python import docx2python
    except ImportError:
        logger.warning("docx2python is not installed; using python-docx native extraction only")
        return ""

    result = None
    try:
        result = docx2python(str(docx_path))
        return _normalize_text(str(getattr(result, "text", "") or ""))
    except Exception:
        logger.warning("docx2python pre-extraction failed; continuing with python-docx", exc_info=True)
        return ""
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close docx2python result", exc_info=True)


def _iter_block_items(document: Any) -> Iterable[Any]:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _is_paragraph(block: Any) -> bool:
    return block.__class__.__name__ == "Paragraph"


def _is_table(block: Any) -> bool:
    return block.__class__.__name__ == "Table"


def _append_paragraph_blocks(
    *,
    paragraph: Any,
    blocks: list[_MarkdownBlock],
    images_dir: Path,
    image_counter: int,
) -> int:
    paragraph_text = _paragraph_to_markdown_text(paragraph)
    if paragraph_text:
        blocks.append(_MarkdownBlock(kind="text", text=paragraph_text))

    for rel_id in _paragraph_image_relationship_ids(paragraph):
        image_counter += 1
        image = _save_related_image(paragraph.part, rel_id, images_dir, image_counter)
        blocks.append(_MarkdownBlock(kind="image", image=image))

    return image_counter


def _paragraph_to_markdown_text(paragraph: Any) -> str:
    text = _normalize_text(paragraph.text)
    if not text:
        return ""

    style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
    heading_level = _heading_level(style_name)
    if heading_level:
        return f"{'#' * heading_level} {text}"

    return text


def _heading_level(style_name: str) -> int | None:
    match = re.match(r"Heading\s+([1-6])$", style_name.strip(), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    if style_name.strip().lower() in {"title", "subtitle"}:
        return 1

    return None


def _paragraph_image_relationship_ids(paragraph: Any) -> list[str]:
    rel_ids: list[str] = []
    seen: set[str] = set()

    for element in paragraph._element.iter():
        if not str(element.tag).endswith("}blip"):
            continue
        rel_id = element.attrib.get(f"{{{REL_NS}}}embed") or element.attrib.get(f"{{{REL_NS}}}link")
        if rel_id and rel_id not in seen:
            rel_ids.append(rel_id)
            seen.add(rel_id)

    return rel_ids


def _save_related_image(part: Any, rel_id: str, images_dir: Path, image_counter: int) -> _DocxImage:
    related_part = part.related_parts[rel_id]
    suffix = _image_suffix(related_part)
    filename = f"docx_image_{image_counter:04d}{suffix}"
    output_path = images_dir / filename
    output_path.write_bytes(related_part.blob)

    width, height = _read_image_dimensions(output_path)
    image_id = Path(filename).stem
    div_tag = (
        '<div style="text-align: center;">'
        f'<img src="imgs/{filename}" alt="Image" width="60%" />'
        "</div>"
    )
    return _DocxImage(
        image_id=image_id,
        filename=filename,
        div_tag=div_tag,
        width=width,
        height=height,
    )


def _image_suffix(related_part: Any) -> str:
    partname = str(getattr(related_part, "partname", "") or "")
    suffix = Path(partname).suffix.lower()
    if suffix:
        return suffix

    content_type = str(getattr(related_part, "content_type", "") or "")
    guessed = mimetypes.guess_extension(content_type)
    return guessed or ".png"


def _read_image_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None

    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        logger.debug("Failed to read DOCX image dimensions: %s", image_path, exc_info=True)
        return None, None


def _table_to_markdown(table: Any) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([_normalize_text(cell.text) for cell in row.cells])

    if not rows:
        return ""

    max_columns = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    header = normalized_rows[0]
    separator = ["---"] * max_columns
    body = normalized_rows[1:]

    lines = [
        "| " + " | ".join(_escape_table_cell(value) for value in header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape_table_cell(value) for value in row) + " |")

    return "\n".join(lines)


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _populate_image_contexts(blocks: Sequence[_MarkdownBlock]) -> None:
    for index, block in enumerate(blocks):
        if block.kind != "image" or block.image is None:
            continue

        block.image.context_before = _nearest_text_before(blocks, index)
        block.image.context_after = _nearest_text_after(blocks, index)


def _nearest_text_before(blocks: Sequence[_MarkdownBlock], index: int) -> str:
    for block in reversed(blocks[:index]):
        if block.kind == "text" and block.text.strip():
            return _strip_markdown(block.text)
    return ""


def _nearest_text_after(blocks: Sequence[_MarkdownBlock], index: int) -> str:
    for block in blocks[index + 1 :]:
        if block.kind == "text" and block.text.strip():
            return _strip_markdown(block.text)
    return ""


def _strip_markdown(text: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", text.strip())


def _blocks_to_markdown(blocks: Sequence[_MarkdownBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.kind == "text" and block.text:
            parts.append(block.text)
        elif block.kind == "image" and block.image is not None:
            parts.append(block.image.div_tag)

    markdown = "\n\n".join(parts).strip()
    return markdown + "\n" if markdown else ""


def _images_to_info(blocks: Sequence[_MarkdownBlock]) -> list[ImageInfo]:
    images_info: list[ImageInfo] = []
    for block in blocks:
        if block.kind != "image" or block.image is None:
            continue

        image = block.image
        images_info.append(
            ImageInfo(
                image_id=image.image_id,
                file_path=Path("imgs") / image.filename,
                page_number=1,
                div_tag=image.div_tag,
                context_before=image.context_before,
                context_after=image.context_after,
                width=image.width,
                height=image.height,
            )
        )

    return images_info


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _save_ocr_artifacts(result: OCRResult, output_dir: Path, layout_info: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_json_dir = output_dir / "page_json"
    page_json_dir.mkdir(parents=True, exist_ok=True)

    md_file = output_dir / f"{result.pdf_name}.md"
    md_file.write_text(result.markdown_content, encoding="utf-8")

    images_info_file = output_dir / "images_info.json"
    images_info_file.write_text(
        json.dumps([image.to_dict() for image in result.images_info], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result.save_summary_json(output_dir / "ocr_summary.json")
    (page_json_dir / "docx_native_layout.json").write_text(
        json.dumps(layout_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
