# 文件作用：将 EPUB 文件解析和转换为后续切块可用的文本。
# 关联说明：作为 easy_dataset 的输入预处理辅助，专门处理 EPUB。

from __future__ import annotations

import posixpath
import re
from io import BytesIO
from typing import Optional
from xml.etree import ElementTree as ET
from zipfile import ZipFile


class EasyDatasetEpubError(RuntimeError):
    pass


def _require_bs4():
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise EasyDatasetEpubError(
            "Missing dependency `beautifulsoup4` required for EPUB preprocessing"
        ) from exc
    return BeautifulSoup


def _require_markdownify():
    try:
        from markdownify import markdownify  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise EasyDatasetEpubError(
            "Missing dependency `markdownify` required for EPUB preprocessing"
        ) from exc
    return markdownify


def _find_first(element: ET.Element, tag_name: str) -> Optional[ET.Element]:
    return element.find(f".//{{*}}{tag_name}")


def _find_all(element: ET.Element, tag_name: str) -> list[ET.Element]:
    return list(element.findall(f".//{{*}}{tag_name}"))


def _read_book_title(opf_root: ET.Element) -> Optional[str]:
    for tag_name in ("title",):
        element = _find_first(opf_root, tag_name)
        if element is not None and (element.text or "").strip():
            return str(element.text or "").strip()
    return None


def _read_chapter_title(html_content: str) -> Optional[str]:
    BeautifulSoup = _require_bs4()
    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception:
        return None

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    for level in range(1, 7):
        heading = soup.find(f"h{level}")
        if heading and heading.get_text(strip=True):
            return heading.get_text(strip=True)

    first_paragraph = soup.find("p")
    if first_paragraph:
        paragraph_text = first_paragraph.get_text(" ", strip=True)
        if paragraph_text and len(paragraph_text) < 100:
            return paragraph_text
    return None


def _extract_body_html(html_content: str) -> str:
    BeautifulSoup = _require_bs4()
    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception:
        return html_content

    body = soup.body or soup
    for tag_name in ("script", "style", "nav", "header", "footer"):
        for node in body.find_all(tag_name):
            node.decompose()
    return "".join(str(node) for node in body.contents) or body.get_text("\n", strip=True)


def process_epub(buffer: bytes) -> str:
    markdownify = _require_markdownify()
    try:
        with ZipFile(BytesIO(buffer)) as archive:
            container_xml = archive.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            container_root = ET.fromstring(container_xml)
            rootfile = _find_first(container_root, "rootfile")
            if rootfile is None:
                raise EasyDatasetEpubError("EPUB container.xml does not declare a rootfile")

            opf_path = str(rootfile.attrib.get("full-path") or "").strip()
            if not opf_path:
                raise EasyDatasetEpubError("EPUB rootfile path is empty")

            opf_xml = archive.read(opf_path).decode("utf-8", errors="ignore")
            opf_root = ET.fromstring(opf_xml)

            manifest_by_id = {}
            for item in _find_all(opf_root, "item"):
                item_id = str(item.attrib.get("id") or "").strip()
                if item_id:
                    manifest_by_id[item_id] = item

            markdown_parts: list[str] = []
            book_title = _read_book_title(opf_root)
            if book_title:
                markdown_parts.append(f"# {book_title}")

            opf_dir = posixpath.dirname(opf_path)
            for itemref in _find_all(opf_root, "itemref"):
                idref = str(itemref.attrib.get("idref") or "").strip()
                if not idref:
                    continue
                manifest_item = manifest_by_id.get(idref)
                if manifest_item is None:
                    continue
                if str(manifest_item.attrib.get("media-type") or "").strip() != "application/xhtml+xml":
                    continue

                href = str(manifest_item.attrib.get("href") or "").strip()
                if not href:
                    continue
                chapter_path = posixpath.normpath(posixpath.join(opf_dir, href))
                chapter_html = archive.read(chapter_path).decode("utf-8", errors="ignore")
                chapter_title = _read_chapter_title(chapter_html)
                if chapter_title and chapter_title != book_title:
                    markdown_parts.append(f"## {chapter_title}")

                body_html = _extract_body_html(chapter_html)
                chapter_markdown = markdownify(body_html, heading_style="ATX")
                chapter_markdown = re.sub(r"\n{3,}", "\n\n", str(chapter_markdown or "")).strip()
                if chapter_markdown:
                    markdown_parts.append(chapter_markdown)
    except EasyDatasetEpubError:
        raise
    except Exception as exc:
        raise EasyDatasetEpubError(f"EPUB preprocessing failed: {exc}") from exc

    return "\n\n".join(part for part in markdown_parts if part).strip()


__all__ = ["EasyDatasetEpubError", "process_epub"]
