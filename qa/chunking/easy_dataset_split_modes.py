# 文件作用：实现 text/token/recursive/code 等非 Markdown 自动切分策略。
# 关联说明：被 easy_dataset.py 的 split_content 调用，和 manual_split、markdown split 并列。

from __future__ import annotations

from typing import Dict, List, Sequence

from .easy_dataset_errors import EasyDatasetChunkingError

_CODE_SEPARATORS: Dict[str, List[str]] = {
    "js": ["\nclass ", "\nfunction ", "\nexport ", "\nconst ", "\nlet ", "\nvar ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "javascript": ["\nclass ", "\nfunction ", "\nexport ", "\nconst ", "\nlet ", "\nvar ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "ts": ["\nclass ", "\nfunction ", "\ninterface ", "\ntype ", "\nexport ", "\nconst ", "\nlet ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "tsx": ["\nfunction ", "\nconst ", "\nexport ", "\nclass ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "py": ["\nclass ", "\ndef ", "\nasync def ", "\nif ", "\nfor ", "\nwhile ", "\ntry:", "\nwith ", "\n\n", "\n", " "],
    "python": ["\nclass ", "\ndef ", "\nasync def ", "\nif ", "\nfor ", "\nwhile ", "\ntry:", "\nwith ", "\n\n", "\n", " "],
    "java": ["\nclass ", "\npublic ", "\nprivate ", "\nprotected ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "go": ["\nfunc ", "\ntype ", "\nif ", "\nfor ", "\nswitch ", "\n\n", "\n", " "],
    "cpp": ["\nclass ", "\nstruct ", "\nvoid ", "\nint ", "\nauto ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "c": ["\nvoid ", "\nint ", "\nchar ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "csharp": ["\nclass ", "\npublic ", "\nprivate ", "\nprotected ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "php": ["\nclass ", "\nfunction ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "],
    "html": ["\n<section", "\n<article", "\n<div", "\n<p", "\n<h1", "\n<h2", "\n\n", "\n", " "],
    "markdown": ["\n# ", "\n## ", "\n### ", "\n#### ", "\n- ", "\n* ", "\n\n", "\n", " "],
}
def _split_fixed(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    return [text[start : start + chunk_size] for start in range(0, len(text), chunk_size) if text[start : start + chunk_size]]
def _apply_overlap(chunks: Sequence[str], chunk_overlap: int) -> List[str]:
    if not chunks:
        return []
    overlap = max(0, int(chunk_overlap))
    if overlap <= 0:
        return [str(chunk) for chunk in chunks if str(chunk).strip()]
    result = [str(chunks[0])]
    for index in range(1, len(chunks)):
        previous = str(chunks[index - 1] or "")
        current = str(chunks[index] or "")
        prefix = previous[-overlap:] if overlap < len(previous) else previous
        merged = f"{prefix}{current}"
        result.append(merged)
    return [chunk for chunk in result if chunk.strip()]
def _recursive_split_impl(text: str, separators: Sequence[str], chunk_size: int) -> List[str]:
    raw_text = str(text or "")
    if len(raw_text) <= chunk_size:
        return [raw_text]
    if not separators:
        return _split_fixed(raw_text, chunk_size)

    separator = str(separators[0] or "")
    remaining = list(separators[1:])
    if separator:
        pieces = raw_text.split(separator)
        if len(pieces) == 1:
            return _recursive_split_impl(raw_text, remaining, chunk_size)
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else f"{current}{separator}{piece}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                if len(current) > chunk_size:
                    chunks.extend(_recursive_split_impl(current, remaining, chunk_size))
                else:
                    chunks.append(current)
            if len(piece) > chunk_size:
                chunks.extend(_recursive_split_impl(piece, remaining, chunk_size))
                current = ""
            else:
                current = piece
        if current:
            if len(current) > chunk_size:
                chunks.extend(_recursive_split_impl(current, remaining, chunk_size))
            else:
                chunks.append(current)
        return [chunk for chunk in chunks if chunk]
    return _split_fixed(raw_text, chunk_size)
def _split_text_mode(content: str, separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = _recursive_split_impl(content, [separator, ""], chunk_size)
    return _apply_overlap(chunks, chunk_overlap)
def _split_recursive_mode(
    content: str,
    separators: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    normalized_separators = [str(item) for item in separators] + [""]
    chunks = _recursive_split_impl(content, normalized_separators, chunk_size)
    return _apply_overlap(chunks, chunk_overlap)
def _split_code_mode(
    content: str,
    split_language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    language = str(split_language or "js").strip().lower()
    separators = _CODE_SEPARATORS.get(language)
    if not separators:
        raise EasyDatasetChunkingError(
            "unsupported_split_language",
            f"Unsupported splitLanguage: {split_language}",
        )
    chunks = _recursive_split_impl(content, separators + [""], chunk_size)
    return _apply_overlap(chunks, chunk_overlap)
def _split_token_mode(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    try:
        import tiktoken  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise EasyDatasetChunkingError(
            "missing_dependency",
            "Missing dependency `tiktoken` required for splitType=token",
        ) from exc
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str(content or ""))
    if not tokens:
        return []
    size = max(1, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), max(0, size - 1)))
    step = max(1, size - overlap)
    result: List[str] = []
    for start in range(0, len(tokens), step):
        piece = encoding.decode(tokens[start : start + size])
        if piece.strip():
            result.append(piece)
        if start + size >= len(tokens):
            break
    return result

__all__ = ["_split_code_mode", "_split_recursive_mode", "_split_text_mode", "_split_token_mode"]
