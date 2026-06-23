# 文件作用：作为文档切块能力的公共 facade。
# 关联说明：统一暴露 easy_dataset、structured_text、tree_chunks 等切块能力。

"""Public facade for QA document chunking capabilities."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DEFAULT_CONFIG": ".easy_dataset",
    "ENGINE_VERSION": ".tree_chunks",
    "EasyDatasetChunkingError": ".easy_dataset",
    "ORIGINAL_PROJECT_INPUT_EXTENSIONS": ".easy_dataset",
    "STANDALONE_INPUT_EXTENSIONS": ".easy_dataset",
    "SUPPORTED_SPLIT_TYPES": ".easy_dataset",
    "build_chunk_report": ".structured_text",
    "build_tree_chunks": ".tree_chunks",
    "build_tree_chunks_easy_dataset": ".easy_dataset",
    "combineMarkdown": ".easy_dataset",
    "combine_markdown": ".easy_dataset",
    "correct_markdown_heading_levels": ".markdown_heading_correction",
    "EasyDatasetEpubError": ".epub_preprocessing",
    "extract_chunk_heading": ".structured_text",
    "extract_markdown_heading_lines": ".markdown_heading_correction",
    "extract_outline": ".easy_dataset",
    "extract_table_of_contents": ".easy_dataset",
    "extractOutline": ".easy_dataset",
    "extractTableOfContents": ".easy_dataset",
    "generate_enhanced_summary": ".easy_dataset",
    "generateEnhancedSummary": ".easy_dataset",
    "get_capabilities": ".easy_dataset",
    "get_default_config": ".easy_dataset",
    "getCapabilities": ".easy_dataset",
    "getDefaultConfig": ".easy_dataset",
    "looks_like_ocr_markdown": ".markdown_heading_correction",
    "manual_split": ".easy_dataset",
    "manualSplit": ".easy_dataset",
    "MarkdownHeadingLine": ".markdown_heading_correction",
    "normalize_split_points": ".easy_dataset",
    "normalizeSplitPoints": ".easy_dataset",
    "preprocess_file": ".easy_dataset",
    "preprocessFile": ".easy_dataset",
    "preview_split_points": ".easy_dataset",
    "previewSplitPoints": ".easy_dataset",
    "process_epub": ".epub_preprocessing",
    "process_sections": ".easy_dataset",
    "processSections": ".easy_dataset",
    "save_to_separate_files": ".easy_dataset",
    "saveToSeparateFiles": ".easy_dataset",
    "split_by_headings": ".easy_dataset",
    "split_content": ".easy_dataset",
    "split_file": ".easy_dataset",
    "split_long_section": ".easy_dataset",
    "split_markdown": ".easy_dataset",
    "split_text": ".structured_text",
    "splitByHeadings": ".easy_dataset",
    "splitContent": ".easy_dataset",
    "splitFile": ".easy_dataset",
    "splitLongSection": ".easy_dataset",
    "splitMarkdown": ".easy_dataset",
    "toc_to_markdown": ".easy_dataset",
    "tocToMarkdown": ".easy_dataset",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
