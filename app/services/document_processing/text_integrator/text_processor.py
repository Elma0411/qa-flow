"""
Text integration stage.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.services.document_processing.ocr_processor.ocr_models import OCRResult
from app.services.image_understanding.image_models import AnalysisResult, ImageDescription

from .integrator_models import TextIntegrationResult


class SimpleTextIntegrator:
    """
    Merge OCR markdown and image descriptions into final outputs.
    """

    def __init__(self, output_base_dir: str = None):
        if output_base_dir:
            self.output_base_dir = Path(output_base_dir)
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            print(f"文字整合器初始化完成，基础输出目录: {self.output_base_dir}")
        else:
            self.output_base_dir = None
            print("文字整合器初始化完成，使用自定义输出目录")

        self._compile_patterns()

    def _compile_patterns(self):
        self.div_pattern = re.compile(r"<div[^>]*>.*?</div>", re.DOTALL)
        self.src_pattern = re.compile(r'src="([^"]+)"')

    def integrate(
        self,
        ocr_result: OCRResult,
        analysis_result: Optional[AnalysisResult] = None,
        output_dir: Optional[str] = None,
        insert_image_descriptions: bool = True,
    ) -> TextIntegrationResult:
        start_time = time.perf_counter()

        if output_dir is None:
            if self.output_base_dir is None:
                output_dir_path = Path(".") / "integrated_output" / f"integrated_{ocr_result.pdf_name}"
            else:
                output_dir_path = self.output_base_dir / f"integrated_{ocr_result.pdf_name}"
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)

        description_count = len(analysis_result.descriptions) if analysis_result else 0
        print(f"开始文字整合: {ocr_result.pdf_name}")
        print(f"OCR文本长度: {len(ocr_result.markdown_content)} 字符")
        print(f"图片描述数量: {description_count} (插入: {insert_image_descriptions})")
        print(f"输出目录: {output_dir_path}")

        try:
            descriptions = analysis_result.descriptions if analysis_result else []
            description_map = self._build_description_map(descriptions) if insert_image_descriptions else {}

            integrated_markdown, replaced_count = self._integrate_text(
                ocr_result.markdown_content,
                description_map,
                remove_unmatched=not insert_image_descriptions,
            )

            cleaned_text = self._clean_text(integrated_markdown, ocr_result.figure_titles)

            markdown_output_file = output_dir_path / f"{ocr_result.pdf_name}_integrated.md"
            with open(markdown_output_file, "w", encoding="utf-8") as f:
                f.write(integrated_markdown)

            text_output_file = output_dir_path / f"{ocr_result.pdf_name}_integrated.txt"
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            processing_time = time.perf_counter() - start_time

            result = TextIntegrationResult(
                pdf_name=ocr_result.pdf_name,
                original_text_length=len(ocr_result.markdown_content),
                integrated_markdown_length=len(integrated_markdown),
                integrated_text_length=len(cleaned_text),
                total_images=analysis_result.total_images if analysis_result else len(ocr_result.images_info),
                replaced_images=replaced_count,
                processing_time=processing_time,
                markdown_output_file=markdown_output_file,
                text_output_file=text_output_file,
                output_file=text_output_file,
            )

            summary_file = output_dir_path / "integration_summary.json"
            result.save_summary(summary_file)

            print("文字整合完成!")
            print(f"  原始文本: {result.original_text_length} 字符")
            print(f"  Markdown文本: {result.integrated_markdown_length} 字符")
            print(f"  整合文本: {result.integrated_text_length} 字符")
            print(f"  图片替换: {result.replaced_images}/{result.total_images}")
            print(f"  Markdown输出: {markdown_output_file}")
            print(f"  文本输出: {text_output_file}")
            print(f"  处理时间: {processing_time:.2f} 秒")

            return result

        except Exception as e:
            print(f"文字整合失败: {e}")
            raise

    def _build_description_map(self, descriptions: List[ImageDescription]) -> Dict[str, str]:
        description_map = {}
        for desc in descriptions:
            if desc.status == "success" and desc.description:
                description_map[desc.image_id] = desc.description

        print(f"构建描述映射: {len(description_map)} 个有效描述")
        return description_map

    def _integrate_text(
        self,
        markdown_content: str,
        description_map: Dict[str, str],
        remove_unmatched: bool = False,
    ) -> Tuple[str, int]:
        div_matches = list(self.div_pattern.finditer(markdown_content))

        if not div_matches:
            print("警告: 未找到div标签")
            return markdown_content, 0

        print(f"找到 {len(div_matches)} 个div标签")

        current_text = markdown_content
        replaced_count = 0

        # Replace from the end to keep match offsets valid.
        for match in reversed(div_matches):
            div_tag = match.group(0)
            image_id = self._extract_image_id_from_div(div_tag)

            replacement = None
            replaced_image = False

            if image_id and image_id in description_map:
                description = description_map[image_id]
                replacement = f"\n【图片描述：{description}】\n"
                replaced_image = True
                print(f"替换图片: {image_id}")
            elif image_id and remove_unmatched:
                replacement = "\n"
            elif image_id:
                print(f"Warning: missing description for image {image_id}")

            if replacement is not None:
                start, end = match.span()
                current_text = current_text[:start] + replacement + current_text[end:]
                if replaced_image:
                    replaced_count += 1

        return current_text, replaced_count

    def _extract_image_id_from_div(self, div_tag: str) -> Optional[str]:
        src_match = self.src_pattern.search(div_tag)
        if not src_match:
            return None

        src_value = src_match.group(1)
        filename = src_value.split("/")[-1]

        if "." in filename:
            return filename.rsplit(".", 1)[0]

        return None

    def _clean_text(self, text: str, figure_titles: List[Dict] = None) -> str:
        if figure_titles:
            text = self._remove_figure_titles(text, figure_titles)

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            if not line.strip():
                cleaned_lines.append("")
                continue

            line_cleaned = re.sub(r"<[^>]+>", "", line)
            line_cleaned = re.sub(r"^#{1,6}\s+", "", line_cleaned)
            line_cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", line_cleaned)
            line_cleaned = re.sub(r"\*(.*?)\*", r"\1", line_cleaned)
            line_cleaned = re.sub(r"`(.*?)`", r"\1", line_cleaned)
            line_cleaned = re.sub(r"\s+", " ", line_cleaned.strip())

            if line_cleaned:
                cleaned_lines.append(line_cleaned)

        result_lines = []
        prev_empty = False

        for line in cleaned_lines:
            if not line:
                if not prev_empty:
                    result_lines.append("")
                    prev_empty = True
            else:
                result_lines.append(line)
                prev_empty = False

        cleaned_text = "\n".join(result_lines)
        if cleaned_text and not cleaned_text.endswith("\n"):
            cleaned_text += "\n"

        return cleaned_text

    def _remove_figure_titles(self, text: str, figure_titles: List[Dict]) -> str:
        if not figure_titles:
            return text

        print(f"开始删除图片标题，共{len(figure_titles)}个标题")

        title_contents = []
        for title in figure_titles:
            content = title.get("block_content", "").strip()
            if content:
                title_contents.append(content)
                print(f"标记删除的标题: {content[:50]}...")

        if not title_contents:
            return text

        for title_content in title_contents:
            escaped_title = re.escape(title_content)
            pattern = rf"\s*{escaped_title}\s*"
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        print("图片标题删除完成")
        return text
