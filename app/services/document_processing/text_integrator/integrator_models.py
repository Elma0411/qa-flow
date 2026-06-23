"""
Data models for text integration results.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextIntegrationResult:
    """
    Final outputs from the text integration stage.
    """

    pdf_name: str
    original_text_length: int
    integrated_markdown_length: int
    integrated_text_length: int
    total_images: int
    replaced_images: int
    processing_time: float
    markdown_output_file: Path
    text_output_file: Path
    output_file: Path

    def save_summary(self, file_path: Path):
        """Persist an integration summary as JSON."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "pdf_name": self.pdf_name,
            "original_text_length": self.original_text_length,
            "integrated_markdown_length": self.integrated_markdown_length,
            "integrated_text_length": self.integrated_text_length,
            "markdown_length_change": self.integrated_markdown_length - self.original_text_length,
            "text_length_change": self.integrated_text_length - self.original_text_length,
            "total_images": self.total_images,
            "replaced_images": self.replaced_images,
            "processing_time": self.processing_time,
            "success_rate": (self.replaced_images / self.total_images * 100) if self.total_images > 0 else 0,
            "markdown_output_file": str(self.markdown_output_file),
            "text_output_file": str(self.text_output_file),
            "output_file": str(self.output_file),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
