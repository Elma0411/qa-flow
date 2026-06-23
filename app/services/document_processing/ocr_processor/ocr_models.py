"""
OCR处理器数据模型
定义OCR处理过程中使用的数据结构和结果格式
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from datetime import datetime


@dataclass
class ImageInfo:
    """
    图片信息数据类
    存储从PDF中提取的每张图片的元数据信息
    """
    image_id: str                    # 图片唯一标识（使用模型生成的原始文件名）
    file_path: Path                 # 图片文件保存路径（相对路径）
    page_number: int                # 图片所在的页码
    div_tag: str                    # 在Markdown中的完整<div>标签
    context_before: str = ""        # 图片前的上下文内容
    context_after: str = ""         # 图片后的上下文内容
    width: Optional[int] = None     # 图片宽度
    height: Optional[int] = None     # 图片高度

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于JSON序列化"""
        return {
            "image_id": self.image_id,
            "file_path": str(self.file_path),
            "page_number": self.page_number,
            "div_tag": self.div_tag,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "width": self.width,
            "height": self.height
        }


@dataclass
class OCRResult:
    """
    OCR处理结果数据类
    存储整个PDF文档的OCR识别结果
    """
    pdf_name: str                   # PDF文件名（不含扩展名）
    total_pages: int                # 总页数
    markdown_content: str          # 完整的Markdown格式文本
    images_info: List[ImageInfo]   # 所有图片信息列表
    processing_time: float         # 处理耗时（秒）
    output_dir: Path               # 输出目录路径
    figure_titles: List[Dict] = field(default_factory=list)  # 所有图片标题
    timestamp: str = ""             # 处理时间戳

    def __post_init__(self):
        """初始化后处理：自动设置时间戳"""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于JSON序列化"""
        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "markdown_content": self.markdown_content,
            "images_info": [img.to_dict() for img in self.images_info],
            "figure_titles": self.figure_titles,  # 新增
            "processing_time": self.processing_time,
            "output_dir": str(self.output_dir),
            "timestamp": self.timestamp
        }

    def save_summary_json(self, file_path: Path):
        """保存结果摘要到JSON文件"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)