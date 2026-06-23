"""
图片分析模块数据模型
定义与OCR模块和文本整合模块对接的数据结构
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json


@dataclass
class ImageDescription:
    """
    单张图片的描述结果
    用于存储AI生成的图片描述信息
    """
    image_id: str  # 图片唯一标识（与OCR模块的image_id对应）
    description: str  # AI生成的图片描述文本
    image_type: str = ""  # 图片类别（若启用分类）
    prompt_key: str = ""  # 选中的prompt类别key
    status: str = "success"  # 处理状态：success/error
    error_message: str = ""  # 错误信息（如果处理失败）


@dataclass
class AnalysisResult:
    """
    图片分析完整结果
    存储整个PDF文档中所有图片的分析结果
    """
    pdf_name: str  # 对应的PDF文档名称
    total_images: int  # 总图片数量
    analyzed_images: int  # 成功分析的图片数量
    descriptions: List[ImageDescription]  # 所有图片的描述结果
    processing_time: float = 0.0
    output_dir: Path = None

    def save_descriptions(self, file_path: str):
        """
        保存描述结果到JSON文件
        这是与文本整合模块对接的关键输出文件！

        Args:
            file_path: 输出文件路径
        """
        # 转换为简单的字典格式：{图片ID: 描述}
        # 文本整合模块只需要图片ID和对应的描述文本

        description_dict = {
            desc.image_id: desc.description
            for desc in self.descriptions
            if desc.status == "success"  # 只保存成功的描述
        }

        # 确保输出目录存在
        file_path = Path(file_path)  # 转换为Path对象
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(description_dict, f, ensure_ascii=False, indent=2)
