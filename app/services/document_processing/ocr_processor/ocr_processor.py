"""
OCR处理器
使用PaddleOCR PPStructureV3进行PDF文档的OCR识别
"""

import time
import re
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from paddleocr import PPStructureV3

from .watermark_remover import WatermarkRemover
from .ocr_models import OCRResult, ImageInfo
from .image_replacer import ImageReplacer  # 新增导入

# 配置日志
logger = logging.getLogger(__name__)


def resolve_model_base_dir(model_base_dir: str) -> Path:
    configured_path = Path(model_base_dir).expanduser()
    if configured_path.exists():
        return configured_path.resolve()

    fallback_path = Path("/app/runtime_assets/models/ocr")
    if fallback_path.exists():
        logger.warning(
            "Configured model directory not found: %s; falling back to %s",
            configured_path,
            fallback_path,
        )
        return fallback_path.resolve()

    return configured_path.resolve()


class SimpleOCRProcessor:
    """
    OCR处理器
    支持双向跨页上下文搜索
    """

    def __init__(self, model_base_dir: str, use_gpu: bool = True,
                 remove_watermark: bool = True, watermark_dpi: int = 200,
                 replace_images: bool = True):
        """
        初始化OCR处理器

        Args:
            model_base_dir: 模型目录
            use_gpu: 是否使用GPU
            remove_watermark: 是否去除水印
            watermark_dpi: 水印去除的分辨率
        """
        self.model_base_dir = resolve_model_base_dir(model_base_dir)
        self.use_gpu = use_gpu
        self.default_remove_watermark = remove_watermark
        self.default_watermark_dpi = watermark_dpi

        self.replace_images = replace_images  # 新增

        # 水印去除器

        # 图片替换器
        self.image_replacer = None
        if self.replace_images:
            # PPStructure/PaddleX PDF bboxes are in rendered-image pixels.
            # ImageReplacer reads the same render scale from PADDLE_PDX_PDF_RENDER_SCALE.
            self.image_replacer = ImageReplacer()

        self._predict_lock = threading.Lock()
        self.pipeline = None
        self._initialize_pipeline()
        logger.info("OCR处理器初始化完成")

    def _initialize_pipeline(self):
        """初始化PPStructureV3模型管道"""
        try:
            logger.info("正在初始化PPStructureV3模型...")
            self.pipeline = PPStructureV3(
                # 布局检测模型
                layout_detection_model_name="PP-DocLayout_plus-L",
                layout_detection_model_dir=str(self.model_base_dir / "PP-DocLayout_plus-L"),
                # 图表识别模型
                chart_recognition_model_name="PP-Chart2Table",
                chart_recognition_model_dir=str(self.model_base_dir / "PP-Chart2Table"),
                # 区域检测模型
                region_detection_model_name="PP-DocBlockLayout",
                region_detection_model_dir=str(self.model_base_dir / "PP-DocBlockLayout"),
                # 文档方向分类模型
                doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
                doc_orientation_classify_model_dir=str(self.model_base_dir / "PP-LCNet_x1_0_doc_ori"),
                # 文档矫正模型
                doc_unwarping_model_name="UVDoc",
                doc_unwarping_model_dir=str(self.model_base_dir / "UVDoc"),
                # 文本检测模型
                text_detection_model_name="PP-OCRv5_server_det",
                text_detection_model_dir=str(self.model_base_dir / "PP-OCRv5_server_det"),
                # 文本行方向模型
                textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
                textline_orientation_model_dir=str(self.model_base_dir / "PP-LCNet_x1_0_textline_ori"),
                # 文本识别模型
                text_recognition_model_name="PP-OCRv5_server_rec",
                text_recognition_model_dir=str(self.model_base_dir / "PP-OCRv5_server_rec"),
                # 表格分类模型
                table_classification_model_name="PP-LCNet_x1_0_table_cls",
                table_classification_model_dir=str(self.model_base_dir / "PP-LCNet_x1_0_table_cls"),
                # 有线表格结构识别模型
                wired_table_structure_recognition_model_name="SLANeXt_wired",
                wired_table_structure_recognition_model_dir=str(self.model_base_dir / "SLANeXt_wired"),
                # 无线表格结构识别模型
                wireless_table_structure_recognition_model_name="SLANet_plus",
                wireless_table_structure_recognition_model_dir=str(self.model_base_dir / "SLANet_plus"),
                # 有线表格单元格检测模型
                wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",
                wired_table_cells_detection_model_dir=str(self.model_base_dir / "RT-DETR-L_wired_table_cell_det"),
                # 无线表格单元格检测模型（设为None，不使用）
                wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",
                wireless_table_cells_detection_model_dir=None,
                # 表格方向分类模型（设为None，不使用）
                table_orientation_classify_model_name=None,
                table_orientation_classify_model_dir=None,
                # 公式识别模型
                formula_recognition_model_name="PP-FormulaNet_plus-L",
                formula_recognition_model_dir=str(self.model_base_dir / "PP-FormulaNet_plus-L"),
                # 功能开关配置
                use_doc_orientation_classify=None,  # 使用默认值
                use_doc_unwarping=False,  # 关闭文档矫正
                use_textline_orientation=False,  # 关闭文本行方向检测
                use_seal_recognition=False,  # 关闭印章识别
                use_table_recognition=False,  # 关闭表格识别
                use_formula_recognition=False,  # 关闭公式识别
                use_chart_recognition=False,  # 关闭图表识别
                use_region_detection=True,  # 开启区域检测
                # 设备配置
                device="gpu" if self.use_gpu else "cpu"
            )
            logger.info("PPStructureV3模型初始化成功")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        remove_watermark: Optional[bool] = None,
        watermark_dpi: Optional[int] = None,
    ) -> OCRResult:
        """
        处理PDF文档的完整流程
        """
        start_time = time.perf_counter()
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        effective_remove_watermark = (
            self.default_remove_watermark
            if remove_watermark is None
            else bool(remove_watermark)
        )
        effective_watermark_dpi = (
            self.default_watermark_dpi
            if watermark_dpi is None
            else int(watermark_dpi)
        )
        input_suffix = pdf_path_obj.suffix.lower()
        input_kind = "pdf" if input_suffix == ".pdf" else "image"
        watermark_remover = None
        if effective_remove_watermark:
            watermark_remover = WatermarkRemover(
                dpi=effective_watermark_dpi,
                remove_watermark=True
            )

        # 保存原始PDF路径，用于后续图片替换
        original_pdf_path = str(pdf_path_obj)  # 保存原始PDF路径

        logger.info(f"开始处理PDF: {pdf_path_obj.name}")
        logger.info(f"输出目录: {output_dir}")

        # 保存原始PDF路径到类属性，用于后续图片替换
        logger.info(f"图片替换: {'启用' if self.replace_images else '禁用'}")
        logger.info(f"去除水印: {'启用' if effective_remove_watermark else '禁用'}")
        logger.info(f"水印DPI: {effective_watermark_dpi}")
        logger.info(f"原始PDF路径: {original_pdf_path}")

        # 水印去除预处理
        processed_pdf_path = str(pdf_path_obj)
        watermark_removed_pdf_path = None

        if effective_remove_watermark and watermark_remover:
            logger.info("步骤0: 去除PDF水印...")
            try:
                # 调用水印去除器，水印去除后的PDF会自动保存到output_dir
                processed_pdf_path = watermark_remover.preprocess(
                    pdf_path,  # 输入PDF
                    output_dir  # 输出目录
                )

                # 获取水印去除后的PDF路径
                if processed_pdf_path != pdf_path:
                    watermark_removed_pdf_path = processed_pdf_path
                    logger.info(f"水印去除完成，使用处理后的PDF: {watermark_removed_pdf_path}")
                else:
                    logger.warning("水印去除失败，使用原PDF进行OCR处理")

            except Exception as e:
                logger.error(f"水印去除失败，使用原PDF: {e}")
                processed_pdf_path = pdf_path
        if input_kind == "image":
            try:
                image_normalizer = WatermarkRemover(
                    dpi=effective_watermark_dpi,
                    remove_watermark=False,
                )
                normalized_image_path = image_normalizer.normalize_image_for_ocr(
                    processed_pdf_path,
                    output_dir,
                )
                if normalized_image_path != processed_pdf_path:
                    processed_pdf_path = normalized_image_path
                    logger.info("Normalized image input for OCR: %s", processed_pdf_path)
            except Exception as e:
                logger.error("Image input normalization failed, using original image: %s", e)
                processed_pdf_path = pdf_path

        try:
            # 步骤 1: 使用 PPStructureV3 处理 PDF
            logger.info("步骤 1: 使用 PPStructureV3 进行 PDF 识别...")
            logger.info("Waiting for shared OCR predict lock")
            with self._predict_lock:
                logger.info("Acquired shared OCR predict lock")
                output = self.pipeline.predict(input=str(processed_pdf_path))

            # 步骤2: 先收集所有页的块信息，为跨页搜索做准备
            logger.info("步骤2: 收集所有页的块信息...")
            all_pages_blocks = []  # 存储每页的块信息
            all_markdown_pages = []

            for page_num, page_result in enumerate(output, 1):
                # 获取当前页的块信息
                page_blocks = self._get_blocks_from_page_result(page_result, page_num)
                all_pages_blocks.append(page_blocks)

                # 保存markdown内容
                md_info = page_result.markdown
                all_markdown_pages.append(md_info)
                logger.info(f"收集第 {page_num} 页的块信息: {len(page_blocks)} 个块")

            # 步骤3: 处理每一页的图片，支持跨页搜索
            all_images_info = []
            all_figure_titles = []

            for page_num, page_result in enumerate(output, 1):
                logger.info(f"步骤3: 处理第 {page_num} 页的图片...")

                # 创建JSON输出目录
                json_dir = output_dir_obj / "page_json"
                json_dir.mkdir(parents=True, exist_ok=True)

                # 保存当前页的JSON结果
                json_file = json_dir / f"page_{page_num}.json"
                page_result.save_to_json(save_path=str(json_file))
                logger.info(f"第 {page_num} 页JSON结果已保存: {json_file}")


                # 获取当前页、上一页和下一页的块信息
                current_page_blocks = all_pages_blocks[page_num-1]
                prev_page_blocks = all_pages_blocks[page_num-2] if page_num > 1 else None
                next_page_blocks = all_pages_blocks[page_num] if page_num < len(all_pages_blocks) else None

                # 处理当前页的图片
                md_info = all_markdown_pages[page_num-1]
                page_images = md_info.get("markdown_images", {})

                if page_images:
                    page_images_info, page_titles = self._process_page_images_with_context(
                        page_images, current_page_blocks, page_num, output_dir_obj,
                        prev_page_blocks, next_page_blocks
                    )
                    all_images_info.extend(page_images_info)
                    all_figure_titles.extend(page_titles)
                    logger.info(f"第 {page_num} 页提取了 {len(page_images_info)} 张图片，{len(page_titles)} 个标题")
                else:
                    logger.info(f"第 {page_num} 页没有图片")

            # 步骤4: 合并所有页的markdown内容
            logger.info("步骤4: 合并所有页的Markdown内容...")
            combined_markdown_result = self.pipeline.concatenate_markdown_pages(all_markdown_pages)
            
            # 兼容不同环境/版本的PaddleOCR返回类型
            combined_markdown = self._extract_markdown_string(combined_markdown_result)

            # 步骤5: 为图片添加div标签信息
            logger.info("步骤5: 提取图片位置信息...")
            images_with_div_tags = self._extract_image_div_tags(combined_markdown, all_images_info)

            # 步骤6: 替换图片为原始PDF的高质量图片
            if self.replace_images and self.image_replacer and images_with_div_tags:
                logger.info("步骤6: 替换图片为原始PDF中的高质量图片...")

                try:
                    # 创建临时OCR结果用于替换
                    ocr_result_for_replacement = OCRResult(
                        pdf_name=pdf_path_obj.stem,
                        total_pages=len(all_markdown_pages),
                        markdown_content=combined_markdown,
                        images_info=images_with_div_tags,
                        figure_titles=all_figure_titles,
                        processing_time=time.perf_counter() - start_time,
                        output_dir=output_dir_obj
                    )

                    # 替换图片
                    ocr_result_updated, replacement_stats = self.image_replacer.replace_images_for_ocr_result(
                        original_pdf_path=original_pdf_path,  # 使用原始PDF
                        ocr_result=ocr_result_for_replacement,
                        output_dir=output_dir_obj
                    )

                    # 更新图片信息
                    images_with_div_tags = ocr_result_updated.images_info

                    # 记录替换结果
                    success_count = sum(1 for stat in replacement_stats if stat.get('success', False))
                    logger.info(
                        f"图片替换完成: 成功 {success_count} 张, 失败 {len(replacement_stats) - success_count} 张")

                except Exception as e:
                    logger.error(f"图片替换过程中出错: {e}", exc_info=True)
                    logger.warning("继续使用原始图片，跳过图片替换步骤")

            # 计算处理时间
            processing_time = time.perf_counter() - start_time

            # 创建结果对象
            result = OCRResult(
                pdf_name=pdf_path_obj.stem,
                total_pages=len(all_markdown_pages),
                markdown_content=combined_markdown,
                images_info=images_with_div_tags,
                figure_titles=all_figure_titles,
                processing_time=processing_time,
                output_dir=output_dir_obj
            )

            # 步骤7: 保存输出文件
            logger.info("步骤7: 保存输出文件...")
            self._save_output_files(result, output_dir_obj)
            logger.info(f"PDF处理完成! 共{len(all_markdown_pages)}页, {len(all_images_info)}张图片, 耗时{processing_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise

    def _extract_markdown_string(self, result: Any) -> str:
        """
        从concatenate_markdown_pages的返回结果中提取markdown字符串
        兼容多种可能的返回类型
        
        Args:
            result: concatenate_markdown_pages的返回值，类型不确定
            
        Returns:
            str: 提取的markdown字符串
        """
        # 情况1: 直接返回字符串
        if isinstance(result, str):
            logger.debug("concatenate_markdown_pages 返回 str 类型")
            return result
        
        # 情况2: 返回字典，尝试常见的键名
        if isinstance(result, dict):
            # 尝试多种可能的键名
            possible_keys = ["markdown", "text", "content", "md", "markdown_content", "result"]
            for key in possible_keys:
                if key in result and isinstance(result[key], str):
                    logger.info(f"concatenate_markdown_pages 返回字典类型，从 '{key}' 字段提取内容")
                    return result[key]
            
            # 如果没有找到已知键，检查是否只有一个字符串值
            str_values = [v for v in result.values() if isinstance(v, str)]
            if len(str_values) == 1:
                logger.info("concatenate_markdown_pages 返回字典类型，提取唯一的字符串值")
                return str_values[0]
            
            # 记录字典的键以便调试
            logger.warning(f"concatenate_markdown_pages 返回字典类型，但未找到已知键。可用键: {list(result.keys())}")
            return ""
        
        # 情况3: 返回列表（可能是多页内容的列表）
        if isinstance(result, list):
            str_items = []
            for item in result:
                if isinstance(item, str):
                    str_items.append(item)
                elif isinstance(item, dict):
                    # 递归提取
                    extracted = self._extract_markdown_string(item)
                    if extracted:
                        str_items.append(extracted)
            if str_items:
                logger.info(f"concatenate_markdown_pages 返回列表类型，合并 {len(str_items)} 个字符串项")
                return "\n\n".join(str_items)
            logger.warning("concatenate_markdown_pages 返回列表类型，但未找到字符串内容")
            return ""
        
        # 情况4: 返回对象，尝试访问常见属性
        if hasattr(result, '__dict__') or hasattr(result, '__slots__'):
            possible_attrs = ["markdown", "text", "content", "md", "markdown_content", "result", "value"]
            for attr in possible_attrs:
                if hasattr(result, attr):
                    value = getattr(result, attr)
                    if isinstance(value, str):
                        logger.info(f"concatenate_markdown_pages 返回对象类型，从 '{attr}' 属性提取内容")
                        return value
                    elif callable(value):
                        # 可能是方法
                        try:
                            called_value = value()
                            if isinstance(called_value, str):
                                logger.info(f"concatenate_markdown_pages 返回对象类型，从 '{attr}()' 方法提取内容")
                                return called_value
                        except Exception:
                            pass
            
            # 记录对象的属性以便调试
            attrs = dir(result)
            public_attrs = [a for a in attrs if not a.startswith('_')]
            logger.warning(f"concatenate_markdown_pages 返回对象类型 {type(result).__name__}，未找到已知属性。可用属性: {public_attrs[:20]}")
        
        # 情况5: None 或空值
        if result is None:
            logger.warning("concatenate_markdown_pages 返回 None")
            return ""
        
        # 情况6: 最后尝试直接转换为字符串
        try:
            str_result = str(result)
            # 检查转换后是否有意义（不是类似 <object at 0x...> 的默认表示）
            if not str_result.startswith('<') and not str_result.endswith('>'):
                logger.warning(f"concatenate_markdown_pages 返回未知类型 {type(result).__name__}，尝试转换为字符串")
                return str_result
        except Exception as e:
            logger.error(f"无法将 concatenate_markdown_pages 结果转换为字符串: {e}")
        
        logger.error(f"无法从 concatenate_markdown_pages 结果中提取 markdown 内容，类型: {type(result)}")
        return ""

    def _get_blocks_from_page_result(self, page_result, page_num: int) -> List[Dict]:
        """
        从page_result获取块信息并按block_id排序
        """
        try:
            # 检查json属性
            if not hasattr(page_result, 'json'):
                logger.warning(f"第 {page_num} 页没有json属性")
                return []

            json_data = page_result.json
            if not isinstance(json_data, dict):
                logger.warning(f"第 {page_num} 页json属性不是字典")
                return []

            # 获取res字典
            res_data = json_data.get('res')
            if not isinstance(res_data, dict):
                logger.warning(f"第 {page_num} 页json['res']不是字典")
                return []

            # 获取parsing_res_list
            parsing_res_list = res_data.get('parsing_res_list')
            if not isinstance(parsing_res_list, list):
                logger.warning(f"第 {page_num} 页json['res']['parsing_res_list']不是列表")
                return []

            # 按block_id排序
            sorted_blocks = sorted(parsing_res_list, key=lambda x: x.get('block_id', 0))
            logger.info(f"第 {page_num} 页获取到 {len(sorted_blocks)} 个块")
            return sorted_blocks

        except Exception as e:
            logger.error(f"第 {page_num} 页提取块信息失败: {e}")
            return []

    def _process_page_images_with_context(self, page_images: Dict, page_blocks: List[Dict],
                                        page_num: int, output_dir: Path,
                                        prev_page_blocks: Optional[List[Dict]] = None,
                                        next_page_blocks: Optional[List[Dict]] = None) -> Tuple[List[ImageInfo], List[Dict]]:
        """
        处理单页图片，并提取上下文和标题
        """
        images_info = []
        figure_titles = []
        images_dir = output_dir / "imgs"
        images_dir.mkdir(parents=True, exist_ok=True)

        if not page_images:
            return images_info, figure_titles

        # 预处理：分类块信息
        text_blocks, image_blocks, title_blocks = self._classify_blocks(page_blocks)

        # 收集本页所有标题
        figure_titles.extend(title_blocks)

        # 处理每个图片
        for img_path, image in page_images.items():
            image_filename = Path(img_path).name
            image_id = Path(img_path).stem
            image_save_path = images_dir / image_filename

            try:
                # 保存图片文件
                image.save(str(image_save_path))

                # 从图片ID提取坐标
                coords = self._extract_coords_from_image_id(image_id)
                if not coords:
                    logger.warning(f"无法从图片ID提取坐标: {image_id}")
                    continue

                print(f"\n处理图片: {image_id}")
                print(f"提取的坐标: {coords}")

                # 查找匹配的图片块
                matched_image_block = self._find_matching_image_block(coords, image_blocks)

                if matched_image_block:
                    print(f"找到匹配的图片块: ID={matched_image_block.get('block_id')}, 标签={matched_image_block.get('block_label')}")
                else:
                    print("未找到匹配的图片块")
                    continue

                context_before, context_after = self._extract_image_context(
                    matched_image_block, page_blocks, prev_page_blocks, next_page_blocks, page_num
                )
                # 创建图片信息对象
                img_info = ImageInfo(
                    image_id=image_id,
                    file_path=image_save_path.relative_to(output_dir),
                    page_number=page_num,
                    div_tag="",  # 稍后填充
                    context_before=context_before,
                    context_after=context_after,
                    width=image.width if hasattr(image, 'width') else None,
                    height=image.height if hasattr(image, 'height') else None
                )
                images_info.append(img_info)
                logger.debug(f"保存图片: {image_save_path.name}")

            except Exception as e:
                logger.error(f"处理图片失败 {image_filename}: {e}")
                continue

        return images_info, figure_titles

    def _classify_blocks(self, blocks: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        将块按类型分类：文本块、图片块、标题块
        """
        text_blocks = []
        image_blocks = []
        title_blocks = []

        for block in blocks:
            block_label = block.get('block_label', '').lower()

            if block_label in ['text', 'paragraph_title']:
                text_blocks.append(block)
            elif block_label in ['image', 'chart', 'table', 'figure', 'formula']:
                image_blocks.append(block)
            elif block_label == 'figure_title':
                title_blocks.append(block)

        return text_blocks, image_blocks, title_blocks

    def _extract_coords_from_image_id(self, image_id: str) -> Optional[List[int]]:
        """
        从图片ID中提取坐标信息
        格式：img_in_image_box_218_796_943_1305
        """
        pattern = r'(\d+)_(\d+)_(\d+)_(\d+)$'
        match = re.search(pattern, image_id)

        if match:
            try:
                coords = [int(match.group(1)), int(match.group(2)),
                         int(match.group(3)), int(match.group(4))]
                return coords
            except ValueError:
                logger.warning(f"图片ID坐标格式错误: {image_id}")
        return None

    def _find_matching_image_block(self, coords: List[int], image_blocks: List[Dict]) -> Optional[Dict]:
        """
        根据坐标查找匹配的图片块
        """
        if not coords or len(coords) != 4:
            return None

        x1, y1, x2, y2 = coords
        print(f"搜索匹配的图片块，坐标: [{x1}, {y1}, {x2}, {y2}]")

        for block in image_blocks:
            block_bbox = block.get('block_bbox', [])
            if len(block_bbox) == 4:
                # 比较坐标是否匹配（允许小的误差）
                bx1, by1, bx2, by2 = block_bbox
                x_diff = abs(bx1 - x1) + abs(bx2 - x2)
                y_diff = abs(by1 - y1) + abs(by2 - y2)
                total_diff = x_diff + y_diff

                if total_diff < 20:  # 允许20像素的总误差
                    print(f"找到匹配块: 坐标差异={total_diff}, 块坐标={block_bbox}")
                    return block

        print("未找到匹配的图片块")
        return None

    def _extract_image_context(self, image_block: Dict, all_blocks: List[Dict],
                               prev_page_blocks: Optional[List[Dict]],
                               next_page_blocks: Optional[List[Dict]],
                               page_num: int) -> Tuple[str, str]:
        """
        上下文提取函数
        在完整块列表中查找图片块位置
        """
        if not image_block or not all_blocks:
            return "", ""

        # 获取图片块的block_id
        image_block_id = image_block.get('block_id')
        if image_block_id is None:
            return "", ""

        print(f"开始为图片块 {image_block_id} 提取上下文...")

        # 在完整块列表中查找图片块位置
        image_index = self._find_block_position_in_all_blocks(image_block_id, all_blocks)
        if image_index == -1:
            return "", ""

        # 提取上文（在完整块列表中向前搜索文本块）
        context_before = self._find_context_before(image_index, all_blocks, prev_page_blocks, page_num)

        # 提取下文（在完整块列表中向后搜索文本块）
        context_after = self._find_context_after(image_index, all_blocks, next_page_blocks, page_num)

        return context_before, context_after

    def _find_block_position_in_all_blocks(self, target_block_id: int, all_blocks: List[Dict]) -> int:
        """
        在完整块列表中查找目标块的位置
        """
        for i, block in enumerate(all_blocks):
            if block.get('block_id') == target_block_id:
                return i
        return -1

    def _find_context_before(self, image_index: int, all_blocks: List[Dict],
                             prev_page_blocks: Optional[List[Dict]], page_num: int) -> str:
        """
        在完整块列表中向上搜索上文
        遇到paragraph_title则停止，支持跨页搜索
        """
        context_parts = []

        # 在当前页的完整块列表中向上搜索
        for i in range(image_index - 1, -1, -1):
            block = all_blocks[i]
            block_label = block.get('block_label', '')
            block_content = block.get('block_content', '')

            # 只收集文本块和段落标题的内容
            if block_label in ['text', 'paragraph_title'] and block_content.strip():
                context_parts.insert(0, block_content)  # 向前插入保持顺序
                print(f"  找到上文块 {i} ({block_label}): {block_content[:50]}...")

            # 遇到paragraph_title则停止
            if block_label == 'paragraph_title':
                break

        # 如果当前页没有上文，且不是第一页，搜索上一页
        if not context_parts and page_num > 1 and prev_page_blocks:
            # 在上一页的完整块列表中搜索
            for i in range(len(prev_page_blocks) - 1, -1, -1):
                block = prev_page_blocks[i]
                block_label = block.get('block_label', '')
                block_content = block.get('block_content', '')

                if block_label in ['text', 'paragraph_title'] and block_content.strip():
                    context_parts.insert(0, block_content)
                    break

        result = " ".join(context_parts)
        return result

    def _find_context_after(self, image_index: int, all_blocks: List[Dict],
                            next_page_blocks: Optional[List[Dict]], page_num: int) -> str:
        """
        在完整块列表中向下搜索下文
        遇到paragraph_title则停止，支持跨页搜索
        """
        context_parts = []

        # 在当前页的完整块列表中向下搜索
        for i in range(image_index + 1, len(all_blocks)):
            block = all_blocks[i]
            block_label = block.get('block_label', '')
            block_content = block.get('block_content', '')

            # 只收集文本块和段落标题的内容
            if block_label in ['text', 'paragraph_title'] and block_content.strip():
                context_parts.append(block_content)

            # 遇到paragraph_title则停止
            if block_label == 'paragraph_title':
                break

        # 如果当前页没有下文，且有下一页，搜索下一页
        if not context_parts and next_page_blocks:
            # 在下一页的完整块列表中搜索
            for block in next_page_blocks:
                block_label = block.get('block_label', '')
                block_content = block.get('block_content', '')

                if block_label in ['text', 'paragraph_title'] and block_content.strip():
                    context_parts.append(block_content)
                    break

        result = " ".join(context_parts)
        return result

    def _extract_image_div_tags(self, markdown_content: str, images_info: List[ImageInfo]) -> List[ImageInfo]:
        """
        为图片添加div标签信息
        """
        updated_images_info = []

        for img_info in images_info:
            image_filename = f"{img_info.image_id}.jpg"

            # 查找包含图片文件名的任何文本
            if image_filename in markdown_content:
                # 查找图片出现的位置
                pos = markdown_content.find(image_filename)

                # 向前查找<div
                div_start = markdown_content.rfind('<div', 0, pos)
                if div_start == -1:
                    # 没有找到<div，跳过
                    updated_images_info.append(img_info)
                    continue

                # 向后查找</div>
                div_end = markdown_content.find('</div>', pos)
                if div_end == -1:
                    # 没有找到</div>，跳过
                    updated_images_info.append(img_info)
                    continue

                # 提取div标签
                div_tag = markdown_content[div_start:div_end + 6]

                # 检查提取的div标签是否包含图片文件名
                if image_filename in div_tag:
                    updated_img_info = ImageInfo(
                        image_id=img_info.image_id,
                        file_path=img_info.file_path,
                        page_number=img_info.page_number,
                        div_tag=div_tag,
                        context_before=img_info.context_before,
                        context_after=img_info.context_after,
                        width=img_info.width,
                        height=img_info.height
                    )
                    updated_images_info.append(updated_img_info)
                else:
                    updated_images_info.append(img_info)
            else:
                updated_images_info.append(img_info)

        return updated_images_info

    def _save_output_files(self, result: OCRResult, output_dir: Path):
        """
        保存输出文件
        """
        try:
            # 保存Markdown文件
            md_file = output_dir / f"{result.pdf_name}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(result.markdown_content)
            logger.info(f"Markdown文件已保存: {md_file}")

            # 保存图片信息JSON文件
            images_info_file = output_dir / "images_info.json"
            images_info_list = [img.to_dict() for img in result.images_info]
            with open(images_info_file, 'w', encoding='utf-8') as f:
                json.dump(images_info_list, f, ensure_ascii=False, indent=2)
            logger.info(f"图片信息已保存: {images_info_file}")

            # 保存处理摘要
            summary_file = output_dir / "ocr_summary.json"
            result.save_summary_json(summary_file)
            logger.info(f"处理摘要已保存: {summary_file}")

        except Exception as e:
            logger.error(f"保存输出文件失败: {e}")
            raise
