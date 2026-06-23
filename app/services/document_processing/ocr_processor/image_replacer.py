import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .ocr_models import OCRResult

try:
    import fitz
except ImportError:  # pragma: no cover - deployment dependency
    fitz = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - Pillow is normally provided by PaddleX
    Image = None
    ImageOps = None


logger = logging.getLogger(__name__)


class ImageReplacer:
    DEFAULT_PDF_RENDER_SCALE = 2.0
    PDF_RENDER_SCALE_ENV = "PADDLE_PDX_PDF_RENDER_SCALE"

    PADDLEX_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
    OPENCV_IMAGE_EXTENSIONS = PADDLEX_IMAGE_EXTENSIONS | {".tif", ".tiff", ".webp"}
    PIL_FALLBACK_IMAGE_EXTENSIONS = {".gif"}
    IMAGE_EXTENSIONS = OPENCV_IMAGE_EXTENSIONS | PIL_FALLBACK_IMAGE_EXTENSIONS
    WRITABLE_IMAGE_EXTENSIONS = OPENCV_IMAGE_EXTENSIONS

    def __init__(
        self,
        margin: int = 3,
        dpi: int = 300,
        source_dpi: Optional[float] = None,
        source_pdf_render_scale: Optional[float] = None,
    ):
        self.margin = margin
        self.dpi = dpi
        self.source_pdf_render_scale = self._resolve_source_pdf_render_scale(
            source_pdf_render_scale=source_pdf_render_scale,
            source_dpi=source_dpi,
        )
        self.source_dpi = self.source_pdf_render_scale * 72.0
        logger.info(
            "Image replacer initialized: margin=%s dpi=%s source_pdf_render_scale=%s source_dpi=%s",
            margin,
            dpi,
            self.source_pdf_render_scale,
            self.source_dpi,
        )

    @classmethod
    def _resolve_source_pdf_render_scale(
        cls,
        source_pdf_render_scale: Optional[float],
        source_dpi: Optional[float],
    ) -> float:
        if source_pdf_render_scale is not None:
            return cls._validate_positive_scale(
                source_pdf_render_scale,
                "source_pdf_render_scale",
            )

        if source_dpi is not None:
            source_dpi_value = float(source_dpi)
            if source_dpi_value <= 0:
                raise ValueError(f"source_dpi must be positive, got {source_dpi}")
            return source_dpi_value / 72.0

        env_value = os.getenv(cls.PDF_RENDER_SCALE_ENV)
        if env_value:
            try:
                return cls._validate_positive_scale(env_value, cls.PDF_RENDER_SCALE_ENV)
            except ValueError as exc:
                logger.warning(
                    "Invalid %s=%r; falling back to PaddleX default scale %.1f: %s",
                    cls.PDF_RENDER_SCALE_ENV,
                    env_value,
                    cls.DEFAULT_PDF_RENDER_SCALE,
                    exc,
                )

        return cls.DEFAULT_PDF_RENDER_SCALE

    @staticmethod
    def _validate_positive_scale(value: float, name: str) -> float:
        scale = float(value)
        if scale <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return scale

    def replace_images_for_ocr_result(
        self,
        original_pdf_path: str,
        ocr_result: OCRResult,
        output_dir: Path,
    ) -> tuple[OCRResult, List[Dict]]:
        if not ocr_result.images_info:
            logger.warning("OCR result has no image info; skipping replacement")
            return ocr_result, []

        img_output_dir = output_dir / "imgs"
        if not img_output_dir.exists():
            logger.error("Image output directory does not exist: %s", img_output_dir)
            return ocr_result, []

        img_backup_dir = output_dir / "imgs_original"
        img_backup_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(original_pdf_path)
        source_kind = self._detect_source_kind(source_path)
        if source_kind is None:
            logger.warning("Unsupported source for image replacement: %s", source_path)
            return ocr_result, []

        logger.info("Starting image replacement from %s source: %s", source_kind, source_path)
        logger.info("OCR image directory: %s", img_output_dir)
        logger.info("Backup image directory: %s", img_backup_dir)

        source_doc = None
        source_image = None
        if source_kind == "pdf":
            if fitz is None:
                logger.error("PyMuPDF is unavailable; cannot replace images from PDF source")
                return ocr_result, []
            try:
                source_doc = fitz.open(str(source_path))
                logger.info("Opened source PDF with %s pages", source_doc.page_count)
            except Exception as exc:
                logger.error("Failed to open source PDF %s: %s", source_path, exc)
                return ocr_result, []
        else:
            source_image = self._read_image(source_path)
            if source_image is None:
                logger.error("Failed to read source image: %s", source_path)
                return ocr_result, []
            logger.info("Loaded source image with shape=%s", getattr(source_image, "shape", None))

        replacement_stats: List[Dict] = []

        try:
            total_count = len(ocr_result.images_info)
            for index, img_info in enumerate(ocr_result.images_info, 1):
                img_stat = {
                    "image_id": img_info.image_id,
                    "page": img_info.page_number,
                    "original_path": str(img_info.file_path),
                    "replaced": False,
                    "success": False,
                    "message": "",
                    "size_change": 0,
                }
                logger.info("Replacing image %s/%s: %s", index, total_count, img_info.image_id)

                try:
                    original_img_path = img_output_dir / Path(img_info.file_path).name
                    if not original_img_path.exists():
                        img_stat["message"] = f"missing OCR image: {original_img_path}"
                        logger.warning(img_stat["message"])
                        replacement_stats.append(img_stat)
                        continue

                    original_size = original_img_path.stat().st_size
                    backup_path = img_backup_dir / Path(img_info.file_path).name
                    try:
                        shutil.copy2(original_img_path, backup_path)
                    except Exception as exc:
                        logger.warning("Failed to back up OCR image %s: %s", original_img_path, exc)

                    pixel_bbox = self._extract_pixel_coords_from_image_id(img_info.image_id)
                    if not pixel_bbox:
                        img_stat["message"] = f"failed to parse coordinates from image_id: {img_info.image_id}"
                        logger.error(img_stat["message"])
                        replacement_stats.append(img_stat)
                        continue

                    target_img_path = img_output_dir / Path(img_info.file_path).name
                    logger.info("Target image path: %s", target_img_path)

                    if source_kind == "pdf":
                        result_path = self._extract_image_from_pdf(
                            pdf_doc=source_doc,
                            page_idx=img_info.page_number - 1,
                            bbox=self._pixel_to_point(pixel_bbox),
                            output_path=str(target_img_path),
                        )
                    else:
                        result_path = self._extract_image_from_image(
                            source_image=source_image,
                            bbox=pixel_bbox,
                            output_path=str(target_img_path),
                        )

                    if not result_path or not Path(result_path).exists():
                        img_stat["message"] = "image extraction failed"
                        logger.error("Image extraction failed for %s", img_info.image_id)
                        replacement_stats.append(img_stat)
                        continue

                    new_size = Path(result_path).stat().st_size
                    img_info.file_path = Path(result_path).relative_to(output_dir)
                    img_stat["replaced"] = True
                    img_stat["success"] = True
                    img_stat["size_change"] = new_size - original_size
                    img_stat["message"] = "image replaced successfully"
                    replacement_stats.append(img_stat)
                    logger.info(
                        "Replaced image %s saved to %s (size change %+d bytes)",
                        img_info.image_id,
                        result_path,
                        img_stat["size_change"],
                    )
                except Exception as exc:
                    img_stat["message"] = f"replacement failed: {exc}"
                    replacement_stats.append(img_stat)
                    logger.error("Replacement failed for %s: %s", img_info.image_id, exc, exc_info=True)
        finally:
            if source_doc is not None:
                source_doc.close()

        success_count = sum(1 for stat in replacement_stats if stat.get("success"))
        failed_count = len(replacement_stats) - success_count
        logger.info(
            "Image replacement completed: success=%s failed=%s output_dir=%s",
            success_count,
            failed_count,
            img_output_dir,
        )
        return ocr_result, replacement_stats

    def _detect_source_kind(self, source_path: Path) -> Optional[str]:
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in self.IMAGE_EXTENSIONS:
            return "image"
        return None

    def _extract_image_from_pdf(
        self,
        pdf_doc,
        page_idx: int,
        bbox: List[float],
        output_path: str,
    ) -> Optional[str]:
        try:
            page = pdf_doc.load_page(page_idx)
            page_rect = page.rect
            x1, y1, x2, y2 = bbox

            expanded_x1 = max(x1 - self.margin, 0)
            expanded_y1 = max(y1 - self.margin, 0)
            expanded_x2 = min(x2 + self.margin, page_rect.width)
            expanded_y2 = min(y2 + self.margin, page_rect.height)

            if expanded_x2 <= expanded_x1 or expanded_y2 <= expanded_y1:
                logger.error(
                    "Invalid PDF crop region on page %s: %s",
                    page_idx + 1,
                    [expanded_x1, expanded_y1, expanded_x2, expanded_y2],
                )
                return None

            clip_rect = fitz.Rect(expanded_x1, expanded_y1, expanded_x2, expanded_y2)
            zoom = self.dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect)

            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(output_path_obj))
            return str(output_path_obj) if output_path_obj.exists() else None
        except Exception as exc:
            logger.error(
                "Failed to extract image from PDF (page=%s bbox=%s): %s",
                page_idx + 1,
                bbox,
                exc,
                exc_info=True,
            )
            return None

    def _extract_image_from_image(
        self,
        source_image: np.ndarray,
        bbox: List[float],
        output_path: str,
    ) -> Optional[str]:
        try:
            if source_image is None or source_image.size == 0:
                return None

            height, width = source_image.shape[:2]
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            expanded_x1 = max(x1 - self.margin, 0)
            expanded_y1 = max(y1 - self.margin, 0)
            expanded_x2 = min(x2 + self.margin, width)
            expanded_y2 = min(y2 + self.margin, height)

            if expanded_x2 <= expanded_x1 or expanded_y2 <= expanded_y1:
                logger.error(
                    "Invalid image crop region: %s within source size %sx%s",
                    [expanded_x1, expanded_y1, expanded_x2, expanded_y2],
                    width,
                    height,
                )
                return None

            cropped = source_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            if cropped.size == 0:
                return None

            output_path_obj = Path(output_path)
            self._write_image(output_path_obj, cropped)
            return str(output_path_obj) if output_path_obj.exists() else None
        except Exception as exc:
            logger.error("Failed to crop source image for bbox=%s: %s", bbox, exc, exc_info=True)
            return None

    def _extract_pixel_coords_from_image_id(self, image_id: str) -> Optional[List[float]]:
        if not image_id:
            logger.warning("Empty image id")
            return None

        pattern = r"(\d+)[_-](\d+)[_-](\d+)[_-](\d+)(?:\.[a-zA-Z]+)?$"
        match = re.search(pattern, image_id)
        if not match:
            logger.warning("Failed to parse coordinates from image id: %s", image_id)
            return None

        try:
            return [
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            ]
        except (ValueError, IndexError) as exc:
            logger.warning("Failed to decode coordinates from %s: %s", image_id, exc)
            return None

    def _pixel_to_point(self, pixel_coords: List[float]) -> List[float]:
        if len(pixel_coords) != 4:
            logger.warning("Expected 4 coordinates, got %s", len(pixel_coords))
            return pixel_coords

        conversion_factor = 1.0 / self.source_pdf_render_scale
        return [value * conversion_factor for value in pixel_coords]

    def _read_image(self, image_path: Path) -> Optional[np.ndarray]:
        image = self._read_image_with_opencv(image_path)
        if image is not None:
            return image
        return self._read_image_with_pillow(image_path)

    def _read_image_with_opencv(self, image_path: Path) -> Optional[np.ndarray]:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    def _read_image_with_pillow(self, image_path: Path) -> Optional[np.ndarray]:
        if Image is None or ImageOps is None:
            return None

        try:
            with Image.open(image_path) as image:
                image.seek(0)
                image = ImageOps.exif_transpose(image)
                if image.mode in {"1", "L", "I", "I;16", "F"}:
                    return np.array(image.convert("L"))

                rgb_image = image.convert("RGB")
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _write_image(self, image_path: Path, image: np.ndarray) -> None:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = image_path.suffix.lower()
        encode_ext = suffix if suffix in self.WRITABLE_IMAGE_EXTENSIONS else ".png"

        if image.ndim == 3 and image.shape[2] == 4 and encode_ext in {".jpg", ".jpeg"}:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        params = []
        if encode_ext in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif encode_ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        success, encoded = cv2.imencode(encode_ext, image, params)
        if not success:
            raise ValueError(f"Failed to encode image as {encode_ext}")
        encoded.tofile(str(image_path))
