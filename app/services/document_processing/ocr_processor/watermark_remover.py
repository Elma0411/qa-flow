"""
Watermark/contrast preprocessing for PDF and raster-image inputs.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - deployment dependency
    fitz = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - Pillow is normally provided by PaddleX
    Image = None
    ImageOps = None


class WatermarkRemover:
    PADDLEX_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
    OPENCV_IMAGE_EXTENSIONS = PADDLEX_IMAGE_EXTENSIONS | {".tif", ".tiff", ".webp"}
    PIL_FALLBACK_IMAGE_EXTENSIONS = {".gif"}
    IMAGE_EXTENSIONS = OPENCV_IMAGE_EXTENSIONS | PIL_FALLBACK_IMAGE_EXTENSIONS
    WRITABLE_IMAGE_EXTENSIONS = OPENCV_IMAGE_EXTENSIONS

    def __init__(self, dpi=200, remove_watermark=True):
        self.dpi = dpi
        self.remove_watermark = remove_watermark
        self.temp_dir = None

    def preprocess(self, input_path: str, output_dir: str = None) -> str:
        path_obj = Path(input_path)
        suffix = path_obj.suffix.lower()

        if suffix == ".pdf":
            return self.preprocess_pdf(input_path, output_dir=output_dir)
        if suffix in self.IMAGE_EXTENSIONS:
            return self.preprocess_image(input_path, output_dir=output_dir)
        return input_path

    def normalize_image_for_ocr(self, input_image: str, output_dir: str = None) -> str:
        input_path = Path(input_image)
        suffix = input_path.suffix.lower()
        if suffix in self.PADDLEX_IMAGE_EXTENSIONS:
            return input_image
        if suffix not in self.IMAGE_EXTENSIONS or not input_path.exists():
            return input_image

        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            normalized_image_path = output_dir_path / f"{input_path.stem}_ocr_input.png"
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="ocr_input_normalize_")
            normalized_image_path = Path(self.temp_dir) / "normalized.png"

        try:
            image = self._read_image(input_path)
            if image is None:
                print(f"Image normalization skipped: failed to read image {input_path}")
                return input_image

            self._write_image(normalized_image_path, image)
            print(f"Image normalized for OCR: {input_path} -> {normalized_image_path}")
            return str(normalized_image_path)
        except Exception as exc:
            print(f"Image normalization failed: {exc}")
            import traceback

            traceback.print_exc()
            return input_image

    def preprocess_pdf(self, input_pdf: str, output_dir: str = None) -> str:
        if not self.remove_watermark or not os.path.exists(input_pdf):
            return input_pdf
        if fitz is None:
            print("PDF preprocessing skipped: missing PyMuPDF/fitz dependency")
            return input_pdf

        start_time = time.perf_counter()
        print(f"水印去除: 开始处理 {os.path.basename(input_pdf)}")

        input_path = Path(input_pdf)
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            processed_pdf_path = output_dir_path / f"{input_path.stem}_watermark_removed.pdf"
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="watermark_removal_")
            processed_pdf_path = Path(self.temp_dir) / "processed.pdf"

        doc = None
        processed_doc = None
        try:
            doc = fitz.open(input_pdf)
            processed_doc = fitz.open()
            print(f"  共 {len(doc)} 页")

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_rect = page.rect
                pix = page.get_pixmap(dpi=self.dpi)
                img_array = self._pixmap_to_numpy(pix)

                if pix.n == 4:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                processed = self._enhance_contrast(img_cv)
                image_bytes = self._encode_image_bytes(processed, ".jpg")
                new_page = processed_doc.new_page(
                    width=page_rect.width,
                    height=page_rect.height,
                )
                new_page.insert_image(
                    fitz.Rect(0, 0, page_rect.width, page_rect.height),
                    stream=image_bytes,
                    keep_proportion=False,
                )

                if (page_num + 1) % 5 == 0 or page_num + 1 == len(doc):
                    print(f"  已处理 {page_num + 1}/{len(doc)} 页")

            if processed_pdf_path.exists():
                processed_pdf_path.unlink()
            processed_doc.save(str(processed_pdf_path))
            processed_doc.close()
            processed_doc = None
            doc.close()
            doc = None

            process_time = time.perf_counter() - start_time
            print(f"水印去除完成! 耗时: {process_time:.1f}秒")
            if output_dir:
                print(f"水印去除后的PDF已保存到: {processed_pdf_path}")
            return str(processed_pdf_path)
        except Exception as exc:
            print(f"水印去除失败: {exc}")
            import traceback

            traceback.print_exc()
            return input_pdf
        finally:
            if processed_doc is not None:
                processed_doc.close()
            if doc is not None:
                doc.close()

    def preprocess_image(self, input_image: str, output_dir: str = None) -> str:
        if not self.remove_watermark or not os.path.exists(input_image):
            return input_image

        start_time = time.perf_counter()
        input_path = Path(input_image)
        print(f"水印去除: 开始处理图片 {input_path.name}")

        output_suffix = input_path.suffix.lower()
        if output_suffix not in self.PADDLEX_IMAGE_EXTENSIONS:
            output_suffix = ".png"

        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            processed_image_path = output_dir_path / f"{input_path.stem}_watermark_removed{output_suffix}"
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="watermark_removal_")
            processed_image_path = Path(self.temp_dir) / f"processed{output_suffix}"

        try:
            image = self._read_image(input_path)
            if image is None:
                print(f"图片预处理失败: 无法读取图片 {input_path}")
                return input_image

            processed = self._enhance_contrast(image)
            self._write_image(processed_image_path, processed)

            process_time = time.perf_counter() - start_time
            print(f"水印去除完成! 耗时: {process_time:.1f}秒")
            if output_dir:
                print(f"处理后的图片已保存到: {processed_image_path}")
            return str(processed_image_path)
        except Exception as exc:
            print(f"图片预处理失败: {exc}")
            import traceback

            traceback.print_exc()
            return input_image

    def _pixmap_to_numpy(self, pix):
        img_data = pix.samples
        height, width = pix.height, pix.width
        channels = pix.n

        img_array = np.frombuffer(bytes(img_data), dtype=np.uint8)
        expected_len = height * width * channels
        if len(img_array) != expected_len:
            if len(img_array) < expected_len:
                img_array = np.pad(img_array, (0, expected_len - len(img_array)))
            else:
                img_array = img_array[:expected_len]

        return img_array.reshape((height, width, channels))

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return cv2.convertScaleAbs(result, alpha=1.2, beta=5)

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

        params = []
        if encode_ext in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif encode_ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        success, encoded = cv2.imencode(encode_ext, image, params)
        if not success:
            raise ValueError(f"Failed to encode image as {encode_ext}")
        encoded.tofile(str(image_path))

    def _encode_image_bytes(self, image: np.ndarray, suffix: str = ".jpg") -> bytes:
        encode_ext = suffix.lower()
        if encode_ext not in self.WRITABLE_IMAGE_EXTENSIONS:
            encode_ext = ".jpg"

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
        return encoded.tobytes()

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except OSError:
                pass
