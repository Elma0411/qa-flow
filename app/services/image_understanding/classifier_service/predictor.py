import base64
import io
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.nn import functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from class_config import CLASS_NAMES, DEFAULT_CLASS_CONFIG, get_class_config_by_model_label
    from model_paths import classifier_weight_path
    from mobilenet_model import build_model
    from transform import NormalTransform
else:
    from .class_config import CLASS_NAMES, DEFAULT_CLASS_CONFIG, get_class_config_by_model_label
    from .model_paths import classifier_weight_path
    from .mobilenet_model import build_model
    from .transform import NormalTransform


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CLASSIFIER_WEIGHTS = classifier_weight_path("best_classifier_MobileNetV3Classifier.pth")


class ImageClassifier:
    def __init__(self, weights_path: Path = DEFAULT_CLASSIFIER_WEIGHTS):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = build_model(weights_path=weights_path, num_classes=len(CLASS_NAMES))
        self.model.to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()
        self.transform = NormalTransform(input_size=512)

    def predict_batch(self, images: List[Image.Image]) -> List[dict]:
        transformed = [self.transform(image) for image in images]
        image_tensor = torch.stack(transformed).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        probabilities = F.softmax(outputs, dim=1).tolist()
        _, predicted_indices = torch.max(outputs, 1)

        results = []
        for predicted_idx, score_vector in zip(predicted_indices.tolist(), probabilities):
            class_config = get_class_config_by_model_label(CLASS_NAMES[predicted_idx])
            results.append(
                {
                    "class_id": class_config.class_id,
                    "model_label": class_config.model_label,
                    "category_key": class_config.category_key,
                    "display_name": class_config.display_name,
                    "confidence": float(score_vector[class_config.class_id]),
                    "status": "success",
                    "error_message": "",
                }
            )
        return results


def get_classifier() -> ImageClassifier:
    classifier = getattr(get_classifier, "_instance", None)
    if classifier is None:
        classifier = ImageClassifier()
        get_classifier._instance = classifier
    return classifier


def _default_result(error_message: str = "") -> dict:
    return {
        "class_id": DEFAULT_CLASS_CONFIG.class_id,
        "model_label": DEFAULT_CLASS_CONFIG.model_label,
        "category_key": DEFAULT_CLASS_CONFIG.category_key,
        "display_name": DEFAULT_CLASS_CONFIG.display_name,
        "confidence": 0.0,
        "status": "error" if error_message else "success",
        "error_message": error_message,
    }


def _load_image_from_base64(image_base64: str) -> Optional[Image.Image]:
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def classify_base64_images(image_base64_list: List[str]) -> List[dict]:
    images = [_load_image_from_base64(item) for item in image_base64_list]
    valid_indices = [idx for idx, image in enumerate(images) if image is not None]
    valid_images = [images[idx] for idx in valid_indices]
    results = [_default_result("invalid image payload") for _ in images]

    if valid_images:
        predicted_results = get_classifier().predict_batch(valid_images)
        for idx, prediction in zip(valid_indices, predicted_results):
            results[idx] = prediction

    return results
