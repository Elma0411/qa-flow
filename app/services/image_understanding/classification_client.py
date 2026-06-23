import base64
import json
import os
from dataclasses import dataclass
from typing import List, Optional
from urllib import error, request


DEFAULT_CLASSIFIER_API_BASE = os.getenv("CLASSIFIER_API_BASE", "http://localhost:10488")
DEFAULT_CLASSIFIER_TIMEOUT = 30
DEFAULT_CLASS_ID = 0
DEFAULT_MODEL_LABEL = "其他"
DEFAULT_CATEGORY_KEY = "others"
DEFAULT_DISPLAY_NAME = "其他"

# Legacy 6-class fallback values:
#
# DEFAULT_CLASS_ID = 5
# DEFAULT_MODEL_LABEL = "5"
# DEFAULT_CATEGORY_KEY = "others"
# DEFAULT_DISPLAY_NAME = "其他"


@dataclass(frozen=True)
class ClassificationResult:
    image_id: str
    class_id: int = DEFAULT_CLASS_ID
    model_label: str = DEFAULT_MODEL_LABEL
    category_key: str = DEFAULT_CATEGORY_KEY
    display_name: str = DEFAULT_DISPLAY_NAME
    confidence: float = 0.0
    status: str = "success"
    error_message: str = ""


def normalize_classifier_api_base(api_base: Optional[str]) -> str:
    raw = (api_base or "").strip()
    if not raw:
        return DEFAULT_CLASSIFIER_API_BASE

    normalized = raw.rstrip("/")
    if normalized.endswith("/classify/batch"):
        normalized = normalized[: -len("/classify/batch")]
    return normalized


def default_classification_result(image_id: str, error_message: str = "") -> ClassificationResult:
    return ClassificationResult(
        image_id=image_id,
        status="error" if error_message else "success",
        error_message=error_message,
    )


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def classify_images_batch(
    image_ids: List[str],
    image_paths: List[str],
    api_base: Optional[str] = None,
    timeout: int = DEFAULT_CLASSIFIER_TIMEOUT,
) -> List[ClassificationResult]:
    if len(image_ids) != len(image_paths):
        raise ValueError("image_ids and image_paths must have the same length")
    if not image_ids:
        return []

    endpoint = normalize_classifier_api_base(api_base) + "/classify/batch"
    default_results = {
        image_id: default_classification_result(image_id)
        for image_id in image_ids
    }
    payload_images = []
    for image_id, image_path in zip(image_ids, image_paths):
        try:
            payload_images.append(
                {
                    "image_id": image_id,
                    "image_base64": _image_to_base64(image_path),
                    "image_path": "",
                }
            )
        except Exception as exc:
            default_results[image_id] = default_classification_result(
                image_id,
                f"failed to read image: {exc}",
            )

    if not payload_images:
        return [default_results[image_id] for image_id in image_ids]

    payload = {"images": payload_images}

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return [
            default_classification_result(image_id, f"classifier http error: {exc.code} {detail}")
            for image_id in image_ids
        ]
    except Exception as exc:
        return [
            default_classification_result(image_id, f"classifier request failed: {exc}")
            for image_id in image_ids
        ]

    rows = body.get("data") or []
    row_by_id = {str(row.get("image_id", "")): row for row in rows}
    results: List[ClassificationResult] = []
    for image_id in image_ids:
        row = row_by_id.get(str(image_id))
        if not row:
            fallback = default_results.get(image_id)
            if fallback and fallback.status == "error":
                results.append(fallback)
            else:
                results.append(default_classification_result(image_id, "missing classifier result"))
            continue

        results.append(
            ClassificationResult(
                image_id=image_id,
                class_id=int(row.get("class_id", DEFAULT_CLASS_ID)),
                model_label=str(row.get("model_label", DEFAULT_MODEL_LABEL)),
                category_key=str(row.get("category_key", DEFAULT_CATEGORY_KEY)),
                display_name=str(row.get("display_name", DEFAULT_DISPLAY_NAME)),
                confidence=float(row.get("confidence", 0.0)),
                status=str(row.get("status", "success")),
                error_message=str(row.get("error_message", "")),
            )
        )

    return results
