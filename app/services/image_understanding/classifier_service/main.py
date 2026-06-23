import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from class_config import CLASS_CONFIGS
    from predictor import classify_base64_images, get_classifier
else:
    from .class_config import CLASS_CONFIGS
    from .predictor import classify_base64_images, get_classifier


app = FastAPI(title="image-classifier", version="1.0.0")


class ImagePayload(BaseModel):
    image_id: str = ""
    image_base64: str = ""


class ClassifyBatchRequest(BaseModel):
    images: List[ImagePayload] = Field(default_factory=list)


def class_catalog() -> List[dict]:
    return [
        {
            "class_id": config.class_id,
            "model_label": config.model_label,
            "category_key": config.category_key,
            "display_name": config.display_name,
        }
        for config in CLASS_CONFIGS
    ]


@app.on_event("startup")
def warmup_model():
    get_classifier()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/classes")
async def classes():
    return {"code": 200, "data": class_catalog()}


@app.post("/classify/batch")
async def classify_batch(req: ClassifyBatchRequest):
    try:
        predictions = classify_base64_images([item.image_base64 for item in req.images])
        return {
            "code": 200,
            "data": [
                {
                    "image_id": item.image_id,
                    **prediction,
                }
                for item, prediction in zip(req.images, predictions)
            ],
            "classes": class_catalog(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("image_classifier_service.main:app", host="0.0.0.0", port=10488)
