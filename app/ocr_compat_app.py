"""Standalone dw-compatible OCR FastAPI app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.ocr_compat_routes import router as ocr_router


def create_app() -> FastAPI:
    app = FastAPI(title="dw-compatible OCR service", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(ocr_router)
    return app


app = create_app()

