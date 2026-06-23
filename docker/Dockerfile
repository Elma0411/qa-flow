# syntax=docker/dockerfile:1.6
#
# Dependency image for QA Flow.
#
# Design intent:
# - Keep the PaddleOCR GPU stack on CUDA 12.6/cuDNN.
# - Bring Milvus standalone runtime from the official Milvus image.
# - Install one Python environment for QA Flow runtime paths.
# - Leave source copy/mount and process orchestration to the deployment layer.

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
ARG MILVUS_IMAGE=milvusdb/milvus:2.5-20250912-5d54d844

FROM ${MILVUS_IMAGE} AS milvus-runtime

FROM ${CUDA_IMAGE}

LABEL description="Runtime dependencies for QA Flow document QA service" \
      maintainer="hjk" \
      version="qa-flow-runtime-v0.1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TOKENIZERS_PARALLELISM=false \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    MODEL_BASE_DIR=/app/models \
    SOFFICE_BINARY=/usr/bin/soffice \
    MILVUS_HOST=127.0.0.1 \
    MILVUS_PORT=19530 \
    APP_RUNTIME_ROOT=/app/runtime_assets \
    TORCH_HOME=/app/runtime_assets/cache/torch \
    TRANSFORMERS_CACHE=/app/runtime_assets/cache/transformers \
    HF_HOME=/app/runtime_assets/cache/huggingface \
    LD_LIBRARY_PATH=/milvus/lib:/usr/lib/libreoffice/program:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        pkg-config \
        ca-certificates \
        wget \
        curl \
        vim \
        git \
        unzip \
        supervisor \
        netcat-openbsd \
        default-jre-headless \
        fontconfig \
        fonts-noto-cjk \
        jbig2dec \
        libreoffice-writer \
        poppler-utils \
        libaio1 \
        libaio-dev \
        libopenblas0 \
        libopenblas-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        libgomp1 \
        libjpeg-dev \
        libjpeg-turbo8 \
        zlib1g \
        zlib1g-dev \
        libpng16-16 \
        libpng-dev \
        libfreetype6 \
        libfreetype6-dev \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel

# Milvus server binaries and libraries from hao's selected base image.
COPY --from=milvus-runtime /milvus /milvus
RUN ln -sf /milvus/bin/milvus /usr/local/bin/milvus \
    && chmod +x /usr/local/bin/milvus \
    && ldconfig

ARG ETCD_VERSION=v3.5.5
RUN wget -q "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz" -O /tmp/etcd.tar.gz \
    && tar -xzf /tmp/etcd.tar.gz -C /tmp \
    && mv "/tmp/etcd-${ETCD_VERSION}-linux-amd64/etcd" "/tmp/etcd-${ETCD_VERSION}-linux-amd64/etcdctl" /usr/local/bin/ \
    && rm -rf /tmp/etcd.tar.gz "/tmp/etcd-${ETCD_VERSION}-linux-amd64"

ARG MINIO_RELEASE=minio.RELEASE.2023-03-20T20-16-18Z
RUN wget -q "https://dl.min.io/server/minio/release/linux-amd64/archive/${MINIO_RELEASE}" -O /usr/local/bin/minio \
    && chmod +x /usr/local/bin/minio

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu126/
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

# Install Paddle first, matching dw/docker/dockerfile.
RUN python -m pip install \
        --index-url "${PADDLE_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        paddlepaddle-gpu==3.2.0 \
    && python -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        "paddleocr[all]==3.5.0"

# Resolved union of dw/docker/requirements.txt and hao/requirements.txt.
# Newer shared web/client packages are taken from dw; hao's code only uses
# stable FastAPI/Uvicorn/OpenAI APIs that remain compatible with these versions.
RUN python -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        PyMuPDF==1.27.2.3 \
        pillow==12.2.0 \
        numpy==2.2.6 \
        scipy==1.15.3 \
        pandas==2.3.3 \
        tqdm==4.67.3 \
        easyofd==20260427 \
        docx2python==3.6.2 \
        python-docx==1.2.0 \
        lxml==6.1.1 \
        fastapi==0.136.3 \
        uvicorn==0.48.0 \
        python-multipart==0.0.29 \
        pydantic==2.11.7 \
        openai==2.38.0 \
        httpx==0.28.1 \
        requests==2.32.3 \
        PyYAML==6.0.2 \
        python-dotenv==1.0.1 \
        "six>=1.16.0" \
        "pymilvus>=2.5.0,<2.6" \
        "grpcio>=1.49.0" \
        transformers==4.57.0 \
        "spacy>=3.7.0,<4.0.0" \
        jieba==0.42.1 \
        "scikit-learn>=1.0.0" \
        "matplotlib>=3.5.0" \
        "seaborn>=0.11.0" \
        "openpyxl>=3.1.0" \
        "language-tool-python>=2.7.1" \
        "mammoth>=1.8.0" \
        "markdownify>=0.14.1" \
        "beautifulsoup4>=4.13.0" \
        "tiktoken>=0.9.0" \
        "sentencepiece>=0.1.99" \
        "sacrebleu>=2.4.0" \
        qwen-vl-utils==0.0.14

# These packages declare Torch dependencies. Install their package code now,
# then satisfy Torch/TorchVision in the final fixed layer below.
RUN python -m pip install --no-deps \
        "sentence-transformers>=2.2.0" \
        "bert-score>=0.3.13"

# Keep dw's install order: install project Python dependencies first, then
# force the final image back onto the CUDA 12.6 Torch/TorchVision pair.
RUN python -m pip install --force-reinstall \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        torch==2.7.1 \
        torchvision==0.22.1

RUN mkdir -p \
        /workspace/dw \
        /workspace/hao \
        /app/models \
        /app/runtime_assets/cache/torch \
        /app/runtime_assets/cache/transformers \
        /app/runtime_assets/cache/huggingface \
        /app/runtime_assets/outputs \
        /data/etcd \
        /data/minio \
        /data/milvus \
        /var/log/supervisor

RUN cat <<'EOF' > /usr/local/bin/check-merged-runtime
#!/bin/bash
set -euo pipefail
python - <<'PY'
import cv2
import docx
import docx2python
import easyofd
import fastapi
import fitz
import numpy
import openai
import paddle
import paddleocr
import pandas
import pymilvus
import sentence_transformers
import torch
import torchvision
import transformers
import uvicorn
from PIL import Image

print("python imports ok")
print("paddle", paddle.__version__, "cuda", paddle.is_compiled_with_cuda())
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY
command -v soffice >/dev/null
command -v milvus >/dev/null
command -v etcd >/dev/null
command -v minio >/dev/null
echo "system binaries ok"
EOF
RUN chmod +x /usr/local/bin/check-merged-runtime

WORKDIR /workspace

EXPOSE 11169 10488 12000 19530 9091 2379 2380 9000 9001
VOLUME ["/data/etcd", "/data/minio", "/data/milvus", "/app/runtime_assets", "/app/models"]

CMD ["/bin/bash"]
