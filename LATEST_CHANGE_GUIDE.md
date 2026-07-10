# Latest Change Guide

更新时间：2026-07-10（Asia/Shanghai）

## Objective

收缩 Docker 部署默认宿主机端口，避免 QA Flow 与同机其他容器因 Milvus、etcd、MinIO、分类器等调试端口发生冲突。

## What Changed

- `docker/docker-compose.yml` 和 `docker/docker-compose.debug.yml`
  - 默认只发布 `QA_FLOW_API_HOST_PORT:12000` 与 `OCR_API_HOST_PORT:11169`。
  - 不再发布图片分类器、Milvus、Milvus metrics、etcd、MinIO API、MinIO Console 的宿主机端口。
- `docker/Dockerfile`
  - `EXPOSE` 元数据收窄为 `11169 12000`，与 Compose 默认发布面保持一致。
- `docs/docker_compose_parameters.md` 和 `INTEGRATION_CONTRACT.md`
  - 记录非业务调试端口保持容器内访问，由 healthcheck 与 `/environment-check` 进行内部探测。

## Expected Behavior

- `docker compose -f docker/docker-compose.yml up -d` 只占用宿主机 `12000` 和 `11169`。
- 容器内 Milvus、etcd、MinIO、Milvus metrics 和图片分类器继续按原端口启动，主 API 可正常访问它们。
- 前端的一键环境检测继续通过 `http://localhost:12000/environment-check` 覆盖依赖、模型、Milvus、LLM、OCR、CUDA 和运行目录等检查。

## Validation

```bash
cd /data2/hjk/qa-flow

docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.debug.yml config
curl http://localhost:12000/environment-check
```
