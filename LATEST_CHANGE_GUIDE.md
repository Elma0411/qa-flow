# Latest Change Guide

更新时间：2026-07-07（Asia/Shanghai）

## Objective

让 Docker 正式运行容器默认启用 QA Flow 主 API 热重载，避免 bind mount 代码更新后运行中的 API 进程继续使用旧 Python 模块。

## What Changed

- `docker/start-qa-flow-services.sh`
  - QA Flow API 启动不再写死 `reload=False`。
  - 新增 `QA_FLOW_API_RELOAD` 开关，启动时打印实际 reload 状态。
- `docker/runtime-common.sh`
  - 设置 `QA_FLOW_API_RELOAD=true` 作为运行默认值。
- `docker/docker-compose.yml`
  - 正式 runtime 环境新增 `QA_FLOW_API_RELOAD`，默认 `true`。
- `docker/docker-compose.debug.yml`
  - debug runtime 环境同步新增 `QA_FLOW_API_RELOAD`，默认 `true`。
- `docker/start-debug-shell.sh`
  - 手动启动 API 的提示命令改为 `run_api(reload=True)`。
- `docs/docker_compose_parameters.md`
  - 记录 `QA_FLOW_API_RELOAD` 的默认值和作用范围。
- `INTEGRATION_CONTRACT.md`
  - 记录 Docker runtime 的热重载行为为共享运行配置。

## Expected Behavior

- `docker compose -f docker/docker-compose.yml up -d` 启动的 `qa-flow-runtime` 默认会以 Uvicorn reload 模式运行主 API。
- 修改挂载进容器的 `app/`、`qa/`、`scripts/` 下 Python 文件后，QA Flow API 会自动重启并加载新代码。
- 如需关闭热重载，可用 `QA_FLOW_API_RELOAD=false docker compose -f docker/docker-compose.yml up -d`。
- OCR API、图片分类服务、Milvus、MinIO、etcd 的启动行为不变。

## Validation

```bash
cd /data2/hjk/qa-flow

bash -n docker/runtime-common.sh \
  docker/start-qa-flow-services.sh \
  docker/start-debug-shell.sh

docker compose -f docker/docker-compose.yml config >/tmp/qa-flow-compose.yml
docker compose -f docker/docker-compose.debug.yml config >/tmp/qa-flow-compose-debug.yml

docker compose -f docker/docker-compose.yml restart qa-flow-runtime
curl http://localhost:12000/test-connection
docker exec qa-flow-runtime sh -lc 'ps -eo pid,lstart,cmd | grep -E "uvicorn|run_api" | grep -v grep'
```
