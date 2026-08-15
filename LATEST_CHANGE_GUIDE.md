# Latest Change Guide

更新时间：2026-08-15（Asia/Shanghai）

## Objective

统一 QA Flow 的并发资源模型，并让集成文档流水线支持“文件就绪即交接、问答生成与评估重叠执行”。本轮不再让多个旧的阶段参数互相钳制同一类请求。

## Effective Changes

- `file_concurrency` 是文件级槽位上限。集成文档流程使用有界文件队列：文件 A 完成 OCR/文本预处理后立即进入 chunk、planner、问答生成和评估，文件 B 可以同时继续 OCR；队列满时自动回压。
- `ocr_concurrency` 只控制 OCR 资源；`vision_model_concurrency` 只控制图片理解 VLM；`text_model_concurrency` 是文本模型共享池，覆盖 chunk summary、Point/Summary planner、候选题、问题编辑、答案、图片契合度、增广、普通 LLM 评估和无监督 Faithfulness 的假设句改写。
- `evaluation_concurrency` 只控制本地评估 worker/调度，不再限制 LLM 请求。独立评估页面和接口也改用 `text_model_concurrency` 与 `evaluation_concurrency`，删除旧的 hypothesis 专用并发字段。
- 前端任务设置只保留一组五项并发控件（文件、OCR、图片模型、文本模型、评估）；标准流程自动忽略 OCR/图片模型值，一体流程提交完整五项，避免新旧控件重复渲染和同名参数重复提交。
- 前端 `app.js` 资源版本已更新，避免浏览器继续使用旧 HTML/脚本组合导致性能面板只显示三项。
- 共享 LLM client 使用显式资源池并发配置；无监督 Faithfulness 改为复用同一 client pool，避免每个线程各自创建 client 后把文本请求数放大。
- 普通 `llm` 评估采用有界队列。generation unit 产出一个通过校验的 QA 后立即进入评估；LLM 评估按单 QA 刷新，本地评估保留小批量以避免重复模型初始化。生成端继续运行，队列满时产生回压。
- Point/Summary planner 继续并行建立两类候选池，但 planner 与其它文本阶段共用 `text_model_concurrency`，不再有单独的 planner 并发配置。
- Docker compose、集成 API 文档、集成契约、批量/独立评估前端字段同步为五个并发参数；旧 `doc_*`、`image_*`、`chunk_*`、`eval_*`、`faithfulness_hypothesis_max_concurrency` 公共字段移除。

## Expected Behavior

任务状态中会记录五个 resolved 并发值以及 `streaming_files`。集成任务的 `file_progress` 可以看到文件预处理阶段和 QA 生成/评估阶段交错推进；LLM 评估阶段会显示流式批次数和已评估数量。无监督评估仍在文件生成完成后运行其本地模型套件，因为它需要统一的文档级批次和聚合结果。

## Validation

已通过：

```bash
git diff --check
for f in static/*.js; do node --check "$f"; done
docker exec qa-flow-runtime sh -lc 'cd /app && python -m compileall -q app qa scripts tests && python -m unittest discover -s tests -v'
docker exec qa-flow-runtime sh -lc 'cd /app && python -c "import app.main"'
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/health
```

修改文件需保持 UTF-8 无 BOM。`AGENTS.md` 的已有本地修改属于用户既有内容，本轮不应提交。
