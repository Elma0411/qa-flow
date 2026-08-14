# Latest Change Guide

更新时间：2026-08-15（Asia/Shanghai）

## Objective

收敛 QA Flow 的“结构材料 → Point/Summary 场景 → 候选问题 → 问题编辑 → 同文档检索 → 答案与证据 → 无参考评估 → 调试审核”链路。当前版本同时修正无参考评分口径、长文档规划审计和调试页面的可见性，保证生成质量问题能够定位到材料路径和模型原始响应。

## Effective Changes

- 无参考评估的 `average_score` 固定为
  `(faithfulness + answerability + coverage_score) / 3`；缺失、非法、非有限或越界值按 `0`，分母始终为 `3`。后端筛选、评测导入、合并产物和普通界面使用同一口径，`filter_basis` 写为 `average_score`。
- 普通无参考评估界面只显示 `faithfulness`、`answerability`、`coverage_score` 和平均分；旧 `p` 只作为 answerability 的输入兼容，`unsupervised_f1`、`coverage_self`、`coverage_recall_soft` 不作为普通展示或过滤分数。
- planner 继续以 `section_path` 合并逻辑材料；planner 回调现在接收并记录批次编号/总批次，运行时的批次字符预算和最大并发会真正传入 `plan_generation_units`。
- planner 单批调用异常会记录在 `scenario_planner_batch_details`，不会让其它批次和 Point 兜底整体失败。批次审计包含请求、返回、校验通过、材料路径、场景意图、读者需求、required/optional 路径和丢弃/错误原因。
- planner 原始请求和 `raw_response` 继续写入 task-scoped debug JSONL；调试接口新增 `planning_batch_index` 与 `planning_scenario_type` 过滤，前端新增“场景规划批次”审计卡片和原始响应查看入口。
- 答案 `evidence_usage` 在持久化边界再次清洗，只保留 `evidence_ref`、`role` 以及后端恢复的 `chunk_id`、`chunk_index`、`title_path`，删除模型自由生成的 `snippet`/`usage`。
- Point/Summary 的 required/optional 覆盖规则和同文档 BGE 证据准入保持不变：Summary 只校验 required 材料，`final_evidence_k` 仍表示最多的补充窗口数，允许实际补充证据为零。
- 保留现有工作台、管理页、评估页的 Logo、配色和 iOS 风格；调试页面继续支持固定等高双面板、内部独立滚动、QA 详情弹窗和 Chunk 正文放大。

## Expected Behavior

- 每个 planner 批次都能在流水线调试视图中看到类型、编号、材料路径、required/optional 来源、意图、读者需求、数量和错误；点击即可读取该类型/批次的原始模型响应。
- planner 某一批次超时或返回异常时，任务仍能继续处理其它批次；Point/auto 模式按现有证据约束策略补足可用材料，异常原因保留在调试信息中。
- 无参考阈值筛选不再读取旧 `unsupervised_f1` 作为平均分；非法指标不会因为缺少某一项而抬高结果。
- 证据审计以服务端材料映射为准，导出的 `evidence_usage` 不含模型自由生成的片段说明。

## Changed Files

- `qa/text_to_qa_pipeline.py`
- `qa/generation/structure_units.py`
- `qa/generation/qa_generation_flow.py`
- `qa/pipeline_runtime.py`
- `app/routers/pipeline_history_routes.py`
- `app/services/storage/consolidation.py`
- `app/services/storage/merge.py`
- `app/services/eval_jobs/result.py`
- `app/services/unsupervised_evaluation/common.py`
- `static/app.js`
- `static/styles.css`
- `INTEGRATION_CONTRACT.md`
- `LATEST_CHANGE_GUIDE.md`
- `QA_GENERATION_CONTEXT_HANDOFF.md`

## Validation

```bash
git diff --check
for f in static/*.js; do node --check "$f"; done
docker exec qa-flow-runtime sh -lc 'cd /app && python -m compileall -q app qa scripts tests && python -m unittest discover -s tests'
docker exec qa-flow-runtime sh -lc 'cd /app && python -c "import app.main"'
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/health
```

主机 Python 3.9 不满足项目的 Python 3.10+ 类型语法和运行依赖，不能用主机测试结果替代 Docker runtime 验证。修改文件须保持 UTF-8 无 BOM。
