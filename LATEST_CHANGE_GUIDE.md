# Latest Change Guide

更新时间：2026-08-20（Asia/Shanghai）

## Objective

将 Summary 正式升级为“2–3 个原子子问题/证据跳”契约，解决
`integrated_document_task_1787210901` 中 Summary required material 绑定过重、
4/5 Summary 触发无意义全材料引用重试，以及单点事实被伪装成 Summary 的问题。

本轮不修改评估指标、平均分、过滤阈值或 Milvus schema。

## Effective Changes

- Summary planner 每条场景必须给出 2–3 个 `summary_hops`。每个 hop 绑定一个
  原子子问题、一份 SectionMaterial，以及 text/visual/mixed 证据和所需图片。
- Summary 的 required materials、required images 和整体 evidence mode 全部由
  hops 派生；optional 材料不承担 hop，也不参与答案覆盖失败。
- 问题生成器和编辑器只看到可读的原子信息缺口，把它们写成一句自然总括问题；
  不暴露 hop/material/image 内部 ID，也不新增 LLM 调用。
- 答案模型使用 `HOP-1..HOP-3` 标记证据关系。后端按每个 hop 的材料和模态要求
  校验 `evidence_usage.hop_refs`，失败原因为
  `incomplete_summary_hop_coverage`，不再盲目要求引用所有 Summary 材料。
- 补充证据只能支撑当前题目，不能顺带回答相邻但未被询问的知识点。
- consolidated JSON、SQLite 调试记录、QA 查询接口和调试页面保留
  `qa_generation_summary_hops` 与 `hop_refs`；Milvus v2 集合保持不变。
- planner 批次详情展示每个 hop 的原子子问题、证据模态和映射后的材料路径。

## Expected Behavior

- 只有能拆成 2–3 个真实原子需求的场景才能成为 Summary；单一事实自动让位给
  Point，不强行凑 Summary 数量。
- 一份材料可以支持多个不同 hop；多份材料也只有承担 hop 时才成为 required。
- 对外仍生成一条自然问题和一条整合答案，不把子问题拆成多条最终 QA。
- mixed/visual hop 必须实际引用对应图片证据，文本 hop 必须引用对应正文证据。
- 新契约预计减少上轮 4 次 `incomplete_primary_material_coverage` 类无意义重试；
  最终耗时仍需用下一次真实任务验证。

## Validation

```bash
docker exec qa-flow-runtime bash -lc 'cd /app && python -m compileall -q app qa scripts tests'
docker exec qa-flow-runtime bash -lc 'cd /app && python -m unittest discover -s tests -v'
docker exec qa-flow-runtime bash -lc 'cd /app && python -c "import app.main, qa.generation, qa.augmentation"'
bash -ic 'node --check static/app.js && node --check static/admin.js && node --check static/app_render.js && node --check static/app_query.js'
curl -fsS http://localhost:12000/health
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/milvus-status
curl -fsS http://localhost:11169/health
```

修改文件必须为 UTF-8 无 BOM。`AGENTS.md` 的既有本地修改属于用户，不纳入提交。
