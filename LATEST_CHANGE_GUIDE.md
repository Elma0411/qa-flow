# Latest Change Guide

更新时间：2026-08-18（Asia/Shanghai）

## Objective

将 QA 生成改为“planner 决定证据语义，后续模型只负责语言表达”的稳定链路：

`SectionMaterial -> frozen ScenarioContract -> question writer -> wording editor -> retrieval -> evidence answer`

本轮同时删除难度字段和题型理由，清理模型提示词中的内部字段污染，并让图片成为可引用、可评估的独立证据。

## Effective Changes

- planner 是唯一能决定 Point/Summary、required/optional materials、
  `text|visual|mixed`、required images 和题型的阶段。映射为真实 ID 后的
  `ScenarioContract` 不可被候选题、编辑器或答案调用改写。
- Summary 最多绑定三份紧密相关的必需材料，避免“概括整份手册”式场景。
- 分类模板改为 planner 专用的简短读者画像；候选题和答案不再注入完整分类模板。
  few-shot 只作为可选、已审核、按 Point/Summary 匹配的一条风格示例；空配置不会
  再向模型发送 `null`。
- 候选题生成器输入为可读 writing brief，输出只有 `{"question":"..."}`。
  不再让模型输出题型理由、难度、材料 refs、图片 refs 或证据模式。
- 问题编辑器输入为原问题加 writing brief，输出也只有最终 `question`。它不再
  返回 keep/rewrite/drop，也不再改材料、图片或证据模式。
- 答案输入改为 `正文证据-N`、`图片证据-N` 和可选补充正文证据块。内部
  `typed_primary_materials`、真实 chunk/image ID、原始 evidence_mode 字段不再
  出现在模型输入。答案通过临时证据标签引用实际使用的正文/图片；后端恢复审计 ID。
- `visual` 答案必须引用所有 required 图片证据；`mixed` 必须同时引用正文和图片
  证据。评估可复用同一套已渲染证据，避免图片答案被误判为无依据。
- 已从生成、验证、JSON/CSV、调试、API、搜索、管理页和 QA Milvus schema 删除
  `difficulty_level`、`difficulty_score`、`question_type_reason`。
  QA 向量集合升级为固定 `qa_pairs_collection_v2`，旧集合不再被应用读取或写入。
- generation timing 新增每个阶段的 p50/p95；调试 unit 表展示“问题编辑”耗时，
  且 valid QA 为 0 的 unit 显示“未产出”。

## Operational Note

当前运行时已创建 `qa_pairs_collection_v2` 并删除旧 `qa_pairs_collection`；现有
QA Milvus 仅保留新版集合和独立的 `doc_content_chunks_v2` 文档集合。

## Validation

```bash
docker exec qa-flow-runtime bash -lc 'cd /app && python -m compileall -q app qa scripts tests'
docker exec qa-flow-runtime bash -lc 'cd /app && python -m unittest discover -s tests -v'
docker exec qa-flow-runtime bash -lc 'cd /app && python -c "import app.main, qa.generation, qa.augmentation"'
curl -fsS http://localhost:12000/health
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/milvus-status
```

修改文件必须保持 UTF-8 无 BOM。`AGENTS.md` 的既有本地修改仍属于用户，不纳入本轮提交。
