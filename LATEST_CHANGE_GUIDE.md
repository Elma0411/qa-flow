# Latest Change Guide

更新时间：2026-08-21（Asia/Shanghai）

## Objective

根据 `integrated_document_task_1787248428` 修正复合 visual Summary 被误删、
Summary HOP 非原子与交叉引用、补充证据跨业务阶段污染答案，以及宽泛
`source_fact_text` 导致不同 Point 被误去重的问题。

本轮不修改评估指标、过滤阈值或 Milvus schema，也不增加 LLM 调用阶段。

## Effective Changes

- 场景级去重按完整契约执行：Point 只与 Point 比较，Summary 只在完整 HOP 集合
  等价时去重。Point 与 Summary 共享图片或单个 HOP 时均保留。
- visual promotion 只作用于 Point，不会用一个共享视觉事实替换复合 Summary。
- Summary 每个 HOP 只能表达一个独立信息缺口，最多依赖两张图片；不同 required
  材料若内容相同或 token 重合达到 90%，场景无效；optional material 最多一份。
- Summary 问题生成器和编辑器将 HOP 压缩为一句上位总括问题。无法自然合并时返回
  空问题并明确丢弃，不追加编辑器调用。
- `hop_refs` 按 HOP 的可用 evidence ref 白名单恢复，模型返回的跨 HOP 引用不会持久化。
- 检索链路升级为 `bm25_dense_rrf_bge_structure_scope_v3`。补充证据只能来自 required
  source 的同 section 或结构祖先/后代；同父节点的业务兄弟 section 即使 BGE 高分也拒绝。
- 仅引用补充证据而未引用绑定主材料时，返回 `primary_evidence_mismatch`，不重试答案。
- 最终 QA 去重同时要求问题意图和答案结论等价；`source_fact_text` 只保留为审计信息，
  不再凭包含关系单独删除问题。
- 前端构建标识更新为 `2026-08-21-3`，展示新的检索链路名称。

## Validation

```bash
docker exec qa-flow-runtime bash -lc 'cd /app && python -m compileall -q app qa scripts tests'
docker exec qa-flow-runtime bash -lc 'cd /app && python -m unittest discover -s tests -v'
bash -ic 'node --check static/app.js && node --check static/admin.js && node --check static/eval.js && node --check static/ui.js'
curl -fsS http://localhost:12000/health
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/milvus-status
curl -fsS http://localhost:11169/health
```

修改文件必须为 UTF-8 无 BOM。用户现有的 `AGENTS.md` 修改不纳入本次提交。
