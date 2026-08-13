# Latest Change Guide

更新时间：2026-08-13（Asia/Shanghai）

## Objective

一次性修正章节/内容块混用和图片全局归属问题，并把答案证据检索替换为固定的成熟混合检索与真实 BGE 重排链路。

## Effective Changes

- 切块输出拆分章节结构、内容 chunk 和物理 fragment：新增 `section_*`、`fragment_*`、`content_kind`、`source_asset_ids` 字段，移除新输出中的 `is_leaf/index_path` 混合语义。
- 内部章节可以保存自身正文；空父章节只存在于结构树；长章节的 `Part N/M` 不再进入标题或章节路径。图片描述只归属实际包含该图片 marker 的 chunk。
- 检索固定为：问题规范化 -> 标准 BM25 + BGE-M3 dense -> RRF -> `chunk_id` 去重 -> 本地 `bge-reranker-v2-m3` 原子块精排 -> 结构窗口补全 -> 窗口二次 BGE 精排 -> 去重/token 预算。
- 同一 `fragment_group_id` 必须整体恢复；同章节只合并连续命中；单块仅在“上述/如下/分别”等明确依赖信号下补一个相邻上下文；多个子章节命中同一父章节时只组合真实命中块，并将“含父正文/不含父正文”作为互斥候选交给二次重排。同分时短窗口优先，重叠窗口和超预算窗口不入选。
- 候选题不再生成 `retrieval_query`、`must_have_terms`、`answer_scope_hint`。标准和一体化接口只公开 `final_evidence_k`（默认 5）与 `evidence_token_budget`（默认 4000）。
- 新 chunk 写入版本化 Milvus 集合 `doc_content_chunks_v2`。旧 `doc_tree_chunks` 保留但不回退查询；`POST /doc-chunks/rebuild` 接受原文并重新切块，完整校验和生成向量后才替换同一 `task_id + original_filename`，写入失败会恢复原记录。
- 文档块存储状态由进程级 `DocumentChunkStore` 管理，不再使用模块全局 Milvus client。
- BGE reranker 由进程级 `RerankerService` 延迟加载并复用。模型缺失、文件不完整、依赖缺失或设备不可用会明确失败，不会退回旧排序。

## Expected Behavior

- `section_is_leaf` 描述章节是否有子章节，不再描述图片或内容 chunk 是否“叶子”。
- 每个证据归因仍落在真实 `chunk_id`；Evidence Window 只是查询时的虚拟组合，不制造持久化假 ID。
- 工作台只显示最终证据窗口数和 token 预算，不再暴露检索模式、手工权重、轻量重排数或证据范围策略。
- 新任务自动写入 v2 集合；旧任务需通过重建接口提交原始正文后才会出现在新块查询中。

## Validation

```bash
python -m compileall -q app qa scripts
python -m unittest discover -s tests
node --check static/app.js
node --check static/app_query.js
git diff --check
```

Docker runtime verification:

```bash
docker exec qa-flow-runtime sh -lc 'cd /app && python -m compileall -q app qa scripts && python -m unittest discover -s tests'
docker exec qa-flow-runtime sh -lc 'cd /app && QA_RERANKER_DEVICE=cpu python -c "from qa.retrieval import get_reranker_service; print(get_reranker_service().rank(\"申请材料\", [(\"a\", \"申请材料包括身份证明\"), (\"b\", \"天气晴朗\")]))"'
curl http://localhost:12000/test-connection
```
