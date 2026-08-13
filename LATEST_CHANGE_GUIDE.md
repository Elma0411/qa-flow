# Latest Change Guide

更新时间：2026-08-14（Asia/Shanghai）

## Objective

把出题流程一次性收敛为“逻辑 section 材料 -> Point/Summary 场景 -> 分类出题 -> LLM 编辑 -> 相关证据检索 -> 真实来源回填”，同时移除旧 Milvus 文档块集合兼容。

## Effective Changes

- 同一 `section_path` 的正文、物理 fragment 和已接受图片描述合并为一个 `SectionMaterial`；不同 section 不再因为同属一章或同一父标题而自动合并。
- 新增 LLM 场景规划：`PointScenario` 绑定一份材料和一个事实需求；`SummaryScenario` 只绑定确实共同服务于一个读者需求的多个事实。长文档按内部字符预算分批，Summary 批次保留结构父邻域；`auto` 最终仍在文档级按 35% 目标分配，合格总结场景不足时额度回流给单点题。Point/Summary 返回值只进入对应类型池；同一 section 若包含多个不同事实，可形成多个不同 Point 场景。LLM 偶发少返回单点场景时，只用一材料一场景的确定性 Point fallback 补缺口，绝不伪造 Summary。
- 每个场景只生成一道题。Point/Summary 使用不同约束，随后统一经过一次 LLM `keep/rewrite/drop` 编辑；程序只校验 JSON、必填值、合法 ID/类型和精确重复，不用硬规则判定语言自然度。
- 后续 fragment 的重复 Markdown 标题在 `SectionMaterial` 中只保留一次；所有场景结束后再做一次忽略空白和标点的文档级问题去重。
- 答案实际引用的 `主材料-N` 决定来源。标量 `source_chunk_id/index/title_path` 指向第一条直接主证据，`source_chunk_ids/indexes/title_paths` 保存总结题的完整主证据集合；多材料总结题必须覆盖每份绑定材料，否则进入既有重试。生成结束后不再被锚点 chunk 覆盖。
- BGE 原子重排和窗口重排都增加相关性准入。最低原始 logit 为 `-1.0`；原子/窗口除头部相对分差外，还必须分别位于真实主材料得分下 `1.0/2.0` 以内。校准样本覆盖中英文相关、同主题硬负例和无关项，并有可执行脚本复现。`final_evidence_k` 只是上限，补充证据允许为 0。
- 文档块只使用固定集合 `doc_content_chunks_v2`（schema v2）。代码不再暴露旧集合常量或可配置的集合重定向；运行环境中的旧 `doc_tree_chunks` 已删除。

## Expected Behavior

```text
content chunks
-> SectionMaterial
-> PointScenario / SummaryScenario
-> global allocation
-> typed question generation
-> LLM keep/rewrite/drop editor
-> BM25 + dense + RRF + calibrated BGE admission
-> answer generation
-> actual primary-source attribution
```

- 同一个 section 可以同时参与意图不同的单点和总结场景。
- sibling section 只有被场景规划器显式绑定且共同服务于一个读者需求时，才进入同一总结题。
- 问题编辑后的文本才会进入检索和答案生成。
- 无真正相关的补充证据时，答案只使用场景主材料。

## Validation

```bash
python -m compileall -q app qa scripts
python -m unittest discover -s tests
QA_RERANKER_DEVICE=cpu python scripts/calibrate_bge_relevance.py --strict
git diff --check
```

Docker runtime verification:

```bash
docker exec qa-flow-runtime sh -lc 'cd /app && python -m compileall -q app qa scripts && python -m unittest discover -s tests'
curl http://localhost:12000/test-connection
```
