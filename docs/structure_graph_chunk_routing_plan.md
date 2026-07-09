# 轻量结构图、chunk 质量门控与 point/summary 路由落地方案

## 目标

本方案要把 LlamaIndex/Haystack 中成熟的“层级节点 + prev/next + parent/child + auto merge/routing”思想落到 QA Flow 当前代码上。目标不是做重型知识图谱，也不是新开一条用户可见流水线，而是在现有一步式 QA 生成链路中增加一个轻量结构层：

- 用标题路径、chunk 顺序、章节层级、parent/child、prev/next 组织文档。
- 在候选问题 LLM 调用前做 chunk 质量门控，减少低价值 chunk 浪费调用。
- 在 `qa_detail_mode=auto` 时按 chunk/section 特征自动路由到 `point` 或 `summary`。
- 在 evidence 构建阶段做 auto merge，把同一父章节下多个命中的子 chunk 合并为更完整的章节上下文。
- 前端只增加现有配置面板里的选项和调试指标，不新增独立页面或独立 pipeline。

非目标：

- 不抽取实体、关系、三元组，不引入 GraphRAG 级别的全局知识图谱。
- 不要求新建 Milvus collection 或变更向量库 schema。
- 不替换当前 OCR、chunking、evaluation、Milvus 写入流程。
- 不把 summary 规则继续写成硬过滤，而是改成可解释的路由与质量决策。

## 参考实现对应关系

### LlamaIndex

参考文件：

- `external_repos/llama_index/llama-index-core/llama_index/core/node_parser/relational/hierarchical.py`
- `external_repos/llama_index/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py`

关键点：

- `HierarchicalNodeParser` 会把同一文档拆成多层节点，例如 `2048 -> 512 -> 128`。
- `_add_parent_child_relationship` 同时给父节点写 child 关系、给子节点写 parent 关系。
- `SentenceSplitter(..., include_prev_next_rel=True)` 会保留相邻节点关系。
- `AutoMergingRetriever._fill_in_nodes` 会在两个命中节点之间补齐中间节点。
- `AutoMergingRetriever._get_parents_and_merge` 会按父节点分组，计算 `命中的子节点数 / 父节点总子节点数`。超过阈值后删除子节点，加入父节点。
- `_retrieve` 里会循环 `_try_merging`，直到不能继续向上合并。

对 QA Flow 的启发：

- 当前不用真的做多粒度重切分，可以先把已有 `pre_split_chunks` 和 `pre_split_chunk_meta` 组织成层级结构。
- leaf chunk 继续作为精确生成和检索单位。
- section 或 parent 节点作为合并上下文单位。
- auto merge 的核心阈值可以直接采用“同父节点命中覆盖率”。

### Haystack

参考文件：

- `external_repos/haystack/haystack/components/preprocessors/hierarchical_document_splitter.py`
- `external_repos/haystack/haystack/components/retrievers/auto_merging_retriever.py`
- `external_repos/haystack/haystack/components/routers/conditional_router.py`

关键点：

- `HierarchicalDocumentSplitter` 给每个文档节点写入 `__parent_id`、`__children_ids`、`__level`、`__block_size`。
- `AutoMergingRetriever` 要求 leaf 文档必须有 `__parent_id`、`__level`、`__block_size`。
- Haystack 的 auto merge 同样按 parent 分组，`len(child_docs) / len(parent.children_ids) > threshold` 时返回 parent 文档。
- `ConditionalRouter` 用条件表达式把输入分到不同输出分支，证明 routing 应该是显式决策，而不是所有内容都走同一个 prompt。

对 QA Flow 的启发：

- 结构图字段必须显式、可序列化、可调试。
- 路由结果要进入 debug/progress，让我们知道一个 chunk 为什么走 `point`、`summary` 或 `skip`。
- auto merge 不是“多取几个 chunk”，而是把多个命中 leaf 提升为 parent/section 级上下文。

## 当前 QA Flow 代码事实

### 主链路

当前一步式生成入口在 `qa/text_to_qa_pipeline.py`：

```text
process_text_to_qa_one_step
-> parse_one_step_pipeline_runtime
-> resolve_one_step_chunks
-> build_document_chunks
-> QADocumentEvidenceIndex.build
-> ThreadPoolExecutor 提交 run_one_step_chunk_worker
```

当前关键事实：

- `build_document_chunks` 已经接收 `pre_split_chunk_meta`。
- chunk meta 已支持 `chunk_id`、`chunk_index`、`index_path`、`title_path`、`parent_index_path`、`root_index_path`、`level`、`path_summary`、`split_type`。
- `QADocumentEvidenceIndex._rank_with_query_embedding` 已经有 dense、lexical、structure 三路分数。
- structure score 里已有 `same_parent`、`adjacent`、`title_overlap`。
- `build_generation_unit` 已经能根据 `answer_scope` 追加 same-section 或 related evidence。
- `run_one_step_chunk_worker` 在调用 `call_candidate_question_llm` 之前没有质量门控，也没有 route decision。
- `qa_detail_mode` 当前只有全局 `point|summary`，worker 内不会按 chunk 动态改变。

### 后端接口

当前两个主要入口都校验 `qa_detail_mode`：

- `app/routers/pipeline_batch_routes.py`
- `app/routers/pipeline_integrated_routes.py`

当前逻辑：

```python
qa_detail_mode = (qa_detail_mode or "point").strip().lower()
if qa_detail_mode not in ("point", "summary"):
    qa_detail_mode = "point"
```

因此要接入 auto，必须同时改两处为：

```python
if qa_detail_mode not in ("point", "summary", "auto"):
    qa_detail_mode = "auto"
```

### 前端

当前前端位于：

- `static/index.html`
- `static/app.js`

当前 UI 只有：

```html
<option value="point">point（单点事实直答）</option>
<option value="summary">summary（总结/对比/推理）</option>
```

当前 `static/app.js` 已经把 `qa_detail_mode`、`retrieval_mode`、`semantic_top_k`、`rerank_top_n`、`retrieval_structure_weight`、`answer_scope_policy` 传给后端。说明我们可以复用现有表单序列化逻辑，只需要增加少量新字段。

## 推荐总体设计

### 不新增独立模式

最佳方案是把它接入现有一步式生成：

```text
qa_detail_mode:
  auto     新默认，按结构与质量自动选择 point/summary/skip
  point    强制单点事实模式
  summary  强制总结模式
```

`auto` 是同一 pipeline 内的内部策略，不是单独页面、单独 endpoint 或单独任务类型。

原因：

- 当前 8B 和 integrated pipeline 已经承担完整文档处理、OCR、chunk、QA、evaluation、storage。
- 新增独立 pipeline 会复制大量配置和状态管理。
- 用户真正需要的是“更聪明的 chunk 处理”，不是多一个入口。
- A/B 测试可以通过 `qa_detail_mode=point|summary|auto` 和新增开关完成。

### 数据流

新数据流：

```text
raw_chunks + pre_split_chunk_meta
-> build_document_chunks
-> build_lightweight_structure_graph
-> apply_chunk_quality_gate
-> build QADocumentEvidenceIndex(enriched_chunks, graph)
-> chunk route decision
-> run_one_step_chunk_worker with effective mode/context
-> retrieve_many
-> auto_merge_hits
-> build_generation_unit
-> answer LLM
```

重点变化：

- evidence index 仍然对全部 chunk 建索引，避免被跳过的 chunk 无法作为其他 chunk 的证据。
- worker 只对通过门控的 chunk 执行候选问题 LLM。
- summary 路由时，候选问题 LLM 的 source text 应该是 section context，而不是单个碎片。
- answer 阶段通过 auto merge 追加 parent/section 级上下文，减少 summary 问题证据不完整。

## 新增模块设计

建议新增文件：

```text
qa/generation/structure_graph.py
```

放在 `qa/generation` 的原因：

- 结构图服务于 QA 生成和 evidence 构造。
- 当前 `QADocumentEvidenceIndex` 也在 `qa/generation/evidence_units.py`。
- 不需要新增有状态服务类，纯函数和轻量 dataclass 即可。

通过 `qa/generation/__init__.py` 暴露必要 facade，其他模块不要直接 import 内部实现。

### 核心数据结构

```python
@dataclass(frozen=True)
class ChunkQuality:
    action: str  # keep | skip | merge_with_neighbors
    score: float
    flags: List[str]
    reasons: List[str]


@dataclass(frozen=True)
class ChunkRouteDecision:
    effective_mode: str  # point | summary | skip
    reason_code: str
    reason: str
    source_chunk_index: int
    context_chunk_indexes: List[int]
    parent_id: str
    quality: ChunkQuality


@dataclass(frozen=True)
class StructureNode:
    node_id: str
    node_type: str  # chunk | section
    chunk_index: Optional[int]
    chunk_id: str
    parent_id: str
    children_ids: List[str]
    prev_id: str
    next_id: str
    title_path: str
    section_key: str
    level: int
    text: str
    text_for_embedding: str
    quality: Optional[ChunkQuality]


@dataclass
class StructureGraph:
    nodes: Dict[str, StructureNode]
    chunk_node_ids_by_index: Dict[int, str]
    section_node_ids_by_key: Dict[str, str]
```

字段映射：

| QA Flow 当前字段 | 结构图字段 | 说明 |
| --- | --- | --- |
| `chunk_id` | `node_id/chunk_id` | leaf 唯一 ID |
| `chunk_index` | `chunk_index` | 顺序与 prev/next 基础 |
| `title_path` | `title_path` | 标题层级 |
| `parent_index_path` | `section_key/parent_id` | 优先作为父章节 key |
| `level` | `level` | 文档层级 |
| `path_summary` | `text_for_embedding` 辅助 | 参与检索和路由 |

如果 `parent_index_path` 为空，用 `title_path` 去掉最后一级作为父章节 key；如果仍为空，用 `root_index_path` 或 `doc_root`。

### enriched chunk meta

`build_lightweight_structure_graph` 返回 `enriched_chunks`，给现有 chunk dict 增加：

```python
{
    "structure_node_id": "...",
    "structure_parent_id": "...",
    "structure_children_ids": [],
    "prev_chunk_index": 12,
    "next_chunk_index": 14,
    "section_key": "1.2",
    "section_sibling_count": 4,
    "section_sibling_position": 2,
    "quality_action": "keep",
    "quality_score": 0.86,
    "quality_flags": ["has_heading", "has_sentences"],
}
```

这样 `QADocumentEvidenceIndex` 可以无侵入读取新字段；旧字段仍保留。

## chunk 质量门控

### 门控位置

在 `qa/text_to_qa_pipeline.py`：

```text
document_chunks = build_document_chunks(...)
structure_graph, document_chunks = build_lightweight_structure_graph(document_chunks, runtime)
chunk_plans = plan_chunks_for_generation(document_chunks, structure_graph, runtime)
evidence_index = QADocumentEvidenceIndex.build(document_chunks, structure_graph=structure_graph)
```

注意顺序：

- 先建结构图和质量结果。
- evidence index 仍对全部 enriched chunks 建索引。
- `ThreadPoolExecutor` 只提交 `chunk_plans` 中 `effective_mode != "skip"` 的 chunk。

### 质量评分规则

第一版用确定性规则，不调用 LLM。

建议评分维度：

```text
base = 1.0
- 0.35: 字符数过短且没有标题或列表结构
- 0.30: 只有标题、页眉、页脚、目录项
- 0.25: 表格残片只有列名或单位，没有事实句
- 0.25: 图片占位/附件占位，没有可问答文本
- 0.20: 与前后 chunk 高重复
- 0.15: 标点/数字/符号占比异常
+ 0.10: 有明确标题路径
+ 0.10: 有条件词、步骤词、定义词、数值、日期、主体
+ 0.10: 和同章节兄弟 chunk 互补
```

建议 action：

```text
score >= 0.55       keep
0.35 <= score < .55 merge_with_neighbors
score < 0.35        skip
```

`merge_with_neighbors` 不是直接生成，而是：

- 如果前后 chunk 同章节且质量较高，把当前 chunk 作为 context 交给邻近 chunk 或 parent summary。
- 不单独启动候选问题 LLM。
- 保留在 evidence index 里，允许被检索命中。

### 典型 skip 原因

```text
empty_text
heading_only
toc_line
page_header_footer
image_placeholder_only
table_header_only
too_short_no_fact
near_duplicate
```

这些原因进入 progress/debug：

```json
{
  "chunk_index": 12,
  "skip_reason": "heading_only",
  "quality_score": 0.21,
  "quality_flags": ["heading_only", "too_short_no_fact"]
}
```

## point/summary 路由

### 路由入口

新增函数：

```python
def decide_chunk_route(
    *,
    chunk: Dict[str, Any],
    graph: StructureGraph,
    runtime: OneStepPipelineRuntime,
) -> ChunkRouteDecision:
    ...
```

`run_one_step_chunk_worker` 增加参数：

```python
route_decision: Optional[ChunkRouteDecision] = None
```

worker 内部使用：

```python
effective_qa_detail_mode = route_decision.effective_mode if route_decision else runtime.qa_detail_mode
effective_chunk_text = build_effective_source_text(chunk_text, route_decision, graph)
```

然后把 `effective_qa_detail_mode` 传给：

- `call_candidate_question_llm`
- `call_evidence_answer_llm`

### 路由规则

如果用户显式选择：

```text
qa_detail_mode=point   所有 keep chunk 走 point
qa_detail_mode=summary 所有 keep chunk 走 summary
qa_detail_mode=auto    执行自动路由
```

`auto` 第一版建议规则：

走 `summary`：

- 同一 section 下有至少 2 个 keep 或 mergeable 子 chunk。
- 标题或文本包含流程、步骤、条件、材料、范围、规则、标准、要求、对比、分类、组成、清单等结构词。
- 当前 chunk 是列表/条款/流程中的一部分，单独看不完整。
- section context 在 `max_unit_chars` 范围内能容纳主要兄弟 chunk。

走 `point`：

- chunk 内有明确单点事实、定义、数值、时间、电话、地址、主体属性。
- 当前 chunk 自包含，前后 chunk 不影响答案。
- section 兄弟数量少，或 auto merge 覆盖率不足。

走 `skip`：

- quality action 是 `skip`。
- quality action 是 `merge_with_neighbors` 且该 chunk 已被父 section 的 summary plan 覆盖。

### route decision 示例

```json
{
  "chunk_index": 18,
  "effective_mode": "summary",
  "reason_code": "section_list_or_process",
  "reason": "同章节下有 4 个有效子 chunk，标题包含流程/材料类结构词，单 chunk 不足以生成完整总结题。",
  "parent_id": "section:3.2",
  "context_chunk_indexes": [18, 19, 20, 21],
  "quality": {
    "action": "keep",
    "score": 0.82,
    "flags": ["has_title_path", "list_like", "same_section_siblings"]
  }
}
```

## summary source text 构造

当前 `call_candidate_question_llm` 只接收一个 `source_chunk_text`。summary 路由时不要继续传单个碎片，而要传 section context。

新增函数：

```python
def render_route_source_text(
    *,
    chunk: Dict[str, Any],
    route: ChunkRouteDecision,
    graph: StructureGraph,
    max_chars: int,
) -> str:
    ...
```

输出格式：

```text
【当前章节】
标题路径：...

【核心 chunk】
chunk_id：...
内容：...

【同章节补充】
chunk_id：...
内容：...

chunk_id：...
内容：...
```

预算策略：

- `point`：主 chunk + 必要的 prev/next 一小段。
- `summary`：同 parent 下按 chunk 顺序拼接 keep/mergeable 子 chunk。
- 先保留标题路径，再保留核心 chunk，再按顺序加入兄弟 chunk。
- 超预算时只截断补充 chunk，不截断标题和核心 chunk。

## auto merge evidence 设计

### 当前不足

`QADocumentEvidenceIndex.build_generation_unit` 当前逻辑是：

- source_primary 不追加 hits。
- same_section 只追加同 parent 或 adjacent 的命中 chunk。
- cross_chunk 可追加相关补充。

它还没有：

- 命中多个兄弟 chunk 后提升为 section context。
- 补齐两个命中 chunk 之间漏掉的中间 chunk。
- 把 merge 过程写入 trace。

### 新接口

`QADocumentEvidenceIndex` 增加可选 graph：

```python
class QADocumentEvidenceIndex:
    def __init__(self, chunks, embeddings, structure_graph=None):
        self.structure_graph = structure_graph
```

`build` 增加：

```python
@classmethod
def build(cls, chunks, structure_graph=None):
    ...
    return cls(chunks=chunks, embeddings=embeddings, structure_graph=structure_graph)
```

新增函数：

```python
def auto_merge_hits(
    *,
    source_chunk_index: int,
    hits: List[EvidenceHit],
    answer_scope: str,
    max_unit_chars: int,
    threshold: float,
) -> Tuple[List[MergedEvidenceContext], List[EvidenceHit], Dict[str, Any]]:
    ...
```

### merge 规则

1. Fill in adjacent gap

参考 LlamaIndex `_fill_in_nodes`：

```text
hit chunk 10 和 hit chunk 12 同 parent，中间 chunk 11 也是同 parent
-> 如果 chunk 11 质量不是 skip，并且预算允许，则补入 chunk 11
```

2. Parent coverage merge

参考 LlamaIndex/Haystack：

```text
coverage = 命中的有效子 chunk 数 / parent 下有效子 chunk 总数
if coverage > threshold:
    用 parent section context 替换这些子 chunk
```

建议默认：

```text
structure_auto_merge_threshold = 0.5
```

3. Weighted coverage

为了适配 QA Flow，不只看数量，还看质量：

```text
weighted_coverage =
    sum(hit.quality_score for hit in selected_children)
    / sum(child.quality_score for child in all_good_children)
```

第一版可以同时满足任一条件：

```text
count_coverage > threshold
or weighted_coverage > threshold
```

4. Scope 限制

```text
answer_scope=source_primary  不 auto merge
answer_scope=same_section    只允许同 parent merge
answer_scope=cross_chunk     允许同 parent merge + 高置信相关 section context
```

第一版建议只做 same-parent merge，cross-section 先不提升成父节点，避免证据漂移。

### generation unit 渲染

新增 role：

```text
merged_section_context
filled_adjacent_context
same_section_context
related_context
```

渲染顺序：

```text
【主来源块】
...

【自动合并章节上下文：标题路径】
merge_reason：matched_children_coverage
included_chunk_ids：...
内容：...

【同章节上下文】
...

【相关补充】
...
```

trace 增加：

```json
{
  "auto_merge": {
    "enabled": true,
    "threshold": 0.5,
    "merged_contexts": [
      {
        "parent_id": "section:3.2",
        "count_coverage": 0.67,
        "weighted_coverage": 0.71,
        "merged_chunk_indexes": [18, 19, 20],
        "reason_code": "parent_coverage_above_threshold"
      }
    ],
    "filled_adjacent_chunk_indexes": [19]
  }
}
```

## runtime 配置

`OneStepPipelineRuntime` 增加字段：

```python
structure_graph_enabled: bool
chunk_quality_gate_enabled: bool
structure_auto_merge_enabled: bool
structure_auto_merge_threshold: float
qa_auto_route_enabled: bool
summary_route_min_children: int
summary_route_min_chars: int
```

默认值建议：

```text
qa_detail_mode = auto
structure_graph_enabled = true
chunk_quality_gate_enabled = true
qa_auto_route_enabled = true
structure_auto_merge_enabled = true
structure_auto_merge_threshold = 0.5
summary_route_min_children = 2
summary_route_min_chars = 180
```

解析位置：

- `qa/pipeline_runtime.py::parse_one_step_pipeline_runtime`

## 后端接口改动

同步修改：

- `app/routers/pipeline_batch_routes.py`
- `app/routers/pipeline_integrated_routes.py`

### 表单参数

新增：

```python
structure_graph_enabled: bool = Form(True, description="是否启用轻量结构图")
chunk_quality_gate_enabled: bool = Form(True, description="是否启用 chunk 质量门控")
structure_auto_merge_enabled: bool = Form(True, description="是否启用结构 auto merge")
structure_auto_merge_threshold: Optional[float] = Form(None, description="auto merge 覆盖率阈值，默认 0.5")
```

`qa_detail_mode` 说明改为：

```text
auto=按结构和质量自动选择 point/summary，point=单点事实直答，summary=总结/对比/推理
```

校验：

```python
qa_detail_mode = (qa_detail_mode or "auto").strip().lower()
if qa_detail_mode not in ("auto", "point", "summary"):
    qa_detail_mode = "auto"
```

阈值：

```python
structure_auto_merge_threshold = max(
    0.05,
    min(0.95, float(structure_auto_merge_threshold if structure_auto_merge_threshold is not None else 0.5)),
)
```

### status/debug 输出

任务状态里的 `retrieval_config` 增加：

```json
{
  "qa_detail_mode": "auto",
  "structure_graph_enabled": true,
  "chunk_quality_gate_enabled": true,
  "structure_auto_merge_enabled": true,
  "structure_auto_merge_threshold": 0.5
}
```

`generation_chunk_details` 增加：

```json
{
  "chunk_index": 18,
  "route": {
    "effective_mode": "summary",
    "reason_code": "section_list_or_process"
  },
  "quality": {
    "action": "keep",
    "score": 0.82,
    "flags": ["list_like", "same_section_siblings"]
  }
}
```

## 前端改动

文件：

- `static/index.html`
- `static/app.js`

### UI

`qa_detail_mode` 增加 auto 并设为默认：

```html
<option value="auto" selected>auto（按结构自动选择）</option>
<option value="point">point（单点事实直答）</option>
<option value="summary">summary（总结/对比/推理）</option>
```

小字说明改为：

```text
auto 会根据 chunk 质量、章节结构、相邻关系和同章节覆盖度自动选择 point 或 summary。
```

在当前检索配置附近增加高级项：

```html
<label>
  <input type="checkbox" id="structureGraphEnabled" checked />
  启用轻量结构图（structure_graph_enabled）
</label>
<label>
  <input type="checkbox" id="chunkQualityGateEnabled" checked />
  启用 chunk 质量门控（chunk_quality_gate_enabled）
</label>
<label>
  <input type="checkbox" id="structureAutoMergeEnabled" checked />
  启用结构 auto merge（structure_auto_merge_enabled）
</label>
<label>
  auto merge 阈值（structure_auto_merge_threshold）
  <input type="number" id="structureAutoMergeThreshold" value="0.5" min="0.05" max="0.95" step="0.05" />
</label>
```

### 表单提交

`static/app.js` 在现有 `qa_detail_mode` 和 retrieval 参数附近追加：

```javascript
formData.append('structure_graph_enabled', structureGraphEnabled ? 'true' : 'false');
formData.append('chunk_quality_gate_enabled', chunkQualityGateEnabled ? 'true' : 'false');
formData.append('structure_auto_merge_enabled', structureAutoMergeEnabled ? 'true' : 'false');
if (String(structureAutoMergeThreshold).trim()) {
  formData.append('structure_auto_merge_threshold', String(structureAutoMergeThreshold).trim());
}
```

### 调试展示

当前 `appendTextMetric` 已展示 retrieval config。追加：

```text
QA 粒度策略
结构图
chunk 质量门控
auto merge
auto merge 阈值
```

chunk debug table 追加：

```text
effective_mode
route_reason_code
quality_action
quality_score
quality_flags
auto_merge_count
```

## 实施步骤

### 第 1 步：新增结构图模块

新增：

```text
qa/generation/structure_graph.py
```

实现：

- `build_lightweight_structure_graph(document_chunks, config)`.
- `evaluate_chunk_quality(chunk, prev_chunk, next_chunk, section_siblings)`.
- `plan_chunks_for_generation(document_chunks, graph, runtime)`.
- `render_route_source_text(chunk, route, graph, max_chars)`.

测试：

- parent/child 是否正确。
- prev/next 是否正确。
- 空 parent_index_path 时 section_key fallback 是否稳定。
- heading-only/table-header/image-placeholder 是否被标记。
- 同章节列表是否 route 到 summary。

### 第 2 步：接入主 pipeline

修改：

- `qa/text_to_qa_pipeline.py`
- `qa/pipeline_runtime.py`

要点：

- `parse_one_step_pipeline_runtime` 增加配置字段。
- `process_text_to_qa_one_step` 在 `build_document_chunks` 后建 graph。
- `ThreadPoolExecutor` 按 route plan 提交 worker。
- skipped chunk 也写入 progress/debug，不执行 LLM。
- `run_one_step_chunk_worker` 接收 route decision，使用 effective mode 和 effective source text。

### 第 3 步：接入 evidence auto merge

修改：

- `qa/generation/evidence_units.py`
- `qa/generation/__init__.py`

要点：

- `QADocumentEvidenceIndex` 接收 `structure_graph`。
- `build_generation_unit` 在 selected hits 后调用 auto merge。
- `_render_unit_text` 支持 `merged_section_context`。
- `retrieval_trace` 输出 auto merge 决策。

### 第 4 步：接口与前端

修改：

- `app/routers/pipeline_batch_routes.py`
- `app/routers/pipeline_integrated_routes.py`
- `static/index.html`
- `static/app.js`

要点：

- `qa_detail_mode` 支持 `auto`。
- 新增 4 个结构相关表单参数。
- 状态返回补充 retrieval/structure config。
- 前端展示 route/quality/merge 调试信息。

### 第 5 步：验证与 A/B

不要作为单独用户模式测试，而是作为同一 pipeline 的 A/B 配置测试：

```text
baseline:
  qa_detail_mode=point 或 summary
  structure_graph_enabled=false
  chunk_quality_gate_enabled=false
  structure_auto_merge_enabled=false

new:
  qa_detail_mode=auto
  structure_graph_enabled=true
  chunk_quality_gate_enabled=true
  structure_auto_merge_enabled=true
```

建议复用之前观察过的任务样本：

- `integrated_document_task_1783416848`
- `integrated_document_task_1783417492`

观察指标：

- `candidate_question_seconds` 是否下降。
- `valid_items / candidate_questions` 是否上升。
- `summary_question_not_grouped` 是否下降。
- `summary_question_too_shallow_list` 是否下降。
- `summary_source_fact_segment_not_grounded_in_chunk` 是否下降。
- `summary_source_fact_not_grounded_in_chunk` 是否下降。
- answer_scope_decision 中 `evidence_not_confident` 是否下降。
- 人工抽查 summary 问题是否更像流程、条件、组成、对比、清单。

## 验收标准

后端：

- `qa_detail_mode=auto` 能走通 batch 和 integrated 两个入口。
- Docker 内 `import app.main` 成功。
- `/test-connection` 成功。
- 小样本文档能生成 QA。
- chunk debug 中能看到 route 和 quality。
- retrieval trace 中能看到 auto merge 统计。

前端：

- UI 中有 auto 选项。
- 结构图、质量门控、auto merge 开关能正确提交。
- 任务详情能展示结构配置和 chunk route/quality 信息。

质量：

- 对明显标题、页眉、目录、表头残片不单独调用候选问题 LLM。
- 对流程/条件/材料/规则类 section，auto 能路由到 summary。
- 对电话、日期、定义、数值类单点事实，auto 能路由到 point。
- 同父章节多个 evidence 命中时，generation unit 中出现 merged section context。

## 风险与控制

### 风险：summary 上下文过大

控制：

- `render_route_source_text` 和 `build_generation_unit` 都必须受 `max_unit_chars` 限制。
- summary context 优先保留标题、核心 chunk、同章节高质量兄弟 chunk。

### 风险：auto merge 引入不相关上下文

控制：

- 第一版只允许 same-parent merge。
- merge 要同时看 count coverage 和 quality weighted coverage。
- `answer_scope=source_primary` 时禁止 merge。

### 风险：质量门控误跳过有价值短句

控制：

- 有数值、日期、电话、地址、定义词、条件词的短句不能仅因短而 skip。
- `merge_with_neighbors` 优先于 `skip`。
- skipped chunk 仍保留在 evidence index 中。

### 风险：并发 worker 和 graph 对象共享

控制：

- `StructureGraph` 构建后只读。
- dataclass 尽量 frozen。
- worker 只读取 route decision 和 graph 渲染后的 source text。

## 推荐最终形态

最终默认使用：

```text
qa_detail_mode=auto
structure_graph_enabled=true
chunk_quality_gate_enabled=true
structure_auto_merge_enabled=true
answer_scope_policy=same_section
retrieval_mode=hybrid
retrieval_structure_weight=0.08
structure_auto_merge_threshold=0.5
```

理由：

- `auto` 解决 point/summary 需要人工预选的问题。
- `same_section` 比 `cross_chunk` 更稳，能让 summary 获得足够上下文，又不容易漂移。
- hybrid + structure_weight 已经是当前代码支持的成熟路径。
- auto merge 只在 same-parent 内发生，和现有 `same_parent/adjacent/title_overlap` 逻辑一致。

这个方案的核心价值是把“文档结构”前置到生成决策中，而不是等 QA 生成后再用硬规则过滤。这样能减少无效 LLM 调用，也能让 summary 问题天然拿到完整章节证据。
