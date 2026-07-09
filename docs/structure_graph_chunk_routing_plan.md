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

对应实现方式：

- 在 `qa/generation/structure_graph.py` 新增 `build_lightweight_structure_graph`，把 `build_document_chunks` 产出的 flat chunk list 转成带 parent/child 和 prev/next 的只读结构图。
- 在 `qa/text_to_qa_pipeline.py::process_text_to_qa_one_step` 的 `build_document_chunks` 之后调用结构图构建函数，不改 OCR 或 chunking 主流程。
- 在 `qa/generation/evidence_units.py::build_generation_unit` 内增加 `auto_merge_hits`，用同父节点命中覆盖率把多个 leaf evidence 提升成 section context。

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

对应实现方式：

- 在 enriched chunk meta 中写入 `structure_parent_id`、`structure_children_ids`、`prev_chunk_index`、`next_chunk_index`、`quality_score`、`quality_flags`。
- 在 `generation_chunk_details` 中输出 `route.effective_mode`、`route.reason_code`、`quality.action`、`quality.score`、`quality.flags`。
- 在 `retrieval_trace.auto_merge` 中输出 `merged_contexts`、`count_coverage`、`weighted_coverage`、`merged_chunk_indexes`，让前端和调试日志能解释每次合并。

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

这里不是单纯“字符匹配”。第一版质量门控应使用确定性特征，不调用 LLM；其中一部分是正则或词表匹配，一部分是结构字段、文本统计和邻接关系计算。

建议实现成 `evaluate_chunk_quality`，每个维度都返回 flag 和 reason：

| 评分维度 | 实现方式 | 说明 |
| --- | --- | --- |
| 字符数过短且没有标题或列表结构 | `len(text.strip())` + `title_path` 是否为空 + 列表/条款正则 | 不是短就跳过；短句有标题、编号、数值、定义词时仍可保留 |
| 只有标题、页眉、页脚、目录项 | 标题/目录/页码正则 + 行数/标点密度 | 例如 `第.*章`、`目录`、`第 x 页`、连续点线目录项 |
| 表格残片只有列名或单位 | 分隔符密度、短 token 比例、单位/列名词表、事实句数量 | 表格内容有数值和主体时不能直接跳过 |
| 图片占位/附件占位 | 占位词正则 | 例如 `图片`、`图示`、`附件`、`image`、`figure`，但带说明文字时降级为 merge 而非 skip |
| 与前后 chunk 高重复 | 和 prev/next 做字符 n-gram 或 token Jaccard | 用于识别 OCR 页眉页脚、重复标题、跨页重复 |
| 标点/数字/符号占比异常 | 统计中文/英文/数字/标点/空白比例 | 只含符号、编号、单位时扣分；有完整句时保留 |
| 有明确标题路径 | 读取 `title_path`、`parent_index_path`、`level` | 结构字段加分，不靠正文匹配 |
| 有条件词、步骤词、定义词、数值、日期、主体 | 小词表/正则 + 实体样式正则 | 例如 `条件/流程/步骤/要求/定义/包括/应当`、日期、金额、电话、百分比 |
| 和同章节兄弟 chunk 互补 | 同 parent 下兄弟 chunk 的关键词差异和顺序关系 | 用 section sibling 信息判断是否适合并入 summary |

这两项分别解决两个问题：

- “有条件词、步骤词、定义词、数值、日期、主体”判断当前 chunk 是否有可问答事实或规则信号。它不是做 NER，也不是理解全文，只是用小词表和正则识别“这段有信息量”。
- “和同章节兄弟 chunk 互补”判断当前 chunk 是否只是一个章节中的局部片段。它不是判断内容好坏，而是判断“单独出 point 题是否太碎，是否更适合把同章节兄弟合起来走 summary”。

事实/规则信号第一版可这样实现：

```python
SIGNAL_TERMS = {
    "condition": ["条件", "若", "如果", "当", "满足", "符合", "仅限", "除非", "适用", "不适用"],
    "step": ["流程", "步骤", "首先", "然后", "提交", "申请", "审核", "办理", "确认", "完成"],
    "definition": ["是指", "定义为", "包括", "包含", "由", "组成", "分为", "分类", "以下简称"],
    "requirement": ["应当", "必须", "不得", "禁止", "需要", "要求", "标准", "范围"],
}

PATTERNS = {
    "date": r"\d{4}[年/-]\d{1,2}([月/-]\d{1,2}日?)?",
    "percent": r"\d+(\.\d+)?%",
    "money": r"\d+(\.\d+)?\s*(元|万元|亿元|人民币)",
    "phone": r"(1[3-9]\d{9}|400[- ]?\d{3}[- ]?\d{4}|\d{3,4}[- ]?\d{7,8})",
    "ratio": r"\d+[:：]\d+",
    "number_with_unit": r"\d+(\.\d+)?\s*(天|日|小时|分钟|个月|年|次|项|人|份|kg|m|㎡)",
    "subject_colon": r"^[\u4e00-\u9fa5A-Za-z0-9（）()《》]{2,30}[：:]",
}
```

实现要点：

- 词表命中要按类别计数，不要一个词重复刷分。
- 正则命中要返回 flag，例如 `has_date`、`has_money`、`has_requirement_term`。
- “主体”第一版不用复杂模型，可以来自三类信号：`title_path` 最后一段、行首 `主体：内容`、条款/列表项开头的名词短语。
- 短文本只要有日期、金额、电话、定义、强条件词，就不能直接按 `too_short_no_fact` 跳过。

这里说“不用复杂模型”主要是指第一版不做 NER。NER 是 Named Entity Recognition，即命名实体识别，用模型或规则从文本里抽取人名、机构名、地点、产品名等实体。当前目标只是判断 chunk 是否有可问答主体，不需要先引入 NER 模型。

伪代码：

```python
def extract_fact_signals(text: str, title_path: str) -> Dict[str, Any]:
    flags = []
    category_hits = {}
    for category, terms in SIGNAL_TERMS.items():
        hits = [term for term in terms if term in text]
        if hits:
            category_hits[category] = hits[:5]
            flags.append(f"has_{category}_term")

    for name, pattern in PATTERNS.items():
        if re.search(pattern, text):
            flags.append(f"has_{name}")

    if title_path:
        flags.append("has_title_subject")

    has_fact_signal = bool(flags)
    return {
        "has_fact_signal": has_fact_signal,
        "flags": flags,
        "category_hits": category_hits,
    }
```

同章节兄弟互补性第一版可这样实现：

```python
def sibling_complementarity(
    *,
    chunk: Dict[str, Any],
    siblings: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    current_tokens = set(_lexical_tokens(chunk["text"]))
    other_tokens = set()
    adjacent_tokens = set()
    current_index = int(chunk["chunk_index"])

    for sibling in siblings:
        sibling_index = int(sibling["chunk_index"])
        if sibling_index == current_index:
            continue
        tokens = set(_lexical_tokens(sibling["text"]))
        other_tokens |= tokens
        if abs(sibling_index - current_index) == 1:
            adjacent_tokens |= tokens

    overlap = len(current_tokens & other_tokens) / max(1, len(current_tokens | other_tokens))
    adjacent_overlap = len(current_tokens & adjacent_tokens) / max(1, len(current_tokens | adjacent_tokens))
    novelty = len(current_tokens - other_tokens) / max(1, len(current_tokens))

    return {
        "same_section_sibling_count": max(0, len(siblings) - 1),
        "section_overlap": overlap,
        "adjacent_overlap": adjacent_overlap,
        "novelty": novelty,
        "is_complementary": (
            len(siblings) >= 2
            and overlap >= 0.08
            and novelty >= 0.18
            and adjacent_overlap < 0.85
        ),
    }
```

判断逻辑：

- `overlap` 太低，说明当前 chunk 和同章节其他内容关系弱，不应强行合并。
- `novelty` 太低，说明它和兄弟 chunk 高重复，可能是页眉、重复标题或 OCR 重复，不算互补。
- `adjacent_overlap` 过高，说明和相邻 chunk 近似重复，应走 `near_duplicate` 或 `merge_with_neighbors`，不应给 summary 加分。
- 同章节有多个 chunk，且当前 chunk 与兄弟 chunk 有一定主题重叠但又提供新信息，才算互补。

例子：

```text
父章节：退费规则
chunk 10：退费申请条件
chunk 11：退费所需材料
chunk 12：退费办理流程
chunk 13：不予退费情形
```

这些 chunk 的标题/关键词有共同主题“退费”，但各自提供不同子项。它们互补，适合合并成 summary 上下文。

反例：

```text
chunk 10：某某公司内部资料 第 3 页
chunk 11：某某公司内部资料 第 4 页
chunk 12：某某公司内部资料 第 5 页
```

这些只是高重复页眉页脚，不算互补，应该被质量门控压低分。

第一版可先用轻量函数实现，不引入模型：

```python
def evaluate_chunk_quality(
    *,
    chunk: Dict[str, Any],
    prev_chunk: Optional[Dict[str, Any]],
    next_chunk: Optional[Dict[str, Any]],
    section_siblings: Sequence[Dict[str, Any]],
) -> ChunkQuality:
    score = 1.0
    flags = []
    reasons = []
    # regex/statistics/structure checks here
    return ChunkQuality(action=action, score=score, flags=flags, reasons=reasons)
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

`auto` 第一版不要把“结构词”或“单点事实词”当成硬标签。它们只是弱信号，不能单独决定走 `summary` 或 `point`。真正的路由原则是先判断这个 chunk 是否值得单独生成，再判断它是否必须和同章节兄弟一起看。

决策顺序：

1. 先看质量门控。
   - `quality.action == "skip"`：直接 `skip`。
   - `quality.action == "merge_with_neighbors"`：不单独生成；如果同章节 summary plan 覆盖它，则 `skip`，否则交给相邻 chunk 或 parent context。
2. 再看用户强制模式。
   - `qa_detail_mode=point`：所有 keep chunk 走 `point`。
   - `qa_detail_mode=summary`：所有 keep chunk 走 `summary`。
3. 只有 `qa_detail_mode=auto` 时才自动判断。

`auto` 下的强约束：

- 默认走 `point`。除非 summary 置信度足够高，否则不升级到 summary。
- 结构词、条件词、数值、日期、主体都只是加分项，不是最终标签。
- chunk 同时有结构词和单点事实时，先按 `point` 处理，除非它明显属于一个同章节组合。

走 `summary` 必须同时满足：

- 同一 `parent_index_path` 下至少有 `summary_route_min_children` 个有效子 chunk，默认 2。
- 当前 chunk 与同父兄弟 chunk 共享章节主题，但内容不是高度重复。
- 同父兄弟 chunk 至少覆盖两个不同子项，例如条件/材料/流程/时限，或定义/分类/适用范围。
- 拼出的 section context 不超过 `max_unit_chars`，否则只选核心兄弟 chunk。
- `section_bundle_score` 达到阈值，建议默认 `>= 0.65`。

`section_bundle_score` 可以这样算：

```text
section_bundle_score =
  0.30 * sibling_count_ok
  + 0.25 * sibling_complementarity
  + 0.20 * structure_pattern_score
  + 0.15 * context_budget_ok
  + 0.10 * non_duplicate_score
```

走 `point` 的原则：

- 当前 chunk 自包含，有明确事实、定义、数值、日期、电话、地址、主体属性。
- 或者同章节兄弟数量不足。
- 或者同章节兄弟虽然存在，但互补性不够、重复度太高、section bundle score 不足。
- 或者 summary context 过长，不能稳定塞进生成单元。

直观理解：

- `point` 像从一个句子或一小段里问“客服电话是多少”“办理时限是几天”“某概念定义是什么”。
- `summary` 像把同一章节里的多个小段合起来问“退费规则包括哪些条件、材料和流程”。
- 如果系统拿不准，就选 `point`，因为 point 对证据范围要求更小，更不容易漂移。

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
