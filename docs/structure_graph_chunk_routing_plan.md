# 轻量结构图、Generation Unit 与自动问答粒度改造方案

## 目标

这次改造不是要把 QA Flow 改成完整知识图谱系统，也不是靠一组关键词硬猜 `point` 或 `summary`。更合适的目标是借鉴 LlamaIndex、Haystack、LangChain 都在用的 small-to-big 思路：

```text
小块负责精确命中
父级章节负责提供完整上下文
命中到多个同父子块时，自动合并成父级上下文
出题时不再直接按 chunk，而是按 generation unit
```

落到 QA Flow 上，就是在现有 chunk 切分之后，先用 `title_path`、`index_path`、`parent_index_path`、`chunk_index` 组织一个轻量结构图，再把平铺 chunk 转成适合出题的 generation unit。后续候选问题生成、问题检索、答案提示词构建都围绕 generation unit 运行。

## 参考依据

本方案参考的是本地已经下载并阅读过的成熟实现，不再使用凭空设计的规则。

### LlamaIndex

本地参考路径：

- `external_repos/llama_index/llama-index-core/llama_index/core/node_parser/relational/hierarchical.py`
- `external_repos/llama_index/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py`

关键实现方式：

- `HierarchicalNodeParser` 把文档切成多层节点，默认类似 `[2048, 512, 128]` 这种父块、子块、叶子块结构。
- 节点之间保留 `parent/child` 关系，也可以保留 `prev/next` 相邻关系。
- 返回结果仍然可以是一个 flat node list，只是每个节点带关系元数据。
- `AutoMergingRetriever` 先从向量库召回小节点，再检查这些小节点是否属于同一个父节点。
- 如果命中的子节点数 / 父节点子节点总数超过阈值，默认阈值是 `0.5`，就删除子节点，加入父节点。
- 它还会用 `prev/next` 补齐中间缺失节点，然后递归尝试继续向上合并。

对 QA Flow 的启示：

```text
flat list 不等于没有层级。
只要每个 chunk 有 parent、prev、next 元数据，就可以在运行时构造层级。
自动合并不应该靠关键词判断，而应该靠“同父子节点覆盖率”判断。
```

### Haystack

本地参考路径：

- `external_repos/haystack/haystack/components/preprocessors/hierarchical_document_splitter.py`
- `external_repos/haystack/haystack/components/retrievers/auto_merging_retriever.py`

关键实现方式：

- `HierarchicalDocumentSplitter` 把原始文档切成 root、parent、leaf 多层文档。
- 每个文档用元数据保存 `__parent_id`、`__children_ids`、`__level`、`__block_size`。
- `AutoMergingRetriever` 假设已经有层级结构，输入是被普通 retriever 命中的 leaf documents。
- 它按 parent 分组，计算：

```text
merge_score = matched_child_count / parent_children_count
```

- 如果 `merge_score > threshold`，默认阈值也是 `0.5`，就返回 parent document，而不是零散 leaf documents。
- 如果 parent 继续有 parent，会递归向上合并。

对 QA Flow 的启示：

```text
同父节点覆盖率是成熟方案里的核心机制。
它不是语义标签，也不是人工关键词，而是一个结构比例。
QA Flow 当前的 parent_index_path 已经足够实现第一版。
```

### LangChain

本地参考路径：

- `external_repos/langchain_classic_wheel/langchain_classic/retrievers/parent_document_retriever.py`

说明：已尝试通过 `proxyon + gitproxyon` clone `https://github.com/langchain-ai/langchain.git`，`127.0.0.1:7897` 代理恢复后能访问 GitHub，但 LangChain 全仓库浅克隆仍在 240 秒内超时。为了避免网络阻塞，当前使用 PyPI 下载的官方 `langchain-classic==1.0.8` wheel 源码作为参考。

关键实现方式：

- `ParentDocumentRetriever` 的目标是在“小块 embedding 更准”和“大块上下文更完整”之间折中。
- 写入时可以先用 `parent_splitter` 得到父文档，再用 `child_splitter` 得到子文档。
- 子文档进入 vectorstore，父文档进入 docstore。
- 检索时先召回小子块，再通过子块 metadata 里的 parent id 找回父文档。

对 QA Flow 的启示：

```text
embedding / retrieval 用小块。
最终构建提示词时可以返回更大的父级上下文。
这和 QA Flow 当前“候选问题生成后再检索同文档证据”的流程不冲突。
```

## QA Flow 当前基础

当前 chunk 生成阶段已经提供了足够的结构字段。

来源代码：

- `qa/chunking/easy_dataset.py`
- `qa/generation/evidence_units.py::build_document_chunks`

当前每个 chunk 已有字段：

```text
chunk_id
chunk_index
index_path
title_path
parent_index_path
root_index_path
level
text
text_for_embedding
retrieval_text
path_summary
split_type
```

这些字段足够支撑第一版轻量结构图，不需要新增数据库，也不需要先做完整知识图谱。

当前答案证据检索也已经有结构意识：

- `qa/generation/evidence_units.py::QADocumentEvidenceIndex._rank_with_query_embedding`

已有结构分信号：

```text
same_parent
adjacent
title_overlap
```

所以第一版改造不是从零开始，而是把已有结构信息正式前置到“出题单位规划”和“证据合并”里。

## 为什么不是继续按 chunk 出题

现在的一步式流程可以简化理解为：

```text
document_chunks
-> 每个 chunk 调一次候选问题 LLM
-> 根据候选问题检索证据 chunk
-> 构建答案 generation unit
-> 生成最终 QA
```

问题在于，chunk 只是切分结果，不一定等于合理出题单位：

- 一个短 chunk 可能只是某个章节的一部分，单独看不完整。
- 多个兄弟 chunk 合起来才是完整流程、条件、分类或清单。
- 一个长 chunk 内部可能已经包含多个自然段、条款或列表，适合总结题。
- 有些 chunk 只是标题、目录、页眉页脚、表格残片，不值得单独出题。

因此要从：

```text
QA per chunk
```

改成：

```text
QA per generation unit
```

外部 API 第一版可以继续保留 `qa_per_chunk` 这个字段，避免前端和调用方大改。内部解释为“按原 chunk 数估算整篇文档目标题量”，再把题量分配给 generation units。

```text
target_total_qa = qa_per_chunk * 原有效 leaf chunk 数
generation_units = planner(document_chunks)
qa_budget 按 unit 分配
```

这样第一版不需要马上把接口字段改成 `qa_per_generation_unit`，但真实执行单位已经从 chunk 迁移到 generation unit。

## Generation Unit 是什么

generation unit 是 QA Flow 交给候选问题生成器的最小工作单元。它不是新的文档切分产物，而是从现有 chunk 运行时组织出来的出题上下文。

建议数据形态：

```python
{
    "unit_id": "section:1.2",
    "unit_type": "leaf | section | virtual_parent",
    "qa_mode": "point | summary",
    "anchor_chunk_index": 3,
    "source_chunk_indexes": [3, 4, 5],
    "parent_index_path": "1.2",
    "title_path": "文档标题 > 章节标题",
    "unit_text": "...",
    "child_count": 3,
    "usable_child_count": 3,
    "quality_child_coverage": 1.0,
    "qa_budget": 2,
    "debug": {
        "selection_reason": "parent_group_covered",
        "skipped_chunk_indexes": []
    }
}
```

第一版只需要三类 unit。

### leaf unit

来源是单个质量合格 chunk。

适用场景：

- chunk 内容自包含。
- 不需要兄弟 chunk 才能问清楚。
- 用于生成单点事实题。

对应模式：

```text
qa_mode = point
```

### section unit

来源是同一个 `parent_index_path` 下的多个兄弟 chunk。

适用场景：

- 同父节点下有多个质量合格子 chunk。
- 合并后长度没有超过答案生成上下文预算。
- 这些子 chunk 是同一章节下的连续内容。

对应模式：

```text
qa_mode = summary
```

这里不需要判断“是不是流程、条件、清单”等标签。只要结构上是同一父章节下的多子块，并且质量覆盖率足够，就先把它当作 section unit。具体问题类型仍由候选问题 LLM 根据 unit 内容生成。

### virtual_parent unit

来源是单个很长 chunk 内部再轻量切出的虚拟子段。

适用场景：

- 当前 chunk 已经很长。
- chunk 内部包含多个自然段、编号项、列表项、条款或小标题。
- 由于上游切分没有把它拆成多个兄弟 chunk，所以需要在运行时模拟 parent/child。

对应模式：

```text
qa_mode = summary
```

这个设计来自 LlamaIndex / LangChain 的多粒度切分思想：父块负责上下文，子块负责定位。QA Flow 第一版不必真的把虚拟子段写回 chunk list，只在 planner 内部用于判断这个长 chunk 是否应该作为 summary unit。

## 从 chunk 到 generation unit 的具体流程

这是决定效果的核心步骤。

### 第 1 步：规范化 chunk

输入来自：

```text
build_document_chunks(pre_split_chunks, pre_split_chunk_meta)
```

每个 chunk 补齐这些运行时字段：

```text
prev_chunk_index
next_chunk_index
siblings_by_parent
text_char_count
line_count
normalized_text
```

其中：

- `prev_chunk_index` / `next_chunk_index` 来自 `chunk_index` 顺序。
- `siblings_by_parent` 来自 `parent_index_path` 分组。
- `normalized_text` 只用于重复率、占位符、符号比例等轻量判断，不改变原文。

### 第 2 步：构建轻量结构图

建议新增模块：

```text
qa/generation/structure_units.py
```

核心函数：

```python
def build_structure_graph(document_chunks: list[dict]) -> StructureGraph:
    ...
```

图里只保留轻量边：

```text
parent -> children
child -> parent
prev -> next
same_section
```

embedding 相似边第一版不参与 generation unit 划分，只用于后面的证据检索和调试解释。原因是：出题单位应该主要由文档结构决定，embedding 相似更适合回答“这个问题还需要哪些补充证据”。

### 第 3 步：chunk 质量门控

质量门控不是为了判断 `point` 或 `summary`，而是为了避免明显低价值内容直接消耗 LLM 调用。

它也不是纯字符匹配。第一版建议使用确定性特征组合：

```text
长度、行数、标题路径是否存在
是否只有标题/目录/页眉/页脚
是否只有表格列名或单位
是否只有图片/附件/占位符
标点、数字、符号比例是否异常
是否和前后 chunk 高重复
是否有至少一个可问答文本句或条款
```

输出不要做复杂分类，只给三种状态：

```text
usable       可以作为 unit 的来源
context_only 不单独出题，但可作为同章节上下文
drop         不进入出题和证据上下文
```

建议函数：

```python
def evaluate_chunk_quality(chunk: dict, graph: StructureGraph) -> ChunkQuality:
    ...
```

这里可以保留分数，但分数只用于排序和调试，不作为难以解释的黑盒标签。

### 第 4 步：生成候选 units

先生成候选，不急着最终选择。

候选 1：leaf unit

```text
每个 usable chunk -> 一个 leaf unit 候选
```

候选 2：section unit

```text
按 parent_index_path 分组
组内 usable + context_only children >= 2
usable_child_count / total_child_count > 0.5
合并后的 unit_text 不超过 summary_unit_max_chars
-> 一个 section unit 候选
```

这里的：

```text
usable_child_count / total_child_count > 0.5
```

就是借鉴 LlamaIndex / Haystack 的同父子节点覆盖率。区别是：在出题规划阶段，它表示“这个父章节下有超过一半子块值得使用”；在答案检索阶段，它表示“某个问题命中了超过一半同父子块，应该返回父级上下文”。

候选 3：virtual_parent unit

```text
单个 chunk 的长度 >= 2 * 当前 chunk_size
且可以按自然段/编号/列表切成 >= 2 个虚拟子段
且虚拟子段合并后不超过 summary_unit_max_chars
-> 一个 virtual_parent unit 候选
```

如果运行时拿不到 `chunk_size`，第一版可以用后端默认 chunk_size 作为基准。这里的阈值来自多粒度切分比例，而不是为了给内容打语义标签。

### 第 5 步：选择最终 units

选择顺序：

```text
section unit
-> virtual_parent unit
-> leaf unit
```

原因是 section / virtual_parent 会覆盖多个子块或子段，如果先把所有 chunk 都变成 leaf unit，后续很容易重复出题。

覆盖规则：

```text
被 section unit 覆盖的 chunk，不再默认生成 leaf unit。
被 virtual_parent unit 覆盖的原 chunk，不再生成 leaf unit。
未被覆盖的 usable chunk，生成 leaf unit。
context_only 和 drop chunk 不生成 leaf unit。
```

第一版不要引入“例外重开”规则，否则很容易回到拍脑袋式判断。宁可让目标题量是软目标，也不要为了凑数生成重复问题。

### 第 6 步：分配 qa_budget

兼容现有 `qa_per_chunk`：

```text
target_total_qa = qa_per_chunk * usable_leaf_chunk_count
```

预算分配建议：

```text
leaf unit 默认 1 题
section unit 默认 min(2, child_count) 题
virtual_parent unit 默认 2 题
如果总预算不足，优先保留 section / virtual_parent 的 1 题，再分配 leaf
如果总预算有剩余，按 unit 文本长度和 child_count 追加，但不强行填满
```

这里的原则是：预算服务于覆盖，不服务于凑数。

### 第 7 步：确定 qa_mode

`qa_detail_mode` 第一版扩展为：

```text
point
summary
auto
```

当用户显式选择：

```text
point   -> 所有 unit 尽量按 point 生成，但 section/virtual_parent 会被压缩为更具体的问题
summary -> 所有 unit 尽量按 summary 生成，leaf unit 也允许生成小总结
auto    -> 由 unit_type 决定
```

`auto` 的规则必须简单、可解释：

```text
leaf unit           -> point
section unit        -> summary
virtual_parent unit -> summary
```

这比“标题里有某某词就 summary”稳定得多，也和成熟实现的层级节点思想一致。

## 答案证据阶段怎么改

用户后续是“基于问题进行检索”，这个方案不冲突。出题单位和答题检索是两层。

```text
generation unit 决定候选问题从哪里来
question retrieval 决定答案还需要哪些证据
```

当前已有：

- `QADocumentEvidenceIndex.retrieve_many`
- `QADocumentEvidenceIndex.build_generation_unit`

建议改成：

```text
source generation unit
-> 候选问题
-> 用候选问题检索同文档 leaf chunks
-> 对召回 hits 做 parent auto-merge
-> source unit chunks 与 retrieved evidence 去重
-> 构建最终答案提示词
```

### parent auto-merge

新增方法：

```python
def auto_merge_hits(
    hits: list[EvidenceHit],
    *,
    threshold: float = 0.5,
    max_context_chars: int,
) -> list[EvidenceContext]:
    ...
```

计算方式：

```text
按 parent_index_path 分组 retrieved hits
matched_child_count = 命中的同父 child 数
parent_children_count = 该 parent 下可用 child 总数
coverage = matched_child_count / parent_children_count

if coverage > threshold:
    用 parent section context 替代这些 children
else:
    保留 children
```

这就是 LlamaIndex / Haystack 的核心机制在 QA Flow 里的对应实现。

### source unit 和 evidence 去重

如果 source unit 已经包含 chunk 3、4、5，检索结果里又命中了 chunk 4，就不要在提示词里重复放两遍。

去重顺序：

```text
source unit chunks 优先
parent auto-merged context 其次
其他 semantic hits 最后
```

如果 parent context 覆盖了 source unit 已有 chunk，只渲染缺失部分，或者在 trace 里标记为已覆盖。

### answer_scope_policy 的含义保持不变

现有前端有：

```text
source_primary
same_section
cross_chunk
```

接入 generation unit 后：

- `source_primary` 表示答案提示词只使用 source generation unit，不再额外引入语义召回证据。
- `same_section` 允许同章节 auto-merge context。
- `cross_chunk` 允许跨章节 semantic hits，但仍要受 score、budget 和去重约束。

这不会破坏现有检索逻辑，只是 source 从单个 chunk 变成一个 unit。

## 后端改造点

### 1. 新增结构规划模块

新增：

```text
qa/generation/structure_units.py
```

建议公开函数：

```python
def build_structure_graph(document_chunks: list[dict]) -> StructureGraph:
    ...

def plan_generation_units(
    document_chunks: list[dict],
    *,
    qa_per_chunk: int,
    qa_detail_mode: str,
    max_unit_chars: int,
    chunk_size: int | None = None,
) -> list[GenerationUnit]:
    ...
```

这个模块只做轻量规划，不调用大模型，不访问 Milvus，不初始化重依赖。

### 2. 修改一步式主流程

修改：

```text
qa/text_to_qa_pipeline.py::process_text_to_qa_one_step
```

当前：

```text
raw_chunks
-> document_chunks
-> evidence_index
-> ThreadPoolExecutor per chunk
```

改为：

```text
raw_chunks
-> document_chunks
-> generation_units = plan_generation_units(document_chunks, runtime)
-> evidence_index
-> ThreadPoolExecutor per generation unit
```

进度事件里保留 `total_chunks`，新增：

```text
total_generation_units
completed_generation_units
unit_type
unit_source_chunk_indexes
```

这样前端仍能看到 chunk 维度，也能看到新 unit 维度。

### 3. 新增或改造 worker

当前 worker：

```text
qa/pipeline_runtime.py::run_one_step_chunk_worker
```

建议第一版新增：

```text
run_one_step_unit_worker
```

不要强行把旧函数改得过于复杂。新 worker 接收：

```python
unit: GenerationUnit
evidence_index: QADocumentEvidenceIndex
runtime: OneStepPipelineRuntime
```

调用候选问题 LLM 时：

```text
source_chunk_text -> unit.unit_text
source_chunk_meta -> unit.to_source_meta()
qa_detail_mode    -> unit.qa_mode when runtime.qa_detail_mode == "auto"
candidate_count   -> unit.qa_budget * candidate_multiplier
```

返回 item 时追加：

```text
qa_generation_unit_id
qa_generation_unit_type
qa_generation_unit_mode
unit_source_chunk_indexes
unit_parent_index_path
unit_quality_child_coverage
```

### 4. 扩展证据索引

修改：

```text
qa/generation/evidence_units.py::QADocumentEvidenceIndex
```

新增内部索引：

```text
_children_by_parent_index_path
_chunk_by_id
_chunk_by_index
```

新增能力：

```text
get_parent_children(parent_index_path)
render_source_unit(unit)
auto_merge_hits(...)
dedupe_contexts(...)
```

`build_generation_unit` 要支持 source 不再只是单个 chunk：

```python
def build_generation_unit(
    *,
    source_unit: GenerationUnit | None = None,
    source_chunk_index: int | None = None,
    ...
) -> dict:
    ...
```

为了降低改造风险，也可以第一版新增 `build_answer_unit_from_generation_unit`，旧函数先保留给兼容路径。

### 5. 扩展运行时配置

修改：

```text
qa/pipeline_runtime.py::parse_one_step_pipeline_runtime
```

新增或接受：

```text
qa_detail_mode = point | summary | auto
structure_units_enabled = true
structure_merge_threshold = 0.5
summary_unit_max_chars = DEFAULT_MAX_UNIT_CHARS
```

`structure_units_enabled` 第一版可以只在 `qa_detail_mode=auto` 时开启。等效果稳定后，再考虑让 point/summary 也走 unit planner。

### 6. 路由校验

修改：

```text
app/routers/pipeline_batch_routes.py
app/routers/pipeline_integrated_routes.py
```

当前只允许：

```text
point | summary
```

改为：

```text
point | summary | auto
```

如果请求没有传，默认仍保持当前行为：

```text
point
```

这样不会影响已有调用。

## 前端改造点

修改：

```text
static/index.html
static/app.js
```

### 问答粒度下拉框

当前：

```text
point
summary
```

改为：

```text
auto（按文档结构自动）
point（单点事实直答）
summary（总结/对比/推理）
```

推荐把 `auto` 放在第一项，但默认值是否切到 `auto` 要看是否希望立即改变现有行为。第一版稳妥做法：

```text
默认仍是 point
用户手动选 auto 后启用新 planner
```

### 调试展示

在已有 chunk debug 区域增加 generation unit 信息：

```text
unit_id
unit_type
qa_mode
source_chunk_indexes
parent_index_path
quality_child_coverage
qa_budget
selection_reason
```

这对后续分析 integrated task 很重要。否则用户只能看到最终 QA，看不到为什么某些 chunk 被合并或跳过。

### 高级参数

第一版不要暴露太多参数。建议只在高级区显示：

```text
结构化出题：启用/关闭
父节点合并阈值：默认 0.5
```

其他参数先后端默认，避免前端复杂化。

## 和现有评估规则的关系

这次改造会减少对以下硬规则的依赖：

```text
summary_question_not_grouped
summary_question_too_shallow_list
summary_source_fact_segment_not_grounded_in_chunk
summary_source_fact_not_grounded_in_chunk
```

原因是 summary 不再靠问题文字形态硬判，而是由 section / virtual_parent unit 提供真实多块上下文。评估时更应该检查：

```text
summary 问题是否来自 summary unit
summary source_fact 是否覆盖 unit 内多个可定位 evidence span
answer 是否只使用 source unit + approved evidence
retrieval trace 是否解释了 parent merge
```

也就是说，评估规则要从“看起来像不像 summary”转向“它是否真的基于多证据 unit 生成”。

## 推荐实施顺序

### 阶段 1：只加 planner 和 debug，不改变默认行为

实现：

```text
qa/generation/structure_units.py
plan_generation_units(...)
```

在 `process_text_to_qa_one_step` 中生成 unit plan，但默认 `qa_detail_mode=point` 时仍走旧 per-chunk 流程。debug 文件里输出 unit plan。

验收：

```text
同一批 integrated_document_task 能看到每个文档的 unit plan
不会影响现有 QA 输出
```

### 阶段 2：auto 模式切到 per generation unit

实现：

```text
qa_detail_mode=auto 时使用 run_one_step_unit_worker
point/summary 暂时仍走旧逻辑
```

验收：

```text
auto 模式下 chunk_completed 事件可兼容显示
新增 unit_completed 或 unit_debug 信息
输出 QA 带 qa_generation_unit_id
```

### 阶段 3：答案证据 parent auto-merge

实现：

```text
QADocumentEvidenceIndex.auto_merge_hits
source unit 与 retrieved evidence 去重
retrieval_trace 增加 auto_merge_trace
```

验收：

```text
同父多个命中 chunk 的问题，提示词里出现合并后的章节上下文
重复 chunk 不会在 source 和 evidence 中出现两遍
```

### 阶段 4：前端 auto 和调试面板

实现：

```text
qaDetailMode 增加 auto
请求序列化允许 auto
任务结果面板展示 generation unit debug
```

验收：

```text
前端可选择 auto
任务详情能解释每个问题来自哪个 unit
```

### 阶段 5：评估规则更新

实现：

```text
summary 相关规则改成基于 unit/evidence trace
弱化或移除只看问题形态的历史规则
```

验收：

```text
summary 模式不再因为浅层文字形态被误伤
证据不完整的问题仍能被定位
```

## 验证方案

### 单元级验证

用合成 chunks 测：

```text
同 parent 下 3 个 usable chunks -> 生成 1 个 section unit
同 parent 下 3 个 chunks 只有 1 个 usable -> 不生成 section unit
长 chunk 内有 3 个编号项 -> 生成 virtual_parent unit
标题/目录/占位符 chunk -> context_only 或 drop
被 section 覆盖的 leaf -> 不重复生成 leaf unit
```

### API 级验证

在 Docker runtime 中跑：

```bash
docker exec -it qa-flow-runtime bash
python -m compileall qa app
curl http://localhost:12000/test-connection
```

### 任务级验证

用之前的问题任务对比：

```text
integrated_document_task_1783416848
integrated_document_task_1783417492
```

重点看：

```text
总 LLM 调用次数是否下降
低价值 chunk 是否减少出题
summary 问题是否真的来自 section/virtual_parent unit
答案 source_fact 是否更容易定位到 unit 内证据
retrieval_trace 是否能解释为什么合并父章节
```

## 第一版边界

第一版不做：

```text
不做完整知识图谱数据库
不做 NER 实体图谱
不引入图数据库
不把 embedding 相似边作为 unit 划分主依据
不让关键词直接决定 point/summary
不强行保证 target_total_qa 一定填满
```

第一版要做：

```text
用现有 flat chunk list 派生 parent/child/prev/next
用轻量质量门控减少无效 chunk
用 generation unit 替代 per-chunk 出题
用 unit_type 决定 auto 下的 point/summary
用同父子节点覆盖率实现 parent auto-merge
保留问题级检索，并在提示词构建时去重
```

## 最终效果预期

这套方案会让 QA Flow 从“每个切出来的 chunk 都尝试出题”，升级为“先理解文档结构，再选择合适的出题单位”。短碎片不会单独浪费模型调用，同章节的多个相关 chunk 可以合并成总结题，长 chunk 也能按内部结构走 summary；答案阶段仍然保留问题级检索，并在多个同父证据命中时自动合并为更完整的章节上下文。整体预期是减少无效题和重复题，提高 summary 题的证据完整性，同时让每个问题为什么这么出、用了哪些证据更容易追踪。
