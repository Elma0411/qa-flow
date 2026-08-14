# QA Flow 问答生成治理交接文档

> 用途：把当前 QA Flow 的真实问题、设计目标、实现逻辑和验收标准交接给另一个 Codex/开发对话。
>
> 本文是需求与实现计划，不代表代码已经完成。接手者必须先检查当前工作区和实际代码，再判断哪些内容已经落地。

## 1. 当前项目到底在做什么

QA Flow 不是一个简单的“把每个 chunk 直接问答生成一遍”的程序，而是一条文档到问答对的质量控制流水线：

```text
文档上传 / OCR / 文本解析
→ 文档切块
→ 结构节点和材料整理
→ Point / Summary 场景规划
→ 候选问题生成
→ 问题编辑和指代清晰度检查
→ 同文档证据检索
→ 答案生成
→ 无参考评估
→ 平均分阈值过滤
→ 保存、检索和人工审核
```
图片理解属于前置能力。本次问答生成性能目标不包含 OCR 和图片理解耗时。

代码仓库：

```text
/data2/hjk/qa-flow
```

主要相关模块：

```text
qa/generation/structure_units.py       场景规划、SectionMaterial、GenerationUnit
qa/generation/qa_generation_flow.py    planner、候选题、编辑、答案、引用恢复
qa/generation/evidence_units.py        同文档检索和模型可读证据材料
qa/prompts/qa_generation_prompts.py    候选题、编辑、答案、planner 提示词
qa/pipeline_runtime.py                 单个 generation unit 的执行链路
qa/text_to_qa_pipeline.py              单文件一步式流水线
app/services/pipeline_execution/service.py  批量完整流水线
app/routers/pipeline_batch_routes.py   普通批量路由
app/routers/pipeline_integrated_routes.py OCR/图片/QA 集成路由
static/                                QA 生成、调试、评估和审核界面
```

## 2. 需求起点：图片中的问题

图片中提出了两组需求。

### 2.1 无参考评估展示

从现在开始，无参考评估的普通界面只展示：

```text
faithfulness
answerability
coverage_score
```

另外展示这三个分数的算术平均分。以下字段不能出现在普通列表、卡片、详情或筛选结果中：

```text
unsupervised_f1
coverage_self
coverage_recall_soft
r_soft
其他内部中间指标和实现字段
```

后端可以为了调试或历史兼容暂时保留内部字段，但前端展示必须收敛到上述三个指标和平均分。

### 2.2 integrated_document_task_1786649898 的四个疑问

1. 运行速度过慢。希望问答生成相关任务（不包括图片理解）的总时长达到每个问答对不大于 10 秒。
2. 有些生成结果被遗弃，但人工看起来质量还可以，需要知道具体在哪个阶段、因为什么被丢弃。
3. 有些块经常没有生成候选题，容易达不到目标数量。
4. 需要解释 Point / Summary 是怎么分配的、各自绑定哪些块、是否使用大模型、`scenario_intent` 和 `reader_need` 从哪里来，以及长文档如何规划。

## 3. 评估分数和过滤规则

无参考评估的过滤分数统一定义为：

```text
average_score =
    (faithfulness + answerability + coverage_score) / 3
```

缺失或非法的单项分数按 0 处理，分母仍然是 3。例如：

```text
faithfulness = 0.90
answerability = 0.80
coverage_score = 0.70
average_score = 0.80
```

阈值过滤必须使用：

```text
average_score >= score_threshold
```

前端显示的平均分、存储的 `average_score`、搜索过滤和批量任务的过滤必须使用同一套计算逻辑，不能继续使用旧的 `unsupervised_f1` 作为过滤依据。

## 4. 对任务 1786649898 的分析框架

当前不能只看“最后生成了多少题”，而要把每一个候选从规划到答案的生命周期拆开：

```text
场景规划
→ 场景类型校验
→ 材料引用校验
→ 预算选择
→ 候选问题生成
→ 候选问题去重
→ 问题编辑
→ 证据检索
→ 答案生成
→ evidence_usage 校验
→ Summary 主材料覆盖校验
→ 最终保存
```

此前对该任务观察到过类似数量：

```text
原始文本 chunk：约 73 个
Point 候选：约 25 个
Summary 候选：约 9 个
最终规划单元：Point 16、Summary 9
因预算或最终选择丢弃：约 9 个
最终成功生成：Point 4、Summary 2
```

这些数字不能直接说明模型质量。候选题可能在多个阶段被丢弃，因此必须记录明确原因，例如：

```text
candidate_empty
candidate_generation_error
unknown_material_ref
invalid_scenario_type
scenario_type_mismatch
point_requires_one_material
summary_requires_multiple_materials
duplicate_question
question_editor_drop
retrieval_error
missing_primary_evidence_usage
incomplete_summary_primary_coverage
answer_generation_error
budget_exceeded
```

调试页面和 debug JSONL 都应该能看到这些原因及计数。

## 5. 为什么不能直接取消场景规划

场景规划是质量控制环节，不是无意义的额外调用。它负责判断：

- 哪些事实适合生成单材料的 Point；
- 哪些相关事实应该组合成 Summary；
- 一个问题需要哪些材料才能完整回答；
- 读者真正想了解的内容是什么；
- 如何避免每个 chunk 机械生成相似问题；
- 如何减少只看局部文本导致的代词不明和上下文缺失。

提速方向应是：

```text
保留场景规划
→ 使用有界的字符/token预算
→ 按逻辑材料而不是按每个 chunk 调用
→ 互不依赖的规划批次并发
→ 结果按原文顺序恢复
→ 对相同材料和配置进行缓存
→ 后续 generation unit 并发执行
```

不能为了达到速度目标而删除 Point/Summary 语义拆分。

## 6. SectionMaterial 是什么

原始 chunk 不是最终的规划单位。首先要按逻辑节点把 chunk 组合成 `SectionMaterial`。

例如原文：

```markdown
# 办理指南

## 申请材料

申请人需要提交身份证明。
还需要提交申请表。

## 办理时限

材料完整后，五个工作日内办结。
```

切块后可能是：

```text
chunk 1
section_path: 办理指南 > 申请材料
正文：申请人需要提交身份证明。

chunk 2
section_path: 办理指南 > 申请材料
正文：还需要提交申请表。

chunk 3
section_path: 办理指南 > 办理时限
正文：材料完整后，五个工作日内办结。
```

chunk 1 和 chunk 2 的 `section_path` 相同，所以合并成：

```text
SectionMaterial A
路径：办理指南 > 申请材料
父节点：办理指南
正文：申请人需要提交身份证明。还需要提交申请表。
来源：chunk 1、chunk 2
```

chunk 3 形成：

```text
SectionMaterial B
路径：办理指南 > 办理时限
父节点：办理指南
正文：材料完整后，五个工作日内办结。
来源：chunk 3
```

合并时必须保持：

- 原始文档顺序；
- `section_path`；
- `section_parent_path`；
- 节点标题路径；
- 所有来源 chunk 的顺序和映射；
- 图片或其他资源的来源信息。

## 7. Point 和 Summary 的真实判定逻辑

不是人工固定某个 chunk 一定是 Point 或 Summary，也不是简单地按照 chunk 序号切分。

真实流程是：

```text
SectionMaterial 的路径和正文
→ planner 大模型提出场景候选
→ 后端校验场景类型和材料绑定
→ 去重、预算分配、质量选择
→ 生成最终 GenerationUnit
```

### 7.1 Point

Point 是主要由一个逻辑材料回答的具体事实问题。

例如：

```text
问题：申请人办理该事项时需要提交哪些材料？
required_materials：办理指南 > 申请材料
```

Point 的硬约束：

```text
required_material_refs 必须恰好一个
不能把两个不同 SectionMaterial 绑定成 Point
```

下面这种问题应当是 Summary，而不是 Point：

```text
申请材料和办理时限分别是什么？
```

### 7.2 Summary

Summary 需要组织多个相关事实，或者总结一个材料内部的多个真实信息点。

例如：

```text
问题：办理该事项需要准备哪些材料，办理周期通常是多久？
required_materials：
  - 办理指南 > 申请材料
  - 办理指南 > 办理时限
```

Summary 可以绑定：

- 多个必需材料；
- 同一逻辑材料内的多个事实点；
- 少量只用于辅助理解的 optional 材料。

但 optional 材料不能因为没有被最终答案引用，就导致高质量答案被丢弃。

### 7.3 `qa_detail_mode`

```text
point
  只规划 Point

summary
  只规划 Summary

auto
  同时建立 Point 和 Summary 候选池，由系统按目标比例、材料关系、硬约束和预算选择
```

在 `auto` 模式下，用户只选择整体模式，不需要逐块手工选择。每个材料的具体类型由 planner 的语义判断和后端规则共同决定。

## 8. 长文档规划方式

不能把整篇长文一次性发给 planner。需要按文档原始顺序，把 `SectionMaterial` 分成有界批次。

批次组织优先级：

```text
文档顺序
→ SectionMaterial 顺序
→ section_path
→ section_parent_path
→ 自适应字符/token预算
```

planner 看到的是：

```text
主材料-A
节点路径：办理指南 > 申请材料
父节点路径：办理指南
正文：申请人需要提交身份证明和申请表。
```

不能把以下内部信息直接暴露给模型：

```text
chunk_id
chunk_index
section-14
数据库内部数字 ID
```

`主材料-A` 只是本次请求内部的临时别名，不代表真实 chunk 序号。模型返回别名后，由后端映射回真实材料和来源 chunk。

### 8.1 父节点关系

```text
文档 > 第一章 办理指南 > 申请材料
```

其父节点路径是：

```text
文档 > 第一章 办理指南
```

父节点用于判断材料是否属于同一主题，减少不相关章节被错误组合成 Summary。

### 8.2 跨批次 Summary

如果一个长文档被拆成多个规划批次，每个批次先产生局部 Summary 候选。之后后端根据以下条件进行保守合并：

- 具有共同父节点；
- `scenario_intent` 有较高语义重合；
- `reader_need` 有较高语义重合；
- 绑定的材料属于同一主题；
- 合并后仍然满足 Summary 的材料约束。

这不是把全文再次发送给大模型，而是：

```text
各批次局部规划
→ 收集候选的意图、读者需求和节点路径
→ 后端判断是否同主题
→ 合并材料引用
→ 生成答案时再取回完整材料正文
```

## 9. `scenario_intent` 和 `reader_need`

这两个字段通常来自 planner 大模型。

`scenario_intent` 表示场景要覆盖的信息，例如：

```text
询问申请所需材料
总结办理时限和收费要求
比较普通申请与特殊申请的条件差异
```

`reader_need` 表示读者为什么需要这个问题，例如：

```text
读者想知道首次申请时必须准备哪些文件
读者希望一次了解办理周期和是否收费
```

它们会继续传给：

```text
候选题提示词
→ 问题编辑提示词
→ 答案生成约束
```

它们不是简单复制 chunk 内容，也不是人工逐块写死。planner 返回为空时可以使用保守兜底，但不能凭空编造事实。

## 10. 材料路径和代词不明确问题

只给模型一段孤立的 chunk 正文，容易生成：

```text
该项需要提交什么？
上述要求什么时候完成？
此类情况如何处理？
```

模型不知道“该项”“上述要求”“此类情况”分别指什么。

候选题、问题编辑和答案生成都应同时提供：

```text
节点路径
父节点路径
主材料正文
必要的关联材料路径和正文
```

提示词必须明确要求：

- 主体明确；
- 对象明确；
- 动作明确；
- 条件明确；
- 时间、金额、数量和适用范围明确；
- 不得使用没有先行词的“该项”“上述”“此类”“相关材料”等代词；
- 如果原文使用代词，生成的问题和答案必须展开成具体名称。

## 11. Summary 覆盖校验

之前的 `incomplete_summary_primary_coverage` 可能过于严格：如果 planner 把冗余材料也绑定成主材料，即使答案质量很好，也可能因为没有引用冗余材料而被丢弃。

应该把绑定材料分成：

```text
required_materials
  回答问题不可缺少的材料

optional_materials
  只用于辅助理解的材料
```

校验规则：

```text
Point：
  必须引用唯一 required material。

Summary：
  必须覆盖所有 required material。
  optional material 未被引用不能导致丢弃。
```

如果 planner 把明显无关或冗余的材料标成 required，应优先在规划校验阶段压缩或拒绝该候选，而不是让答案生成后承担不合理的覆盖要求。

## 12. 删除 `evidence_usage[].snippet`

模型不需要返回自由文本摘录：

```json
{
  "evidence_ref": "主材料-1",
  "snippet": "费用由责任主体承担。",
  "usage": "说明费用承担方式"
}
```

因为 `snippet` 可能是模型改写的文本，不是权威来源，也会和原文重复。

模型只需返回：

```json
{
  "evidence_ref": "主材料-1",
  "role": "primary_source"
}
```

后端根据 `evidence_ref` 恢复可审计的材料路径：

```json
{
  "evidence_ref": "主材料-1",
  "role": "primary_source",
  "title_path": "办理指南 > 费用标准"
}
```

审核人员需要看原文时，应通过材料路径查看真实正文，而不是依赖模型生成的 snippet。

## 13. 候选题缺失和生成失败处理

每个 generation unit 的处理链路应记录：

```text
候选题调用
→ 空候选或调用异常
→ 最多一次定向重试
→ 问题编辑
→ 检索
→ 答案生成
→ 引用和字段校验
```

候选题为空时：

1. 记录 `candidate_empty` 或实际异常；
2. 在剩余尝试次数内做一次定向重试；
3. 仍为空时，记录最终原因；
4. 如果整体数量不足，对尚未覆盖的可用材料生成证据约束的 Point 兜底候选；
5. 兜底候选必须继续经过正常编辑、检索、答案和引用校验。

兜底不能脱离材料自由编造问题。

## 14. 调试界面必须能回答什么

调试界面不应只显示“成功 6 条、失败 9 条”，而应能查看：

- 总 chunk 数；
- SectionMaterial 数；
- Point 规划批次；
- Summary 规划批次；
- 每个批次的材料路径；
- required/optional 材料路径；
- `scenario_intent`；
- `reader_need`；
- 请求数量；
- planner 返回数量；
- 校验通过数量；
- 丢弃数量；
- 每个丢弃原因；
- 最终选择的 Point/Summary 单元；
- 每个单元的来源路径；
- planner 原始响应；
- planner 实际输入材料路径；
- 候选题、编辑、检索、答案阶段耗时。

模型请求中不能出现内部 chunk ID，但调试审核界面可以显示后端映射后的来源路径和内部审计信息。

## 15. 一步实施计划

不要把这次任务拆成互相孤立的 UI、评分和后端小修补。一次性完成以下闭环：

### 后端规划和材料引用

- 保留 Point/Summary planner；
- 按逻辑 `section_path` 合并 SectionMaterial；
- 保留文档顺序、节点路径和父节点；
- 长文档使用自适应字符/token预算分批；
- 互不依赖的 planner 批次有界并发；
- 模型使用临时材料别名，不暴露 chunk ID；
- planner 返回后由后端映射真实材料；
- Point 只允许一个 required material；
- Summary 支持多个 required material 和 optional material；
- 只对 required material 做覆盖校验；
- 记录完整原始规划响应和校验结果。

### 速度和稳定性

- 保留质量规划，不采用取消 planner 的提速方式；
- 避免每个 chunk 单独调用 planner；
- 对相同材料和配置使用缓存；
- generation unit 并发执行；
- 可批量的 embedding/检索调用尽量批量化；
- 限制总 LLM/VLM 并发，避免并发过高导致超时；
- 分开记录墙钟耗时和并发累计耗时；
- 以非图片理解流程平均每问答对不超过 10 秒作为目标，并用 p50/p95 验证。

### 质量和数量

- 候选题空结果有明确原因和有限重试；
- 问题编辑丢弃有具体原因；
- 候选不足时使用材料约束的 Point 兜底；
- 去重、预算丢弃和答案覆盖失败都可追溯；
- 问题和答案必须明确主体、对象、动作、条件；
- 不允许无先行词代词；
- 不允许 Point 绑定多个 SectionMaterial；
- Summary 必须覆盖 required material；
- optional material 不得导致误丢弃。

### 前端和评估展示

- 无参考评估只展示三个指标和平均分；
- 筛选使用三项平均分；
- 调试模板内部区域可以独立滚动；
- 左右调试面板保持固定且等高；
- 左侧导航在桌面端滚动时保持可见；
- QA 列表独立滚动；
- QA 详情通过列表点击放大查看，不再占用永久布局；
- chunk 可点击放大查看正文；
- 保留已有 Logo、配色和 iOS 风格改造，不改变生成接口行为。

## 16. 验收标准

至少验证以下内容：

### 规划测试

- 单材料 Point；
- 多材料 Summary；
- Point 错误绑定多个材料；
- Summary required/optional 材料；
- 同一 section 的多个 chunk 合并；
- 长文档分批；
- planner 并发后结果仍按原文顺序；
- 跨批次 Summary 的保守合并；
- 未知材料别名；
- planner 返回空候选；
- Point 兜底。

### 证据和提示词测试

- 生成提示词包含节点路径；
- 模型输入不包含 chunk ID；
- 代词不明反例存在；
- `evidence_usage` 不再输出 `snippet` 和 `usage`；
- Summary 只按 required material 做覆盖校验；
- optional material 未引用不会误丢弃。

### 评估测试

- 三项平均分计算正确；
- 缺失项按 0 且分母为 3；
- 阈值过滤使用平均分；
- 前端只展示三个指标和平均分。

### 工程验证

- Python 语法和导入检查；
- JavaScript 语法检查；
- 相关单元测试；
- Docker runtime 内的 API smoke test；
- `/test-connection` 检查；
- 真实或回归任务的生成流程检查；
- UTF-8 无 BOM；
- 审核 diff 后提交本地未提交代码。

## 17. 可直接粘贴到另一个对话的提示词

下面是基于本文档生成的交接提示词。新对话应该把它作为当前任务上下文，而不是把旧回复中的“已完成”当作事实。

```text
你现在接手的是 /data2/hjk/qa-flow 项目。请先阅读仓库根目录的 AGENTS.md、INTEGRATION_CONTRACT.md、LATEST_CHANGE_GUIDE.md，以及 QA_GENERATION_CONTEXT_HANDOFF.md，然后检查当前工作区和实际代码。

这不是一个单纯的评分界面或 CSS 修改任务，而是一次问答生成质量、效率和可解释性治理。目标是完善：

文档/OCR
→ chunk 和结构节点
→ Point/Summary 场景规划
→ 候选问题
→ 问题编辑
→ 同文档证据检索
→ 答案生成
→ 无参考评估
→ 平均分过滤
→ 调试审核

不要依据旧对话中“已完成”的描述判断当前代码状态，也不要先做无关重构。先检查已有未提交改动，保留用户现有修改，不要 reset、checkout 或覆盖无关文件。

当前需求如下：

1. 无参考评估普通界面只展示 faithfulness、answerability、coverage_score 和三项平均分。不要展示 unsupervised_f1、coverage_self、coverage_recall_soft、r_soft 等内部指标。

2. 无参考评估的筛选分数统一使用：
   average_score = (faithfulness + answerability + coverage_score) / 3
   缺失或非法项按 0，分母仍然是 3。后端过滤、存储字段和前端显示必须一致，不能继续按旧 unsupervised_f1 过滤。

3. 保留 Point/Summary 场景规划，因为它负责语义拆分和质量控制。提速不能通过取消 planner 实现，而要使用逻辑材料分组、自适应字符/token预算、长文档分批、有界并发、缓存和 generation unit 并发。问答生成部分（不包含 OCR 和图片理解）目标平均每问答对不超过 10 秒，并记录墙钟时间、p50/p95 和各阶段耗时。

4. 原始 chunk 要按 section_path 合并为 SectionMaterial，保持文档顺序、section_path、section_parent_path、title_path 和来源映射。长文档不能全文一次性发送给 planner，要按材料顺序和自适应预算分批。Summary 批次要尽量保持父节点邻近关系。

5. planner 给模型的材料只能使用本次请求内的临时别名，例如 主材料-A、主材料-B，并且必须附带节点路径、父节点路径和正文。模型输入不能出现 chunk ID、chunk index、section 数字 ID 或数据库内部 ID。模型返回别名后由后端映射到真实 SectionMaterial。

6. Point 必须绑定恰好一个 required material。Summary 可以绑定多个 required material，也可以有 optional material。auto 模式不是人工逐 chunk 选择，而是同时建立 Point/Summary 候选池，再由 planner 语义判断、后端硬约束、去重和预算选择最终类型。

7. Summary 覆盖校验只检查 required material。optional material 没有被 evidence_usage 引用时不能导致答案被丢弃。分析并修正 incomplete_summary_primary_coverage 过严的问题。

8. evidence_usage 不再使用 snippet 或 usage。模型只返回 evidence_ref 和 role，后端根据映射恢复可审计的材料路径。真实原文通过路径和审核界面查看，不信任模型自由生成的 snippet。

9. 候选题为空、候选调用异常、问题编辑丢弃、检索失败、答案解析失败和预算丢弃都必须记录具体原因。空候选最多做一次定向重试；数量仍不足时，对未覆盖的可用材料生成有证据约束的 Point 兜底，并继续经过编辑、检索、答案和引用校验。

10. 候选题、问题编辑和答案提示词必须提供节点路径和必要的父节点信息，明确要求主体、对象、动作、条件、时间、范围清楚，禁止无先行词的“该项”“上述”“此类”“相关材料”等代词，并加入正反例。

11. planner 原始调试信息必须可查看，包括批次编号、Point/Summary 类型、材料路径、required/optional 路径、scenario_intent、reader_need、请求数量、返回数量、校验通过数量、丢弃原因和 raw_response。模型请求不暴露内部 chunk ID，但调试界面可以展示后端映射后的来源审计信息。

12. 前端保留已有 QA Flow 功能和接口行为，完成调试界面滚动修复：左右面板固定且等高、各自内部独立滚动、左侧导航桌面端滚动时保持可见、QA 列表独立滚动、QA 详情改为点击放大、chunk 可点击放大查看。保留已有 Logo、配色和 iOS 风格改造。

请按以下顺序工作：

第一，检查当前改动和实际代码，确认哪些需求已实现、哪些只是旧回复中的口头描述。
第二，追踪规划配置从前端/路由到 text_to_qa_pipeline、pipeline_runtime 和 integrated/batch service 的完整传递。
第三，修正材料路径、Point/Summary 绑定、长文档分批并发、Summary required/optional 覆盖、snippet 删除和候选不足补偿。
第四，完成 planner 原始调试信息的后端到前端展示。
第五，修复调试页面滚动、等高和 chunk 放大交互。
第六，补充或修正测试，执行 Python/JS/Docker/API 回归验证，检查 UTF-8 无 BOM。
第七，更新必要的 INTEGRATION_CONTRACT.md 和 LATEST_CHANGE_GUIDE.md，审核 git diff，最后提交所有本地未提交的相关代码。

不要只给我一个泛泛的计划，也不要只修改分数展示。请以代码和测试结果为依据汇报完成情况；如果某项还没有完成，明确指出实际原因。
```
