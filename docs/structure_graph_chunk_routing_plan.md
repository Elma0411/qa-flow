# 轻量结构图与自动出题策略方案总览

## 一句话说明

这个方案不是要做复杂知识图谱，而是让 QA Flow 在出题前先理解文档的基本结构：哪些 chunk 是同一章节的兄弟，哪些 chunk 只是低价值碎片，哪些内容适合单点问答，哪些内容需要合并成一段上下文后再生成总结题。

最终目标是让系统少处理无用 chunk，少生成浅层或证据不完整的问题，并且在构建答案提示词时继续保留“基于具体问题检索证据”的能力。

## 当前流程先保持不变

当前 QA Flow 大体流程可以理解为：

```text
文档输入 / OCR
-> 文本整合
-> chunk 切分
-> 一步式 QA 生成
   -> 每个 chunk 生成候选问题
   -> 根据候选问题检索同文档证据
   -> 构建答案提示词
   -> 生成最终 QA
-> 评估
-> 保存结果 / 入库
```

这条主流程不需要推倒重做。新方案只是在“chunk 切分之后、候选问题生成之前”增加一层轻量决策，并在“答案提示词构建时”增加去重和合并逻辑。

## 要做的改进

### 1. 把 flat chunk list 组织成轻量结构图

当前 chunk 虽然是平铺列表，但每个 chunk 已经有这些信息：

- `chunk_index`
- `index_path`
- `title_path`
- `parent_index_path`
- `root_index_path`
- `level`
- `text`
- `text_for_embedding`

这些信息足够先在内存里组织出一个轻量结构图：

```text
同一个 parent_index_path 下的 chunk 是兄弟
chunk_index 前后相邻的是 prev / next
title_path 和 level 表示章节路径
```

第一版不需要新建数据库表，也不需要把 parent 节点持久化。运行时临时构建即可。

插入位置：

```text
chunk 切分完成
-> build_document_chunks
-> 新增：构建轻量结构图
-> 后续 QA 生成
```

### 2. 加 chunk 质量门控

不是所有 chunk 都值得单独调用大模型生成问题。比如：

- 只有标题
- 目录行
- 页眉页脚
- 表格列名残片
- 图片或附件占位
- 和前后 chunk 高度重复

这些 chunk 不应该直接生成问题。它们可以被跳过，或者作为上下文交给相邻 chunk / 同章节 summary 使用。

质量门控不需要大模型，第一版只用轻量规则：

- 长度和行数
- 标题路径是否存在
- 是否有完整句
- 是否包含日期、金额、电话、定义、条件、步骤等信息信号
- 是否和前后 chunk 重复
- 是否处在同一章节的连续片段中

插入位置：

```text
构建轻量结构图
-> 新增：判断每个 chunk 的质量
-> 只把值得处理的 chunk / unit 交给候选问题生成
```

### 3. 从“按 chunk 出题”改成“按 generation unit 出题”

这是本方案最重要的调整。

之前容易把问题简化成：这个 chunk 走 `point` 还是 `summary`。这个判断会有矛盾，因为一个 chunk 可能很长，内部已经包含多个条件、流程、材料、分类项，本身就适合 summary；也可能一个章节被切成多个短 chunk，需要合起来才适合 summary。

因此后续不应该直接按单个 chunk 二选一，而是先生成 generation unit。

generation unit 可以有三类：

```text
point unit
  单个 chunk 就能回答一个清晰事实问题

section summary unit
  同一章节下多个兄弟 chunk 合起来才完整

long-chunk summary unit
  单个 chunk 很长，内部包含多个小段、条款、列表或流程
```

这样可以解决两个问题：

- 多个短 chunk 组成一个章节时，可以合成 summary unit。
- 单个长 chunk 适合 summary 时，也不会被误判成 point。

插入位置：

```text
chunk 质量门控
-> 新增：构建 generation units
-> 候选问题生成不再只看单个 chunk，而是看 unit
```

### 4. `auto` 不再是简单打标签

`qa_detail_mode=auto` 不应该理解成“看到流程词就 summary，看到电话就 point”。这些词只能作为弱信号，不能作为最终标签。

更合理的原则是：

```text
先看这个内容是否值得处理
再看它是单点事实，还是需要成组理解
最后再决定这个 generation unit 用 point prompt 还是 summary prompt
```

直观例子：

```text
客服电话：400-xxx
```

适合 point。

```text
退费规则：
一、申请条件
二、申请材料
三、办理流程
四、不予退费情形
```

适合 summary，即使它在同一个长 chunk 里。

```text
某公司内部资料 第 3 页
某公司内部资料 第 4 页
```

不适合出题，应该被质量门控压低。

插入位置：

```text
generation unit 构建完成
-> 新增：为每个 unit 决定 point / summary / skip
-> 再调用候选问题 LLM
```

### 5. 保留基于具体问题的检索，不和新方案冲突

结构图和 generation unit 只决定“从哪里出题”。它们不替代后面的基于问题检索。

后续仍然应该保持：

```text
候选问题生成后
-> 用候选问题做同文档检索
-> 找到回答这个具体问题需要的证据
-> 构建答案提示词
```

区别是：答案提示词要知道哪些 chunk 已经在 source unit 里，避免重复塞入。

需要新增几个概念：

```text
source_unit_chunk_ids
  当前出题单元本身包含的 chunk

retrieved_chunk_ids
  基于问题检索回来的 chunk

final_prompt_chunk_ids
  去重和裁剪后真正放进答案提示词的 chunk
```

这样就不会出现“summary unit 已经包含了某几个 chunk，问题检索又重复塞一遍”的情况。

插入位置：

```text
候选问题生成
-> 现有：基于问题检索
-> 新增：source unit 与 retrieved evidence 去重
-> 构建最终答案提示词
```

### 6. 答案提示词要从“主来源块”升级为“主来源单元”

当前答案生成主要围绕 source chunk。引入 generation unit 后，提示词里的主来源应该改成：

```text
主来源单元
  可能是一个 chunk
  也可能是同章节多个 chunk
  也可能是一个长 chunk 内部的多个片段
```

然后再追加问题检索得到的补充证据。

提示词结构可以理解为：

```text
【主来源单元】
这里是本次出题/回答的核心上下文

【问题检索补充证据】
这里是根据具体问题找回来的额外证据

【证据范围说明】
哪些 chunk 是主来源，哪些 chunk 是补充，哪些被去重或丢弃
```

插入位置：

```text
build_generation_unit
-> 新增：主来源单元渲染
-> 新增：补充证据去重
-> 新增：证据范围说明
```

### 7. 前端不新增独立页面，只加少量可理解配置

前端不需要新建“结构图模式”页面。还是在现有生成配置里增加：

- `auto` 问答粒度
- 是否启用轻量结构图
- 是否启用 chunk 质量门控
- 是否启用自动合并上下文

调试展示里增加：

- 哪些 chunk 被跳过
- 哪些 chunk 被合并成一个 generation unit
- 某个 unit 为什么走 point 或 summary
- 最终答案提示词用了哪些 chunk

插入位置：

```text
现有生成配置面板
-> 增加少量开关和 auto 选项

现有任务详情 / debug
-> 增加 route、quality、unit、prompt chunk 信息
```

## 插入到现有流程的完整位置

推荐后的流程：

```text
1. 文档输入 / OCR
   不变

2. 文本整合
   不变

3. chunk 切分
   不变，继续产出 pre_split_chunks 和 pre_split_chunk_meta

4. 构建 document_chunks
   现有 build_document_chunks

5. 新增：轻量结构图
   根据 chunk_index、title_path、parent_index_path、level 组织 prev/next 和 parent/children

6. 新增：chunk 质量门控
   判断 keep / skip / merge_with_neighbors

7. 新增：generation unit planning
   生成 point unit、section summary unit、long-chunk summary unit

8. 候选问题生成
   从“按 chunk 生成”改成“按 generation unit 生成”

9. 基于问题检索证据
   保留现有逻辑

10. 构建答案提示词
   主来源从 source chunk 升级为 source unit
   检索证据继续追加，但要去重和控制长度

11. 答案生成
   基本不变，只是接收更清楚的上下文

12. 评估 / 保存 / 入库
   基本不变
```

## 这个方案能提升什么

### 减少无效调用

标题、目录、页眉页脚、重复块不会再单独触发候选问题生成。

### 提升 summary 质量

summary 不再靠单个 chunk 硬生成，而是先组织成 section unit 或 long-chunk unit，再生成问题。

### 降低证据不完整

同一章节多个相关 chunk 会作为主来源单元一起进入提示词，减少 summary 答案找不到依据的问题。

### 避免 prompt 重复和过长

source unit 和 question retrieval 结果会去重，最终只把必要 chunk 放进答案提示词。

### 保持现有检索能力

候选问题生成之后仍然基于具体问题检索证据，结构图只提供更好的出题上下文，不替代检索。

## 第一版不做什么

第一版先不做：

- 不做实体关系图谱。
- 不引入 NER 模型。
- 不新建用户可见 pipeline。
- 不改变 OCR 和 chunking 主流程。
- 不强行改 Milvus schema。
- 不把所有阈值一次性定死。

第一版重点是把流程位置放对：

```text
chunk 后先做结构和质量判断
再构建 generation unit
再出题
再基于问题检索证据
最后构建去重后的答案提示词
```

## 后续再写详细实现方案

这份文档只确定产品和流程层面的方案。后续详细实现方案再展开：

- 具体新增哪些 dataclass。
- `generation unit` 的字段结构。
- `point unit / section summary unit / long-chunk summary unit` 的判定细节。
- 质量门控的具体规则和阈值。
- source unit 与 retrieval evidence 的去重算法。
- 后端接口参数如何命名。
- 前端如何展示 debug 信息。
- Docker 内怎么验证。

在进入实现前，应该先用已有任务样本做一次人工审阅，确认这三个 unit 类型是否覆盖主要问题：

- 普通单点事实
- 同章节多 chunk 总结
- 单个长 chunk 内部总结
