# Latest Change Guide

更新时间：2026-07-07（Asia/Shanghai）

## Objective

优化 QA 生成调试中的硬过滤规则，降低 summary 模式误杀率。过滤策略从“问题必须命中固定关系词”调整为“只拦截明显泛化的浅层清单题”，并把 summary 来源事实定位从全片段强匹配改成多数关键片段可定位。

## What Changed

- `qa/generation/qa_generation_flow.py`
  - `_summary_question_shape_reason()` 不再要求总结题必须包含固定关系词。
  - `summary_question_not_grouped` 保留为历史 reason code，但新逻辑不再主动产生。
  - `summary_question_too_shallow_list` 只用于明显泛化的问题，例如“该部分有哪些内容？”或 “What items are listed?”。
- `qa/grounding/source_fact_grounding.py`
  - summary 来源事实分段定位阈值从 `0.84` 调整到 `0.76`。
  - 多片段 summary 不再要求每个片段都命中；至少 2 个片段且不少于 66% 片段可定位即可通过。
  - 如果没有任何片段命中，返回 `summary_source_fact_not_grounded_in_chunk`；部分命中不足时返回 `summary_source_fact_segment_not_grounded_in_chunk`。
- `qa/generation/text_quality_filters.py`
  - 指代过滤从简单词命中改为“未消解指代”判断。
  - 允许同一句内有明确先行词的中文“其中/其”和英文 “this/that + 明确名词短语”。
- `static/app.js`
  - 更新调试 reason 文案，避免把软化后的规则描述成绝对错误。
- `tests/test_generation_quality_filters.py`
  - 增加 summary 形态、summary grounding、未消解指代过滤测试。

## External Reference

这次调整参考了成熟 RAG 评估框架的通用做法：RAGAS/TruLens/DeepEval 的 faithfulness 或 groundedness 更强调答案 claim 与 retrieval context 的证据一致性/覆盖率，而不是仅靠问题表面关键词一票否决。DeepEval 的 faithfulness 文档也把判断建模为 truthful claims / total claims，并把 ambiguous claims 作为可配置惩罚项，而不是默认永久硬失败。

## Expected Behavior

- summary 模式下，“需要哪些材料和记录”“覆盖哪些步骤”这类可由多个事实回答的问题不再因为没有固定关系词被丢弃。
- 过于泛化的“该部分有哪些内容”“What items are listed?” 仍会被过滤。
- summary 来源事实允许少量片段因改写、截断或表达差异未能定位，但多数核心片段必须能回到证据。
- 未消解指代仍会被拦截；同一句中已经给出明确先行词的局部指代不再误杀。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m py_compile \
  qa/generation/qa_generation_flow.py \
  qa/generation/text_quality_filters.py \
  qa/grounding/source_fact_grounding.py

python -m unittest tests.test_generation_quality_filters

node --check static/app.js
git diff --check
```
