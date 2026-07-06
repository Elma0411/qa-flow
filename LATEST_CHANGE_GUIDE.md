# Latest Change Guide

更新时间：2026-07-07（Asia/Shanghai）

## Objective

收紧 `qa_detail_mode=summary` 的候选问题生成与过滤，减少“只列孤立条目”的浅层清单题被当作总结型 QA 保留。总结型现在更偏向多个事实之间的关系：组成、顺序、条件、因果、对比、作用、约束、例外、依赖或取舍。

## What Changed

- `qa/prompts/qa_generation_prompts.py`
  - summary 候选问题 prompt 明确要求生成“多个事实之间是什么关系”的问题。
  - 新增通用反例约束：不要只问孤立条目、名称、标签、数值、字段、选项或对象。
  - summary 答案 prompt 同步要求：如果候选问题只是浅层清单，且没有询问组成、顺序、条件、因果、对比、作用、约束、例外、依赖或取舍，应输出空结果。
- `qa/generation/qa_generation_flow.py`
  - `_summary_question_shape_reason()` 不再因为出现“哪些”就放行。
  - 新增浅层清单过滤：`summary_question_too_shallow_list`。
  - 过滤依据是抽象关系语义，不绑定具体文件类型或业务场景。
- `static/app.js`
  - 新增 `summary_question_too_shallow_list` 的中文解释。

## Expected Behavior

- summary 模式下，候选问题会减少简单清单题。
- 只问“有哪些条目/对象/选项”的问题会倾向于被丢弃。
- 问“这些条目如何组成整体、前后如何衔接、不同条件下如何变化、差异和取舍是什么”的问题会被保留。
- 不改变接口字段、任务流程、检索算法或最终 QA schema。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m py_compile \
  qa/prompts/qa_generation_prompts.py \
  qa/generation/qa_generation_flow.py

node --check static/app.js
git diff --check
```

轻量断言：

```bash
python - <<'PY'
from qa.generation.qa_generation_flow import _summary_question_shape_reason

assert _summary_question_shape_reason("系统结构由哪些部分组成，各部分之间是什么关系？", language_code="zh") == ""
assert _summary_question_shape_reason("不同条件下的处理结果有什么差异？", language_code="zh") == ""
assert _summary_question_shape_reason("该部分有哪些条目？", language_code="zh") == "summary_question_too_shallow_list"
assert _summary_question_shape_reason("可以选择哪些选项？", language_code="zh") == "summary_question_too_shallow_list"
assert _summary_question_shape_reason("该部分包含哪些内容？", language_code="zh") == "summary_question_too_shallow_list"

assert _summary_question_shape_reason("Which components form the system structure and how do they relate?", language_code="en") == ""
assert _summary_question_shape_reason("What results differ under different conditions?", language_code="en") == ""
assert _summary_question_shape_reason("Which items are available?", language_code="en") == "summary_question_too_shallow_list"
assert _summary_question_shape_reason("What does this section contain?", language_code="en") == "summary_question_too_shallow_list"
PY
```
