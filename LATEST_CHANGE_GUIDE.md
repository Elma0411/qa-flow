# 最新变更指南

更新时间：2026-08-11（Asia/Shanghai）

## Objective

取消问答生成完成后的语义硬过滤，避免规则误杀已经生成的 QA；同时修正总结型
候选问题的生成契约，使每个 item 只包含一个中心问题，答案再对相关段落事实
进行归纳。

## What Changed

- `qa/generation/qa_generation_flow.py`
  - 删除候选问题的指代词、总结题形态和来源锚点 grounding 硬过滤。
  - 删除答案阶段的问题文本一致性、source fact 粒度、grounding 和主来源块
    锚定硬过滤；答案模型回写的问题最终统一覆盖为候选问题。
  - 保留缺少必要字段、无效题型结构和精确重复问题等结构处理。
- `qa/validation/qa_item.py`
  - 删除问题、答案、答案解释及 source fact 的指代词正则拒绝。
  - 保留必填字段、单选题四选项/正确项和判断题答案格式归一。
- 删除不再使用的 `qa/generation/text_quality_filters.py` 和
  `qa/grounding/` 生成后硬过滤模块，并移除跨层 validator 参数传递。
- `qa/prompts/qa_generation_prompts.py`
  - 总结模式要求每个 item 只有一个完整问句、一个问号和一个中心意图。
  - “总结型”现在描述答案组织方式：答案可归纳一个相关段落或紧密相关段落组
    中的多个事实，但不能把多个独立问题拼接成一题。
  - 答案阶段不再自行输出空 items 进行二次质量筛选；证据细节不足时应如实说明，
    最终质量接纳交给后续评价阶段。
- `static/app.js` 删除不再可能产生的语义硬过滤 reason 文案，`static/index.html`
  更新资源版本以避免浏览器继续使用旧脚本。
- `tests/test_qa_generation_contract.py` 覆盖单问题总结契约及无语义硬过滤行为。

## Expected Behavior

- 结构完整的候选问题和答案不会因为指代词、固定题形正则、source fact 分段数、
  文本相似度或主块锚定阈值而被丢弃。
- 无法解析的 JSON、缺少问题/答案/答案解释/source fact、单选题结构无效、判断题
  答案无效及精确重复问题仍会被结构处理。
- 总结型问题不会再出现“问题一？问题二？问题三？”的拼接形式；如果来源包含
  多个无关主题，应输出为不同 item。
- 生成结果只代表结构有效，不代表质量通过；需要筛选时使用后续评价体系。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m unittest tests.test_qa_generation_contract
python -m compileall qa app
node --check static/app.js
git diff --check

docker exec qa-flow-runtime bash -lc \
  'cd /app && python -m unittest tests.test_qa_generation_contract'
docker exec qa-flow-runtime bash -lc \
  'cd /app && python -m compileall qa app'
curl http://localhost:12000/test-connection
```
