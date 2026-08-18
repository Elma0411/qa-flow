# Latest Change Guide

更新时间：2026-08-18（Asia/Shanghai）

## Objective

针对 `integrated_document_task_1786989603` 暴露的问答质量问题，完成第二轮
QA 生成治理：在不改变 planner 证据契约的前提下，提高问题自然度、抑制重复图片题，
并使文本/图片题的无参考评估只依据答案实际引用的证据。

链路保持为：

`SectionMaterial -> frozen ScenarioContract -> question writer -> wording editor -> retrieval -> evidence answer -> cited-evidence evaluation`

## Effective Changes

- 借鉴 EasyDataset 的“选定信息焦点 → 静默检查 → 自然表达”思路，但没有复制其
  多题全量提示词或推理策略。planner 只输出场景语义；候选题生成器和编辑器只输出
  `{"question":"..."}`，不会重新判定材料、图片、Point/Summary 或证据模式。
- writer/editor 提示词现在明确要求：问题只保留最少身份上下文，不提前复述数值、
  日期、步骤或截图内容；规则、条件、期限和流程必须直接提问，不能生成“是否/能否/
  是这样吗”式确认题；视觉题自然询问可观察界面事实，不写“图中/截图中”。
- planner 的 `intent` 被约束为单一信息缺口。独立金额、条件、阶段和流程会拆成
  Point；Summary 只保留一个读者结果，最多绑定三份紧密相关的必需材料。扫码、点击
  或离开图片后才可能得到的外部结果不能规划为纯视觉题。
- 必需图片描述高度相同的跨章节场景会在选择阶段去重，避免同一操作流程图片生成多道
  近义题；不同材料但真正不同的图片事实仍可保留。
- 答案首次因缺少必需正文/图片引用、遗漏 Summary 必需材料或空项而失败时，最多进行
  一次定向重试。重试只补充缺失的可读证据标签，不改变冻结的场景契约；调试日志记录
  首次响应、尝试次数、重试原因和错误。
- 证据渲染器为每个 `正文证据-N`、`图片证据-N` 和补充证据保存对应可读文本。
  答案通过引用校验后，后端生成并持久化 `qa_evaluation_evidence_text`：只包含该答案
  实际引用的证据块。faithfulness、answerability、coverage、自动指标、LLM 评估、
  汇总分组和问答增广都优先使用该字段；旧产物缺失时仍回退到完整出题单元。
- 查询详情新增“评分依据”查看区，便于核对图片题是否真的使用了图片事实。真实
  chunk/image ID 仍只留在后端审计字段，未写进 writer/editor 的提示词。
- generation unit 的 `latency_percentiles`（包含候选题、编辑、检索、答案和总耗时
  的 p50/p95）现在同时写入任务进度和最终 consolidated 产物。

## Contract Notes

- 新 primary QA item 的 `qa_evaluation_evidence_text` 是评估首选上下文，不是新的
  LLM 输入字段，也不替代完整的 `qa_generation_unit_text` 调试材料。
- QA 向量集合仍固定为 `qa_pairs_collection_v2`；不会读取或写入旧
  `qa_pairs_collection`，本轮没有新增 Milvus schema。
- `difficulty_level`、`difficulty_score`、`question_type_reason` 仍不在生成、存储、
  调试、搜索或向量 schema 中。

## Validation

```bash
docker exec qa-flow-runtime bash -lc 'cd /app && python -m compileall -q app qa scripts tests'
docker exec qa-flow-runtime bash -lc 'cd /app && python -m unittest discover -s tests -v'
docker exec qa-flow-runtime bash -lc 'cd /app && python -c "import app.main, qa.generation, qa.augmentation"'
bash -ic 'node --check static/app.js && node --check static/admin.js && node --check static/app_render.js && node --check static/app_query.js'
curl -fsS http://localhost:12000/health
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/milvus-status
curl -fsS http://localhost:11169/health
```

修改文件必须为 UTF-8 无 BOM。`AGENTS.md` 的既有本地修改属于用户，不纳入提交。
