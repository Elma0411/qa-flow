# Latest Change Guide

更新时间：2026-08-18（Asia/Shanghai）

## Objective

针对 `integrated_document_task_1787055720` 中暴露的生成链路问题，修复
Summary 规划跑偏、问句语义被编辑器改坏、跨材料重复题、图片题覆盖不足，以及
规划/答案重试的审计盲点。

本轮**不修改** faithfulness、answerability、coverage、平均分或任何过滤逻辑。

## Effective Changes

- Summary-only / Point-only planner 的 JSON 示例现在只允许当前批次类型；不再在
  Summary 请求中展示 `point|summary`。如果一个非空批次因类型不匹配而全部失败，
  后端仅做一次定向纠正重试。调试记录首次响应、重试原因、尝试次数和最终响应。
- planner 的 `intent` 必须带上区分相近规则所需的主体、动作、条件或办理渠道，不能
  使用“本说明”“该文件”“上述”等文档指代。不同阶段的“向经办机构提供资料”和
  “线上上传文件”必须规划成不同场景。
- writer/editor 保留合法的“是否/能否/还能……吗”许可、禁止和资格限制问句，不能
  擅自改成“如何”或“有什么政策”。若模型仍把肯否题改成 how-to，后端保留原始
  肯否关系。题干中的“本/该/这份说明、通知或文件”会用节点路径提取出的简短业务
  对象替换，避免脱离文档后指代不明。
- 图片仍是 `text|visual|mixed` 证据模式，而不是新的题型。没有固定图片配额；当
  同一材料中一个未选视觉场景与一个已选文本场景竞争时，只有可观察的操作、状态、
  分支、反馈或确认动作才会提升为视觉场景。纯数值截图不因多样性而替换文本题。
- 文档级去重在答案落地后还会比较已引用的 `source_fact_text`。同一规则被不同章节
  复述时，直接事实包含或强重叠会判为重复；Point/Summary 冲突时优先保留更完整的
  Summary，随后按原有 reserve unit 机制补足目标数量。
- generation unit 调试新增 `answer_attempt_count`、`answer_retry_count` 和
  `answer_retry_reasons`。场景规划批次新增 `planner_seconds`、重试次数和原因；
  页面可显示这些信息，并可查看重试前的规划原始响应。

## Contract Notes

- planner 继续是唯一决定 Point/Summary、required/optional materials、图片和
  `evidence_mode` 的阶段；writer/editor/answer 不会重分类这些语义字段。
- 新增的去重和编辑保护不会改变冻结的 `ScenarioContract`，也不向模型暴露真实
  chunk/image ID 或其他内部字段。
- 本轮未改动 QA Milvus schema、`qa_pairs_collection_v2`、评分字段或分数阈值。

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
