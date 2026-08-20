# Latest Change Guide

更新时间：2026-08-20（Asia/Shanghai）

## Objective

针对 `integrated_document_task_1787210901` 的非评估问题，修复 Summary 容易被
单点题覆盖、Summary required material 绑定过重、自然问题仍有来源包装和答案泄露、
视觉题把完整步骤写进题干后只回答“是/否”，以及最终去重数量没有进入产物的问题。

本轮不修改任何评估指标、平均分、过滤阈值或 Milvus schema。

## Evidence and References

- 该任务规划了 5 个 Summary，全部生成有效答案，但 4 个因
  `incomplete_primary_material_coverage` 触发答案重试，最终又有 1 个在文档级
  去重中消失。
- 参考 Ragas 的多跳场景/节点关系、EasyDataset 的自然与视觉问题约束、PREMIR 的
  text/visual/multimodal pre-question 分池，以及 ACL 2023 HQDT 的原子问题分解思想。
- 完整论文检索报告保存在 `ResearchStudio/allinone.md`。检索源错误和 28 篇结果均在
  报告内保留。

## Effective Changes

- 文档级去重不再因为 Summary 的完整证据包含一个 Point 子事实，就把 Summary 判为
  重复。Point/Summary 之间只有在完整事实、问题语义和事实长度都高度等价时才合并；
  同类型问题仍可使用直接事实包含关系去重。
- Summary planner 被明确要求：每份 required material 必须提供答案不可缺少的独立
  事实；背景、佐证、重复政策和仅用于理解范围的材料必须放到 optional。没有真实枚举
  或至少两个独立答案贡献时，应少返回 Summary，而不能把单点事实伪装成 Summary。
- planner/writer/editor 都明确禁止把答案中的数值、日期、名单项和完整步骤提前写进
  题干。视觉问题聚焦可观察的操作、状态、分支或反馈，不再通过罗列完整步骤后询问
  “是/否”来利用图片。
- 编辑器会移除不必要的“请问在《…》中”“根据《…》”来源前缀，同时继续保留真正
  的业务主体和合法的许可/禁止类肯否问题。
- `duplicate_questions_dropped` 现在从生成结果传到任务进度、文件 timing、最终
  consolidated timing 和调试页面，便于区分预算丢弃、答案失败和文档级去重。

## Deliberately Not Changed

- 尚未把 Summary 升级为显式“原子子问题/证据跳”字段。该方案最稳健，但会改变
  planner 与答案验证契约，需要产品确认后单独实施。
- 现有 required-material 全覆盖校验仍保留；本轮先通过更准确的 required/optional
  规划减少无意义重试，不直接放松完整性约束。

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
