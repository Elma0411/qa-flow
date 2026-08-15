# Latest Change Guide

更新时间：2026-08-16（Asia/Shanghai）

## Objective

提升 Point/Summary 出题的自然性、主体明确性与图片利用率，并修复 Summary
材料过绑定、跨题型重复和生成后数量不足。图片识别模型本身不在本轮范围内。

## Effective Changes

- 集成文档交接在原有可检索正文之外新增 `image_materials`。每张已接受图片保留
  稳定来源 ID、描述以及图片前后上下文；SectionMaterial 给模型的材料改成
  `text_content + image_materials`，图片使用请求内 `图片-A` 别名，不暴露来源 ID。
- 场景新增 `evidence_mode=text|visual|mixed` 与 `required_image_refs`。planner 会考虑
  图片是否提供可独立提问的新事实，但不强制图片题配额；删除图片描述后仍可完整
  回答的问题归为 text。
- 从可读节点路径推导 `subject_label`。候选题和问题编辑器只在独立问题主体不明时
  使用它，解决“该条例/该系统”脱离文档后无先行词的问题。
- 候选题从 `reader_need` 出发。Point 只保留一个核心意图；Summary 使用一个自然
  总括问题，细节留在答案中。问题编辑器可以把不必需材料从 required 调整为
  optional，从而让 Summary 覆盖校验只约束编辑后真正必需的材料。
- `keep` 决定若命中模糊指代、原句条件从句形态、多意图或异常长度信号，会额外
  进行一次 LLM 复审；信号本身不执行机械改写或删除。问答增广问题也经过同一编辑器。
- planner 输出的 `point only`/`summary only` 会规范成有效枚举；提示词只要求
  `point`/`summary`。文档级去重增加保守语义比较，并跨 Point/Summary 生效。
- 规划结果保留少量候补 generation unit。主单元因编辑、答案、覆盖或去重造成
  数量不足时才按缺口执行候补；达到目标时不会生成多余问答。
- 持久化和调试记录新增 `evidence_mode`、`required_image_refs`、
  `qa_generation_subject_label`、reviewed required/optional material IDs。

## Open-Source Design References

- Docling：文档层级、reading order 与 Picture/Table typed item 思路。
- LlamaIndex：文本节点与图片节点分离、在合成阶段保留多模态来源关系。
- RAG-Anything：图片描述与局部上下文绑定、按模态保留元数据。
- Ragas：先规划场景/读者需求，再生成具体问题以及按分布选择场景。

本轮只借鉴这些数据建模和流程原则，没有复制第三方实现，也没有把
`external_repos/` 纳入提交。

## Expected Behavior

带图片任务的 planner 调试输入能看到独立图片块；visual/mixed 问题必须依赖所列
图片描述。普通问题不再为了图片而写“图中显示”。问题编辑后应尽量避免模糊指代、
条文前半句式问法和多个独立问项；Summary 不再因可选背景未引用而触发
`incomplete_summary_primary_coverage`。最终数量不足时会在候补场景存在的范围内补齐。

## Validation

```bash
docker exec qa-flow-runtime sh -lc 'cd /app && python -m compileall -q app qa scripts tests'
docker exec qa-flow-runtime sh -lc 'cd /app && python -m unittest discover -s tests -v'
docker exec qa-flow-runtime sh -lc 'cd /app && python -c "import app.main"'
curl -fsS http://localhost:12000/test-connection
curl -fsS http://localhost:12000/health
```

修改文件必须保持 UTF-8 无 BOM。`AGENTS.md` 的既有本地修改仍属于用户，不纳入本轮提交。
