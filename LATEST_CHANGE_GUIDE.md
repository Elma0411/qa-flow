# 最新变更指南

更新时间：2026-08-12（Asia/Shanghai）

## Objective

让候选问题表达真实用户的信息需求，而不是机械改写法规、通知、标准或论文的
原句。此次只调整候选问题设计和类别提示词，不改变 QA 字段、检索链路或答案
证据结构。

## What Changed

- `qa/prompts/qa_generation_prompts.py`
  - 候选问题生成先在模型内部确定用户对象、信息需求和会改变答案的最小必要条件，
    再输出自然问句。
  - 标题路径只作为内部消歧和检索语境；禁止在问题中使用“根据/依据”、条号、
    文件/章节标题或“文中指出”等来源视角措辞。
  - 精确检索所需的完整条件、专有名称和来源术语继续放在 `retrieval_query` 与
    `must_have_terms`，不要求堆入问题。
  - 增加正反例和“真实用户会自然这样问吗”的静默自检。
- `qa/prompts/category_templates/`
  - `general`、`normative`、`official_dispatch`、`standard`、`research`、
    `knowledge_material` 都增加默认提问者视角和对应的自然提问方式。
  - 法规、通知、标准和论文模板明确禁止条款/材料复述式问法，并保留其各自的
    专业信息重点。
  - 所有类别答案模板不再要求证据不足时输出空结果；现在要求不臆测，并如实说明
    缺失细节，和生成阶段始终返回已选候选题的规则一致。
- `tests/test_qa_generation_contract.py`
  - 覆盖自然用户提问契约、标题路径内部化以及法规模板提问者视角。

## Expected Behavior

生成链路保持为：

```text
主来源块 -> LLM 生成 question + 检索规划
        -> 检索并组装 evidence
        -> LLM 生成 answer + source_fact_text + evidence_usage
```

- 问题应像实际办事人、执行人、工程人员、研究者或学习者会提出的问题。
- `question` 保持简洁、自然且自洽；完整条件和精确术语仍由检索规划和最终
  `source_fact_text`、`evidence_usage` 保证可追溯。
- 不新增 LLM 调用、不增加硬过滤、不改变接口、存储字段或 Milvus 表结构。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m unittest tests.test_qa_generation_contract
python -m py_compile qa/prompts/qa_generation_prompts.py qa/prompts/category_templates/*.py
git diff --check

docker exec qa-flow-runtime bash -lc \
  'cd /app && python -m unittest tests.test_qa_generation_contract && \
   python -m py_compile qa/prompts/qa_generation_prompts.py qa/prompts/category_templates/*.py'
curl http://localhost:12000/test-connection
```
