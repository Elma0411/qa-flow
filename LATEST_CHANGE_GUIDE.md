# Latest Change Guide

Updated: 2026-08-12 (Asia/Shanghai)

## Objective

Improve the naturalness of generated QA pairs after reviewing the persisted
LLM requests and responses from `integrated_document_task_1786534948`. The
target is a question a real reader would ask, rather than a source clause with
its latter half removed. `answer_explanation` should also be a complete
reader-facing clarification rather than a source-sentence fragment.

## What Changed

- `qa/prompts/qa_generation_prompts.py`
  - Replaced the long, overlapping candidate prompt with a shorter flow:
    identify one reader scenario and information need, then write one natural
    question.
  - Added semantic-compression examples for policy clauses. Full legal
    predicates and exact terms remain in `retrieval_query` and
    `must_have_terms`, not in `question`.
  - Corrected the input description from a single chunk to a primary source
    unit, which may contain tightly connected chunks in summary mode.
  - Defined `answer_explanation` as one or two complete reader-facing
    sentences that begin with a concrete subject or rule, rather than a vague
    reference; an explicit policy-domain positive/negative example anchors
    that distinction. Evidence provenance remains in `source_fact_text` and
    `evidence_usage`.
  - Explicitly forbids adding an unsupported amount, ratio, procedure,
    application step, authority, or deadline in either the answer or its
    explanation.
- `qa/prompts/category_templates/`
  - The six category templates now require scenario-first question design,
    preventing clause-prefix questions in policy, notice, standard, research,
    learning, and general documents.
- `qa/pipeline_runtime.py`
  - The default `candidate_multiplier` is now `1`, matching the requested QA
    count. This prevents over-sampling a source unit and encouraging the model
    to mine every clause for candidates. Callers can still explicitly set a
    larger multiplier when needed.
- `qa/generation/qa_generation_flow.py`
  - Calls the candidate input a primary source unit, matching the actual
    summary-unit behavior.
- `tests/test_qa_generation_contract.py`
  - Covers scenario-first prompts, semantic-compression examples, standalone
    explanations, primary source unit input, and the new default candidate
    multiplier.

## Confirmed Diagnosis

The old prompt was active in the reviewed task. Its persisted debug JSONL
already included the previous natural-language rules, and the Docker runtime
mounts this repository at `/app`. The weak wording originated in the model's
raw candidate responses, not from stale code or downstream rewriting.

The current Easy Dataset main branch uses a short question-writing workflow
with natural-question examples, while Ragas separates reader persona/theme
from question wording. This change applies those shared ideas without adding a
new LLM call or changing QA fields.

## Expected Behavior

```text
primary source unit
  -> scenario + information need
  -> natural question + precise retrieval plan
  -> evidence-grounded answer + reader-facing explanation
```

For example, a policy clause about insurance contribution reductions should
produce a question such as "农村独生子女或双女户家庭参加医保有什么优惠？",
while its retrieval fields retain the full eligibility and insurance terms.

## Validation

```bash
cd /data2/hjk/qa-flow

python -m unittest tests.test_qa_generation_contract
python -m py_compile \
  qa/prompts/qa_generation_prompts.py \
  qa/prompts/category_templates/*.py \
  qa/generation/qa_generation_flow.py \
  qa/pipeline_runtime.py
git diff --check

docker exec qa-flow-runtime bash -lc \
  'cd /app && python -m unittest tests.test_qa_generation_contract && \
   python -m py_compile qa/prompts/qa_generation_prompts.py \
   qa/prompts/category_templates/*.py qa/generation/qa_generation_flow.py \
   qa/pipeline_runtime.py'
curl --fail --silent --show-error http://localhost:12000/test-connection
```
