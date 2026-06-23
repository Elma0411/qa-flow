# 文件作用：构造问答增广使用的大模型提示词。
# 关联说明：被 augmentation/llm_qa_augmentation 调用，服务增广阶段。

from typing import Optional


def build_augment_prompt(
    question: str,
    answer: str,
    theme: str,
    question_type: str,
    augment_count: int,
    language_instruction: str,
    language_code: str = "zh",
    options: Optional[list] = None,
    correct_option: Optional[str] = None,
) -> str:
    options_block_zh = ""
    options_block_en = ""
    if question_type == "单选题" and options:
        opts = "\n".join(options)
        options_block_zh = f"\n当前选项（必须保留数量与正确项一致）:\n{opts}\n正确选项: {correct_option or ''}\n"
        options_block_en = f"\nOptions (keep count and correct option unchanged):\n{opts}\nCorrect option: {correct_option or ''}\n"

    if language_code == "en":
        return f"""
You are an assistant. Given a QA pair and its theme, generate multiple paraphrased QA pairs that keep the same core meaning and avoid errors. Follow this format:

[QA_PAIR]
Q: {question}
A: {answer}
Theme: {theme}
QuestionType: {question_type}
{options_block_en}

[QA_PAIR]
Q: [Paraphrased Question 1]
A: [Paraphrased Answer 1]
Theme: {theme}
QuestionType: {question_type}

[QA_PAIR]
Q: [Paraphrased Question 2]
A: [Paraphrased Answer 2]
Theme: {theme}
QuestionType: {question_type}

...

Requirements:
1. Keep the question meaning unchanged; wording can vary, but core semantics must stay the same.
2. Keep the answer meaning unchanged; rephrase is fine but do not add or change facts.
3. Preserve the theme; all generated QA pairs must stay within the same topic.
4. Generate at least {augment_count} new QA pairs with stylistic diversity but equivalent meaning.
5. Keep the question type unchanged: {question_type}. If single choice, keep the same options and correct option; if True/False, answer must be “True” or “False”.
6. {language_instruction}
7. Return a JSON array of length {augment_count}, items like {{"question": "...", "answer": "..."}}, no Markdown code blocks.
8. Acceptable paraphrases: person/voice changes, different interrogatives (how/why/what), word order tweaks, concise vs. slightly extended, without altering facts.
"""

    return f"""
你是一个智能助手，我会提供一个问答对及其主题，你的任务是生成多个类似的问答对，确保问答的核心语义不变，避免引入错误。请按如下格式输出：

[QA_PAIR]
Q: {question}
A: {answer}
Theme: {theme}
QuestionType: {question_type}
{options_block_zh}

[QA_PAIR]
Q: [变换后的问题1]
A: [变换后的回答1]
Theme: {theme}
QuestionType: {question_type}

[QA_PAIR]
Q: [变换后的问题2]
A: [变换后的回答2]
Theme: {theme}
QuestionType: {question_type}

...

请确保：
1. 问题保持原意，可以调整表述方式，但不得改变核心语义。
2. 答案保持原意，可以换个说法或表达，但不得引入错误信息或新事实。
3. 保留原主题，所有生成的问答对应该与原始问答对属于同一主题。
4. 生成不少于 {augment_count} 组新的问答对，确保它们在语言风格上有所不同，但仍能传达相同的含义。
5. 题型保持不变：{question_type}。若为单选题，必须保留相同数量的选项和同一个正确选项；若为判断题，答案只能是“正确”或“错误”。
6. {language_instruction}
7. 返回 JSON 数组，元素形如 {{"question": "...", "answer": "..."}}, 长度 = {augment_count}，不要返回 Markdown 代码块。
8. 常见可接受的改写方式：人称/视角改写，询问方式替换（如何/为什么/有哪些），语序调整，适度扩写或压缩，不改变事实。
"""


__all__ = ["build_augment_prompt"]
