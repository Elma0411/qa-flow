# 文件作用：定义 LLM 问答质量评估的系统提示词和指标提示词。
# 关联说明：被 qa_evaluation/llm_quality_evaluator 调用，服务评价阶段。

from typing import Dict, Literal

SUPPORTED_METRICS = (
    "relevance",
    "completeness",
    "accuracy",
    "reasonableness",
    "agnosticism",
)


def get_evaluation_prompts(
    language: Literal["zh", "en"] = "zh",
    language_instruction: str = "",
) -> Dict[str, str]:
    """返回评估提示集合，支持中英文。"""
    lang = "en" if language == "en" else "zh"

    if lang == "en":
        lang_hint = language_instruction or "Keep all outputs (scores and reasons) fully in English."
        return {
            "relevance": f"""[Question Relevance Evaluation] Given an article and a derived question, rate how relevant the question is to the article (0-1, decimals allowed) and explain briefly.

Article: {{content}}
Question: {{target}}
Related source text: {{source_text}}

Scoring notes:
- Scores range from 0 to 1; higher means more relevant.
- Prefer two-decimal scores (e.g., 0.73, 0.88); avoid always using the same values or extreme 0.0/0.5/1.0.
- First decide a band (A/B/C/D), then pick a score in that band:
  - Band A (highly relevant): 0.90–0.99, only when the question targets the core content with almost no issues.
  - Band B (quite relevant): 0.70–0.89, generally consistent but with minor drift or missing details.
  - Band C (partially relevant): 0.40–0.69, only touches part of the article or is too generic/background-only.
  - Band D (barely relevant): 0.00–0.39, mostly off-topic.
- Avoid clustering most cases above 0.80; >0.85 only when nearly flawless.

Return format:
- In reasons, explicitly note the band (A/B/C/D), then explain.
- Return a JSON array: [{{"score": 0.0-1.0, "reasons": "English explanation"}}], score to two decimals.
- {lang_hint}""",
            "completeness": f"""[Answer Completeness Evaluation] Given an article and a QA pair, rate how completely the answer covers what the question asks (0-1, decimals allowed) and explain briefly.

Article: {{content}}
QA pair to evaluate: {{target}}
Related source text: {{source_text}}

Scoring notes:
- 0 to 1; higher means the answer covers more required aspects.
- Prefer two-decimal scores (e.g., 0.72, 0.91); avoid repetitive or extreme 0.0/0.5/1.0 values.
- First pick a band (A/B/C/D), then choose a score:
  - Band A (high completeness): 0.90–0.99, covers all key points and necessary details; use sparingly.
  - Band B (fairly complete): 0.70–0.89, main points covered but some minor details missing.
  - Band C (partially complete): 0.40–0.69, only part of the question is answered or major aspects are missing.
  - Band D (mostly incomplete): 0.00–0.39, most key information is absent.
- For average answers, prefer 0.40–0.80; only go above 0.85 when there are almost no omissions.

Return format:
- State the band (A/B/C/D) in reasons and give the rationale.
- Return a JSON array: [{{"score": 0.0-1.0, "reasons": "English explanation"}}], score to two decimals.
- {lang_hint}""",
            "accuracy": f"""[Answer Factual Accuracy] Given an article and a QA pair derived from it, rate how factually accurate the answer is (0-1, decimals allowed) and explain.

Article: {{content}}
QA pair to verify: {{target}}
Related source text: {{source_text}}

Scoring notes:
- 0 to 1; higher means closer to the source with fewer factual errors.
- Use fine-grained decimals (e.g., 0.68, 0.94); avoid repeating extremes like 0.8/0.9.
- First decide a factual band (A/B/C/D), then pick a score:
  - Band A (highly accurate): 0.90–0.99, fully consistent with the source; no substantial errors; use sparingly.
  - Band B (mostly accurate): 0.70–0.89, generally correct but with minor phrasing issues or missing conditions.
  - Band C (partially accurate): 0.40–0.69, contains correct bits but also clear omissions/misinterpretations.
  - Band D (inaccurate): 0.00–0.39, contradicts main conclusions or largely mismatched.
- If issues exist but not catastrophic, prefer Band C (0.40–0.69) instead of forcing >0.80.

Return format:
- In reasons, name the band (A/B/C/D) and specify key errors or gaps.
- Return a JSON array: [{{"score": 0.0-1.0, "reasons": "English explanation"}}], score to two decimals.
- {lang_hint}""",
            "reasonableness": f"""[Logical Reasonableness] Given an article and a statement/answer, rate how logically consistent it is with the article (0-1, decimals allowed) and explain.

Article: {{content}}
Answer to evaluate: {{target}}
Related source text: {{source_text}}

Scoring notes:
- 0 to 1; higher means more self-consistent and aligned with the article.
- Use two-decimal scores (e.g., 0.64, 0.92); avoid repeatedly choosing 0.80/0.90.
- First pick a band (A/B/C/D), then fine-tune:
  - Band A (rigorous): 0.90–0.99, fully consistent reasoning; no obvious gaps; use sparingly.
  - Band B (mostly reasonable): 0.70–0.89, generally sound, with minor leaps.
  - Band C (somewhat reasonable): 0.40–0.69, has clear gaps or weak causality.
  - Band D (weak/contradictory): 0.00–0.39, contains obvious logical errors or conflicts.
- For partial logic or noticeable jumps, prefer Band C (0.40–0.69); do not give >0.85 lightly.

Return format:
- In reasons, state the band (A/B/C/D) and key logical evidence.
- Return a JSON array: [{{"score": 0.0-1.0, "reasons": "English explanation"}}], score to two decimals.
- {lang_hint}""",
            "agnosticism": f"""[Question Clarity Evaluation] Evaluate the clarity of the question based only on its wording (no external context). Return 0-1 with explanation.

Question to evaluate: {{target}}

Scoring notes:
- 0 to 1; higher means clearer and less ambiguous.
- Use two-decimal scores (e.g., 0.69, 0.93); avoid repeating values or extremes.
- First decide a clarity band (A/B/C/D), then select a score:
  - Band A (very clear): 0.90–0.99, fully understandable without context; use sparingly.
  - Band B (quite clear): 0.70–0.89, mostly clear with minor ambiguities.
  - Band C (somewhat clear): 0.40–0.69, understandable but depends on context or has evident vagueness.
  - Band D (unclear): 0.00–0.39, confusing or easily misread.
- If the question only makes sense with specific context, lean toward Band C or below.

Return format:
- In reasons, state the band (A/B/C/D) and give the detailed rationale.
- Return a JSON array: [{{"score": 0.0-1.0, "reasons": "English explanation"}}], score to two decimals.
- {lang_hint}""",
        }

    lang_hint = language_instruction or "请使用简体中文输出评分理由，避免英文或中英混杂。"
    return {
        "relevance": f"""【问题相关性评估】给定一篇文章和由此产生的问题，评估问题与该文章的相关性，并返回0-1分(可包含小数)，并说明为什么要给这个分数。

文章内容：{{content}}
需评估问题：{{target}}
相关原文内容：{{source_text}}

评分说明：
- 评分范围为0到1，分数越高表示问题与文章相关性越强
- 请优先使用带两位小数的分数（如0.73、0.88），避免总是使用相同的分数（例如一直给0.80或0.90），也不要频繁给出0.0/0.5/1.0这样的极端值
- 建议先根据直觉将该案例划分为四档之一（A/B/C/D），然后在对应区间内选择具体得分：
  - 等级A（高度相关）：0.90–0.99，仅在问题几乎完全针对文章核心内容且几乎没有任何明显问题时使用，应少量使用
  - 等级B（比较相关）：0.70–0.89，问题与文章总体一致，但存在轻微偏离或未覆盖部分细节
  - 等级C（部分相关）：0.40–0.69，问题只涉及文章的一部分、过于笼统，或只与背景内容有关
  - 等级D（几乎无关）：0.00–0.39，问题与文章核心内容基本无关或明显跑题
- 在一组典型样本中，请避免让大多数案例落在0.80以上，只有在你认为几乎没有明显问题时才给出0.85以上的分数

返回格式要求：
- 请先在reasons中用简体中文明确写出该案例属于哪一档（A/B/C/D），再解释具体原因
- 返回JSON数组：[{{"score":0.0-1.0,"reasons":"用简体中文详细说明为什么给这个分数，并说明属于哪一档（A/B/C/D）"}}]，score建议精确到小数点后两位
- {lang_hint}""",
        "completeness": f"""【答案完整性评估】给定一篇文章和由其生成的问题答案对，评估答案的完整性，并返回0-1分(可包含小数)，表示答案在多大程度上充分利用了文章中的信息，包括所有子问。同时给出分配分数的原因。

文章内容：{{content}}
需评估QA对：{{target}}
相关原文内容：{{source_text}}

评分说明：
- 评分范围为0到1，分数越高表示答案越完整地覆盖了问题的各个方面
- 请优先使用带两位小数的分数（如0.72、0.91），避免总是使用相同的分数或只用0.0/0.5/1.0
- 建议先判断答案大致属于哪一档完整程度（A/B/C/D），再在对应区间内选择具体得分：
  - 等级A（高度完整）：0.90–0.99，答案覆盖了问题的所有要点及必要细节，没有明显遗漏，应少量使用
  - 等级B（比较完整）：0.70–0.89，主要要点已经覆盖，但缺少部分次要细节或说明
  - 等级C（部分完整）：0.40–0.69，只回答了问题的一部分，或忽略了明显的重要方面
  - 等级D（基本不完整）：0.00–0.39，大部分关键信息缺失，难以支持有效理解
- 对于一般质量的答案，请优先考虑打在0.40–0.80之间，只有在你认为几乎没有明显遗漏时才进入0.85以上的高分区间

返回格式要求：
- 请在reasons中明确写出该答案属于哪一档（A/B/C/D），并解释你的判断依据
- 返回JSON数组：[{{"score":0.0-1.0,"reasons":"用简体中文详细说明为什么给这个分数，并说明属于哪一档（A/B/C/D）"}}]，score建议精确到小数点后两位
- {lang_hint}""",
        "accuracy": f"""【答案准确性验证】给定一篇文章和从这篇文章中生成的问题答案对，评估答案的准确性，并返回0-1分(可包含小数)，表明答案是否准确地从文章中提取，并给出为什么分配这个分数的原因。这包括检查文本中任何声明或陈述的准确性，并验证它们是否有证据支持。

文章内容：{{content}}
需验证QA对：{{target}}
相关原文内容：{{source_text}}

评分说明：
- 评分范围为0到1，分数越高表示答案与原文内容越一致、事实错误越少
- 请使用细粒度的小数分数（如0.68、0.94），避免总是给出极端或重复的分数，尤其避免总是给0.8/0.9
- 建议先判断答案在事实层面大致属于哪一档（A/B/C/D），再在对应区间选择具体分数：
  - 等级A（高度准确）：0.90–0.99，答案与原文高度一致，没有实质性错误或歧义，应少量使用
  - 等级B（基本准确）：0.70–0.89，整体正确，但存在轻微表述偏差、缺少限定条件等小问题
  - 等级C（部分准确）：0.40–0.69，包含正确信息，但与原文相比有明显遗漏、误解或混杂了不完全可靠的内容
  - 等级D（严重不准确）：0.00–0.39，答案与原文主要结论矛盾或大量内容与原文不符
- 对于存在明显问题但不至于完全错误的答案，请优先打在C档（0.40–0.69），不要勉强给到0.80以上

返回格式要求：
- 请在reasons中明确写出该答案属于哪一档（A/B/C/D），并用简体中文说明具体错误或偏差
- 返回JSON数组：[{{"score":0.0-1.0,"reasons":"用简体中文详细说明为什么给这个分数，并说明属于哪一档（A/B/C/D）"}}]，score建议精确到小数点后两位
- {lang_hint}""",
        "reasonableness": f"""【逻辑合理性评估】给定一篇文章和陈述，评估陈述相对于文章的合理性，并返回0-1分(可包含小数)，表明内容在逻辑上是如何一致的，没有明显的矛盾，并提供分配分数的原因。

文章内容：{{content}}
需评估答案：{{target}}
相关原文内容：{{source_text}}

评分说明：
- 评分范围为0到1，分数越高表示答案在逻辑上越自洽、与原文越一致
- 请使用两位小数的分数（如0.64、0.92），避免总是选择0.80/0.90或其他少数几个固定值
- 建议先判断答案逻辑上大致处于哪一档（A/B/C/D），再在对应区间内微调得分：
  - 等级A（逻辑严谨）：0.90–0.99，答案内部逻辑完全一致，与原文推理链条吻合，没有明显漏洞，应少量使用
  - 等级B（基本合理）：0.70–0.89，整体推理合理，但存在部分未完全展开或略显跳跃的地方
  - 等级C（逻辑一般）：0.40–0.69，包含某些合理推断，但也存在明显的逻辑空缺、跳步或弱因果关系
  - 等级D（逻辑薄弱/矛盾）：0.00–0.39，答案中包含明显逻辑错误、自相矛盾或与原文推理冲突
- 对于只部分合理或存在明显跳步的情况，请优先使用C档（0.40–0.69），不要轻易打到0.85以上

返回格式要求：
- 请在reasons中明确写出该答案属于哪一档（A/B/C/D），并说明关键的逻辑依据
- 返回JSON数组：[{{"score":0.0-1.0, "reasons": "用简体中文详细说明为什么给这个分数，并说明属于哪一档（A/B/C/D）"}}]，score建议精确到小数点后两位
- {lang_hint}""",
        "agnosticism": f"""【问题清晰度评估】请只根据问题本身的表述来评估其是否清晰、易懂、指向明确，而不要依赖任何原文或外部上下文。请给出0-1分的评分。

需评估问题：{{target}}

评分说明：
- 评分范围为0到1，分数越高表示问题本身的表述越清晰、歧义越少
- 请使用两位小数的分数（如0.69、0.93），避免总是给出相同的分数或使用极端值
- 建议先判断问题大致处于哪一档清晰度（A/B/C/D），再在对应区间内选择具体分数：
  - 等级A（非常清晰）：0.90–0.99，即使完全不知道上下文，读者也能准确理解问题的对象、条件和目标，应少量使用
  - 等级B（比较清晰）：0.70–0.89，整体表达清楚，但存在少量可以改进的表述或轻微歧义
  - 等级C（一般清晰）：0.40–0.69，大致能看出在问什么，但依赖一定的背景知识或存在明显模糊点
  - 等级D（不清晰）：0.00–0.39，离开上下文几乎无法判断问题指向，或表述混乱、容易误解
- 对于只在特定上下文下才说得通的问题，请优先考虑打在C档或以下

返回格式要求：
- 请评估问题清晰度，并在reasons中明确标注属于哪一档（A/B/C/D），再给出详细原因
- 返回JSON数组：[{{"score": 0.0-1.0, "reasons": "用简体中文详细说明为什么给这个分数，并说明属于哪一档（A/B/C/D）"}}]，score建议精确到小数点后两位
- {lang_hint}""",
    }


def get_system_prompt(
    language: Literal["zh", "en"] = "zh",
    language_instruction: str = "",
) -> Dict[str, str]:
    """返回评估系统提示，支持语言切换。"""
    lang = "en" if language == "en" else "zh"
    if lang == "en":
        lang_hint = language_instruction or "Keep the reasons in English and stay consistent with the question language."
        return {
            "role": "system",
            "content": (
                "You are a professional evaluation expert. Reply with ONE JSON object only, no markdown/text outside JSON.\n"
                'JSON fields: {{"score": <0-1 float>, "reasons": "<brief>"}}; if you cannot evaluate, return score=0 with a reason.\n'
                "- Stay concise; focus on the provided content; do not fabricate.\n"
                "- Prefer two-decimal scores (e.g., 0.73); avoid always using the same value or extremes.\n"
                f"- {lang_hint}"
            ),
        }

    lang_hint = language_instruction or "无论原文或问题使用何种语言，你输出的 reasons 必须使用简体中文，避免英文或中英混杂。"
    return {
        "role": "system",
        "content": (
            "你是一个专业的评估专家。只返回一个 JSON 对象，禁止输出除 JSON 之外的任何文字/Markdown。\n"
            'JSON 字段：{{"score":0-1 浮点数,"reasons":"简要原因"}}；如无法评估，返回 score=0 并给出原因。\n'
            "- 精简输出，专注提供内容，不要编造。\n"
            "- 建议使用两位小数（如 0.73），避免总用相同分值或极端值。\n"
            f"- {lang_hint}"
        ),
    }


__all__ = ["get_evaluation_prompts", "get_system_prompt", "SUPPORTED_METRICS"]
