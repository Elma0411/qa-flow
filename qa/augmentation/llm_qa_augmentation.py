# 文件作用：调用大模型生成原始问答的扩展问法和答案变体。
# 关联说明：依赖 qa.common 和 prompts，作为 augmentation facade 的实现文件。

from openai import OpenAI
import csv
import re
import time
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from qa.common import build_language_instruction, detect_language, extract_first_choice_content
from qa.prompts.qa_augmentation_prompts import build_augment_prompt

# 导入应用配置，优先使用环境变量避免明文 Key
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from app.core.config import CONFIG as APP_CONFIG  # type: ignore
except Exception:
    APP_CONFIG = None

ENV_CONFIG = {
    "api_key": os.environ.get("LLM_API_KEY") or "",
    "base_url": os.environ.get("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/",
    "model": os.environ.get("LLM_MODEL") or "glm-4-flash",
    "max_retries": int(os.environ.get("LLM_MAX_RETRIES", "2") or 2),
}

CONFIG = APP_CONFIG or ENV_CONFIG

# 读取 CSV 文件中的问答对
def read_csv(file_path):
    qa_pairs = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "question" in row and "answer" in row:
                # 检查是否有主题字段
                theme = row.get("theme", "未分类")
                qa_pairs.append((row["question"], row["answer"], theme))
    return qa_pairs

# 构造 Prompt
def construct_prompt(question, answer, theme, augment_count=2):
    language_code = detect_language(f"{question} {answer} {theme}")
    language_instruction = build_language_instruction(language_code)
    return build_augment_prompt(
        question=question,
        answer=answer,
        theme=theme,
        question_type="简答题",
        augment_count=augment_count,
        language_instruction=language_instruction,
        language_code=language_code if language_code in ("zh", "en") else "zh",
    )

# 调用大模型 API
def call_api(prompt, idx, total, config=None):
    print(f"-> 数据项 {idx}/{total} 响应中……")
    start_time = time.time()

    if config is None:
        config = CONFIG

    try:
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "你是一个擅长生成问答数据的 AI 助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            stream=False,
            timeout=30  # 超时设置
        )

        duration = time.time() - start_time
        print(f"-> 数据项 {idx}/{total} 响应成功，生成耗时: {duration:.2f}s")
        return extract_first_choice_content(response)

    except Exception as e:
        print("——" * 10 + " 错误信息 " + "——" * 10)
        print(f"模型响应错误: {str(e)}")
        print(f"数据项 {idx}/{total}")
        print("——" * 25)
        return None

# 解析模型返回的文本，提取问答对
def extract_qa_pairs(response_text, idx, total):
    try:
        print(f"-> 数据项 {idx}/{total} 解析中……")
        pattern = r"\[QA_PAIR\]\s*Q:\s*(.*?)\s*A:\s*(.*?)(?:Theme:\s*(.*?))?(?=\[QA_PAIR\]|$)"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        # 处理提取结果
        parsed = []
        for q, a, t in matches:
            q = q.strip()
            a = a.strip()
            t = t.strip() if t else "未分类"  # 如果没有提取到主题，使用默认值
            parsed.append((q, a, t))
        
        count = len(parsed) - 1
        print(f"-> 数据项 {idx}/{total} 解析成功，获取增广数据 {count} 条")
        return parsed[1:]  # 跳过原始问答对
    except Exception as e:
        print("——" * 10 + " 错误信息 " + "——" * 10)
        print(f"正则解析失败: {str(e)}")
        print(f"数据项 {idx}/{total} 原始响应:\n" + response_text + "\n")
        print("——" * 25)
        return []

# 保存新问答对到 CSV
def save_to_csv(file_path, qa_pairs):
    with open(file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # 始终写入表头，包含主题字段
        writer.writerow(["question", "answer", "theme"])
        for q, a, t in qa_pairs:
            writer.writerow([q, a, t])
    print("-> 已保存至文件", file_path)

# 主流程
def main(input_csv, output_csv, augment_count=10):
    print("=" * 50)
    print(f"开始问答对数据增广流程 (每对增广数量: {augment_count})")
    print("=" * 50)

    print("\n步骤 1: 加载原始数据集")
    original_pairs = read_csv(input_csv)
    total = len(original_pairs)
    print(f"-> 已加载 {total} 个原始问答对")

    print("\n步骤 2: 模型调用")
    prompts = [construct_prompt(q, a, t, augment_count) for q, a, t in original_pairs]
    response_texts = []
    timeout_count = 0
    max_retries = CONFIG.get("max_retries", 2)

    for idx, prompt in enumerate(prompts, 1):
        print("-> 发送请求……")
        retry_count = 0
        response = None
        
        # 添加重试逻辑
        while retry_count < max_retries and response is None:
            if retry_count > 0:
                print(f"-> 第 {retry_count} 次重试...")
                time.sleep(2)  # 重试前等待2秒
            
            response = call_api(prompt, idx, total)
            retry_count += 1
            
        if response is None:
            timeout_count += 1
            response_texts.append(None)
            if timeout_count >= 2:
                print("——" * 10 + " 模型无响应两次，增广中止 " + "——" * 10)
                return
        else:
            response_texts.append(response)
    
    print("\n步骤 3: 数据清洗")
    print("-> 解析模型原始输出，获取增广问答对……")

    all_new_pairs = []
    for idx, response_text in enumerate(response_texts, 1):
        if response_text is None:
            print("——" * 10 + " 错误信息 " + "——" * 10)
            print("该数据项未收到模型响应，无法执行解析")
            print(f"数据项 {idx}/{total}")
            print("——" * 25)
            continue

        new_pairs = extract_qa_pairs(response_text, idx, total)
        all_new_pairs.extend(new_pairs)

    if all_new_pairs:
        print("\n步骤 4: 数据存储")
        print("-> 增广数据存储……")
        save_to_csv(output_csv, all_new_pairs)
        print(f"-> 总共生成 {len(all_new_pairs)} 条增广数据")
        return all_new_pairs
    else:
        print("\n没有生成有效的增广数据")
        return []

if __name__ == "__main__":
    main("input.csv", "output.csv")


# ----------------------- 新增：面向列表的增广（llmda4 功能） ----------------------- #


def _build_augment_prompt(
    question: str,
    answer: str,
    theme: str,
    question_type: str,
    augment_count: int,
    options: Optional[List[str]] = None,
    correct_option: Optional[str] = None,
) -> str:
    options_block = ""
    if question_type == "单选题" and options:
        opts = "\n".join(options)
        options_block = f"\n当前选项（必须保留数量与正确项一致）:\n{opts}\n正确选项: {correct_option or ''}\n"

    return f"""
你是一个智能助手，我会提供一个问答对及其主题，你的任务是生成多个类似的问答对，确保问答的核心语义不变，避免引入错误。请按如下格式输出：

[QA_PAIR]
Q: {question}
A: {answer}
Theme: {theme}
QuestionType: {question_type}
{options_block}

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
6. 使用简体中文，保持数字、单位、实体精确一致。
7. 返回 JSON 数组，元素形如 {{"question": "...", "answer": "..."}}，长度 = {augment_count}，不要返回 Markdown 代码块。
8. 常见可接受的改写方式：人称/视角改写，询问方式替换（如何/为什么/有哪些），语序调整，适度扩写或压缩，不改变事实。
"""


def _parse_augment_response(raw: str) -> List[Dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            for v in data.values():
                if isinstance(v, list):
                    return [d for d in v if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            return []
    return []


def _augment_single(
    client: OpenAI,
    qa: Dict[str, Any],
    augment_per_qa: int,
    model: str,
) -> List[Dict[str, Any]]:
    if augment_per_qa <= 0:
        return []

    parent_key = f"{qa.get('question','')}|||{qa.get('answer','')}"
    question = qa.get("question", "")
    answer = qa.get("answer", "")
    theme = qa.get("knowledge_category", "") or qa.get("theme", "")
    question_type = qa.get("question_type", "简答题")
    options = qa.get("options")
    correct_option = qa.get("correct_option")
    language_code = detect_language(f"{question} {answer} {theme}")
    language_instruction = build_language_instruction(language_code)

    prompt = build_augment_prompt(
        question=question,
        answer=answer,
        theme=theme,
        question_type=question_type,
        augment_count=augment_per_qa,
        options=options,
        correct_option=correct_option,
        language_instruction=language_instruction,
        language_code=language_code if language_code in ("zh", "en") else "zh",
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个擅长语义等价改写的问答增广助手。"},
                {"role": "user", "content": prompt},
            ],
            timeout=45,
        )
        content = extract_first_choice_content(resp)
        parsed = _parse_augment_response(content)
        results: List[Dict[str, Any]] = []
        seen_questions = set()
        for item in parsed:
            q_new = str(item.get("question") or "").strip()
            a_new = str(item.get("answer") or "").strip()
            if not q_new or not a_new:
                continue
            if q_new in seen_questions:
                continue
            seen_questions.add(q_new)
            augmented = dict(qa)
            augmented.update(
                {
                    "question": q_new,
                    "answer": a_new,
                    "is_primary": False,
                    "is_augmented": True,
                    "variant_of_key": parent_key,
                }
            )
            results.append(augmented)
            if len(results) >= augment_per_qa:
                break
        return results
    except Exception as exc:
        print(f"[augment] failed for question: {(qa.get('question') or '')[:40]!r}, err: {exc}")
        return []


def augment_qa_pairs(
    qa_list: List[Dict[str, Any]],
    augment_per_qa: int,
    client: OpenAI,
    model: str,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    对问答列表进行增广，保持题型/主题一致。返回增广后的条目列表。
    """
    if augment_per_qa <= 0 or not qa_list:
        return []

    results: List[Dict[str, Any]] = []
    total = len(qa_list)
    if progress_callback:
        try:
            progress_callback(
                {
                    "event": "start",
                    "total": total,
                    "augment_per_qa": augment_per_qa,
                }
            )
        except Exception:
            pass

    completed = 0
    total_augmented = 0

    def task(qa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return _augment_single(client, qa_item, augment_per_qa, model)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(task, qa): qa for qa in qa_list}
        for future in as_completed(future_map):
            completed += 1
            try:
                aug_list = future.result()
                if aug_list:
                    results.extend(aug_list)
                    total_augmented += len(aug_list)
                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "event": "item_completed",
                                "completed": completed,
                                "total": total,
                                "new_augmented": len(aug_list or []),
                                "total_augmented": total_augmented,
                            }
                        )
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[augment] worker exception: {exc}")
                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "event": "item_completed",
                                "completed": completed,
                                "total": total,
                                "new_augmented": 0,
                                "total_augmented": total_augmented,
                                "error": str(exc),
                            }
                        )
                    except Exception:
                        pass
                continue
    if progress_callback:
        try:
            progress_callback(
                {
                    "event": "done",
                    "total": total,
                    "total_augmented": total_augmented,
                }
            )
        except Exception:
            pass
    return results
