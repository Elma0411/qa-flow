# 文件作用：调用 LLM 对问答质量进行多维度评分。
# 关联说明：调用 prompts/llm_quality_evaluation_prompts，是 LLM 评价主入口。

import json
import time
import re
import os
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from qa.common import build_language_instruction, detect_language
from qa.prompts.llm_quality_evaluation_prompts import (
    SUPPORTED_METRICS,
    get_evaluation_prompts,
    get_system_prompt,
)
from app.services.llm import VLMClientConfig, create_vlm_client, get_llm_client_pool, LLMClientConfig

# 默认配置（优先使用环境变量注入，避免明文 Key）
ENV_API_KEY = os.environ.get("LLM_API_KEY") or ""
ENV_BASE_URL = os.environ.get("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"
ENV_MODEL = os.environ.get("LLM_MODEL") or "glm-4-flash"

DEFAULT_CONFIG = {
    "api_key": ENV_API_KEY,
    "base_url": ENV_BASE_URL,
    "model": ENV_MODEL,
    "input_text": "1.1.txt",                 # 原始文本文件
    "qa_file": "qa_pairs1.json",              # QA对文件
    "output_file": "evaluated_results.json", # 评估结果输出文件
    "max_retries": 2,                        # 最大重试次数
    "batch_size": 1,                         # 批处理大小
    "request_timeout": 120,                  # API请求超时时间
    "text_truncate": 2500                    # 文本截断长度
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估问答对的质量")
    
    parser.add_argument("--input-text", "-t", type=str, 
                        help=f"原始文本文件路径 (默认: {DEFAULT_CONFIG['input_text']})")
    
    parser.add_argument("--qa-file", "-q", type=str, 
                        help=f"问答对文件路径 (默认: {DEFAULT_CONFIG['qa_file']})")
    
    parser.add_argument("--output", "-o", type=str, 
                        help=f"评估结果文件路径 (默认: {DEFAULT_CONFIG['output_file']})")
    
    parser.add_argument("--api-key", type=str, 
                        help=f"API密钥 (默认: {DEFAULT_CONFIG['api_key']})")
    
    parser.add_argument("--base-url", type=str, 
                        help=f"API基础URL (默认: {DEFAULT_CONFIG['base_url']})")
    
    parser.add_argument("--model", type=str, 
                        help=f"使用的模型 (默认: {DEFAULT_CONFIG['model']})")
    
    parser.add_argument("--max-retries", type=int, 
                        help=f"API调用最大重试次数 (默认: {DEFAULT_CONFIG['max_retries']})")
    
    parser.add_argument("--timeout", type=int, 
                        help=f"API请求超时时间 (默认: {DEFAULT_CONFIG['request_timeout']}秒)")
    
    parser.add_argument("--text-truncate", type=int, 
                        help=f"文本截断长度 (默认: {DEFAULT_CONFIG['text_truncate']}字)")
    
    return parser.parse_args()


def load_config(args):
    """加载配置，优先级：命令行参数 > 环境变量 > 默认值"""
    config = DEFAULT_CONFIG.copy()
    
    # 从环境变量加载API密钥
    if os.environ.get("LLM_API_KEY"):
        config["api_key"] = os.environ.get("LLM_API_KEY")
    
    # 从命令行参数更新配置
    if args.input_text:
        config["input_text"] = args.input_text
    if args.qa_file:
        config["qa_file"] = args.qa_file
    if args.output:
        config["output_file"] = args.output
    if args.api_key:
        config["api_key"] = args.api_key
    if args.base_url:
        config["base_url"] = args.base_url
    if args.model:
        config["model"] = args.model
    if args.max_retries:
        config["max_retries"] = args.max_retries
    if args.timeout:
        config["request_timeout"] = args.timeout
    if args.text_truncate:
        config["text_truncate"] = args.text_truncate
    
    # 验证必要参数
    if not config["api_key"]:
        raise ValueError("API密钥未设置。请通过--api-key参数或LLM_API_KEY环境变量提供")
    
    # 确保输入文件存在
    if not os.path.exists(config["input_text"]):
        raise FileNotFoundError(f"原始文本文件 '{config['input_text']}' 不存在")
    
    if not os.path.exists(config["qa_file"]):
        raise FileNotFoundError(f"问答对文件 '{config['qa_file']}' 不存在")
    
    return config


def test_api_connection(client, model: str):
    """API连接测试"""
    try:
        result = client.create_chat_completion_text(
            model=model,
            messages=[{"role": "user", "content": "请回复'OK'"}],
            temperature=0.0,
            max_tokens=5,
            timeout=15,
        ).strip()
        print(f"API连接测试: {result}")
        return True
    except Exception as e:
        print(f"连接测试失败: {str(e)}")
        return False


def load_text(file_path: str) -> str:
    """加载原始文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] 加载文本失败: {str(e)}")
        exit(1)


def load_qa_pairs(file_path: str) -> List[Dict]:
    """加载QA对数据，支持JSON和CSV格式"""
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            # 从JSON文件加载
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_ext == '.csv':
            # 从CSV文件加载
            qa_pairs = []
            import csv
            
            # 尝试不同的编码方式读取文件
            encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
            for encoding in encodings:
                try:
                    # 先尝试读取文件前几行，检查格式
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        # 计算文件行数
                        line_count = 0
                        with open(file_path, 'r', encoding=encoding) as count_f:
                            line_count = sum(1 for _ in count_f)
                        # 重新打开文件并读取样本行
                        f.seek(0)
                        sample_lines = [next(f) for _ in range(min(3, line_count))]
                    
                    # 检查是否有列头
                    has_header = False
                    if 'question' in sample_lines[0].lower() and 'answer' in sample_lines[0].lower():
                        has_header = True
                    
                    # 根据文件格式选择不同的加载方式
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        if has_header:
                            # 有列头的标准CSV格式
                            reader = csv.DictReader(f)
                            for row in reader:
                                if "question" in row and "answer" in row:
                                    qa_pairs.append({
                                        "question": row["question"],
                                        "answer": row["answer"],
                                        "source_fact": row.get("source_fact", "")
                                    })
                        else:
                            # 无列头的简单CSV格式（假设第一列是问题，第二列是答案）
                            reader = csv.reader(f)
                            for row in reader:
                                if len(row) >= 2:
                                    qa_pairs.append({
                                        "question": row[0].strip(),
                                        "answer": row[1].strip(),
                                        "source_fact": "" if len(row) < 3 else row[2].strip()
                                    })
                    
                    if not qa_pairs:
                        raise ValueError("CSV文件中未找到有效的问答对")
                    
                    print(f"成功从CSV加载了 {len(qa_pairs)} 个问答对 (使用编码: {encoding})")
                    return qa_pairs
                    
                except UnicodeDecodeError:
                    # 如果当前编码不适用，尝试下一个
                    print(f"尝试使用 {encoding} 编码失败，尝试下一种编码...")
                    continue
                except Exception as e:
                    # 如果是其他错误，直接抛出
                    raise e
            
            # 如果所有编码都尝试失败
            raise ValueError(f"无法使用任何支持的编码读取CSV文件: {encodings}")
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持JSON和CSV格式")
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON解析错误: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"[ERROR] 加载QA对失败: {str(e)}")
        exit(1)


def resolve_prompt_language(question: str, answer: str, source_text: str) -> str:
    """根据问答及来源文本选择提示语种，默认回落到中文。"""
    parts = [question or "", answer or ""]
    if source_text and source_text != "未提供相关原文内容":
        parts.append(source_text)
    combined = " ".join(parts).strip()
    language_code = detect_language(combined)
    return language_code if language_code in ("zh", "en") else "zh"


def get_prompt_params(metric: str, content: str, target_data: Dict, text_truncate: int) -> dict:
    """生成提示模板参数"""
    # 提取source_text
    source_text = target_data.get("source_text", "未提供相关原文内容")
    
    # 从target_data中移除source_text，避免在JSON中重复
    target_data_copy = target_data.copy()
    if "source_text" in target_data_copy:
        del target_data_copy["source_text"]
    
    return {
        "content": content[:text_truncate],
        "target": json.dumps(target_data_copy, ensure_ascii=False),
        "source_text": source_text
    }


def parse_evaluation(raw: str) -> dict:
    """增强版解析函数：处理多评分项并保留最低分及对应原因"""

    def extract_valid_items(data: Any) -> List[Dict]:
        """递归提取有效评分项"""
        items = []
        if isinstance(data, dict):
            # 直接检查评分项可能的键名
            score_keys = ['score', 'Score', '分数', 'score_value', 'scoreValue']
            reason_keys = ['reasons', 'reason', '原因', 'explanation', 'comment']
            
            # 查找评分值
            score = None
            for key in score_keys:
                if key in data:
                    try:
                        # 处理可能的字符串形式的分数
                        value = data[key]
                        if isinstance(value, str):
                            # 尝试从字符串中提取数字
                            match = re.search(r'(\d+(\.\d+)?)', value)
                            if match:
                                score = float(match.group(1))
                        else:
                            score = float(value)
                        break
                    except (ValueError, TypeError):
                        pass
            
            # 查找原因
            reason = None
            for key in reason_keys:
                if key in data and data[key]:
                    reason = str(data[key])
                    break
            
            # 如果找到了分数和原因，添加到结果中
            if score is not None:
                items.append({
                    'score': max(0.0, min(1.0, score)),  # 确保分数在0-1范围内
                    'reasons': (reason or "未提供原因")[:500]
                })
            
            # 处理可能的嵌套结构
            for k, v in data.items():
                # 特殊处理常见的嵌套结构
                if k in ['result', 'results', 'evaluation', 'scores', 'assessments']:
                    items.extend(extract_valid_items(v))
                # 处理列表值
                elif isinstance(v, (list, dict)):
                    items.extend(extract_valid_items(v))
                
        elif isinstance(data, list):
            for item in data:
                items.extend(extract_valid_items(item))
        return items

    try:
        # 第一阶段：原始响应清理
        cleaned = re.sub(r'\\(?!u)', '', raw)  # 移除无效转义符
        cleaned = re.sub(r'(?<!\\)\\n', '', cleaned)  # 移除换行符

        # 如果响应包含多个JSON结构，尝试提取它们
        json_structures = []
        
        # 首先尝试直接解析整个响应
        try:
            parsed = json.loads(cleaned)
            json_structures.append(parsed)
        except json.JSONDecodeError:
            # 如果整体解析失败，尝试使用正则表达式提取JSON结构
            patterns = [
                r'\[.*\]',  # 匹配数组
                r'{.*}',  # 匹配对象
                r'"score"\s*:\s*\d+\.?\d*'  # 匹配分数特征，支持整数和小数
            ]

            # 提取所有可能的JSON结构
            for pattern in patterns:
                for match in re.finditer(pattern, cleaned, re.DOTALL):
                    try:
                        text = match.group(0)
                        # 如果找到的是独立的score键值对，包装成完整的对象
                        if text.startswith('"score"'):
                            text = '{' + text + '}'
                        parsed = json.loads(text)
                        json_structures.append(parsed)
                    except:
                        continue

        # 如果没有找到任何JSON结构，尝试通过正则表达式直接提取分数
        if not json_structures:
            # 尝试直接从文本中提取分数
            score_matches = re.findall(r'(\b(?:分数|得分|评分|score)\b\s*[:：]\s*)(\d+\.?\d*)', cleaned, re.IGNORECASE)
            reason_matches = re.findall(r'(\b(?:原因|理由|说明|reason)\b\s*[:：]\s*)([^.,。，\n]+)', cleaned, re.IGNORECASE)
            
            if score_matches:
                score = float(score_matches[0][1])
                reason = reason_matches[0][1] if reason_matches else "未提供原因"
                json_structures.append({"score": score, "reason": reason})

        # 第三阶段：递归提取评分项
        all_items = []
        for struct in json_structures:
            all_items.extend(extract_valid_items(struct))

        # 第四阶段：智能选择最低分
        if not all_items:
            raise ValueError("未找到有效评分项")

        # 按分数排序，保留所有最低分项
        min_score = min(item['score'] for item in all_items)
        avg_score = sum(item["score"] for item in all_items) / len(all_items)
        candidates = []
        for item in all_items:
            candidates.append(
                (
                    abs(item["score"] - avg_score),
                    -len(item.get("reasons", "")),
                    item,
                )
            )
        candidates.sort(key=lambda x: (x[0], x[1]))
        selected = candidates[0][2]

        # 第五阶段：结果处理
        clean_reasons = re.sub(
            r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）【】…—《》\s]',
            ' ',
            selected.get('reasons', '')
        ).strip()[:500]

        return {
            # 保留模型原始分数，不做四舍五入，只做0-1范围截断
            "score": max(0.0, min(1.0, avg_score if avg_score is not None else selected["score"])),
            "reasons": clean_reasons or "原因未提供"
        }

    except Exception as e:
        error_msg = f"解析错误：{str(e)}"
        raw_snippet = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。]', '', raw[:200])
        return {
            "score": 0.0,  # 错误情况下给0分
            "reasons": f"{error_msg} | 原始内容：{raw_snippet}"[:500]
        }


def _format_prompt_safe(template: str, params: dict) -> str:
    """
    在保留 {content}/{target}/{source_text} 占位符的前提下，
    先转义其他花括号，再执行 format，避免 "score" 等键被误当占位符。
    """
    safe = template
    placeholders = ["content", "target", "source_text"]
    for key in placeholders:
        safe = safe.replace(f"{{{key}}}", f"@@{key}@@")
    safe = safe.replace("{", "{{").replace("}", "}}")
    for key in placeholders:
        safe = safe.replace(f"@@{key}@@", f"{{{key}}}")
    return safe.format(**params)


def evaluate_metric(client, metric: str, paper_text: str, qa_pair: Dict, config: dict) -> Dict:
    """评估单个指标，进行多次评估并取平均分"""
    raw_history: List[str] = []
    try:
        # 为当前 QA 对选择可用的原文片段（优先使用流水线生成时附带的字段）
        source_text = (
            qa_pair.get("qa_generation_unit_text")
            or qa_pair.get("source_fact_text")
            or qa_pair.get("source_fact")
            or qa_pair.get("source")
            or qa_pair.get("source_text")
            or "未提供相关原文内容"
        )

        # 为指标创建评估输入
        evaluation_inputs = {
            "relevance": {
                "question": qa_pair["question"],
                "source_text": source_text,
            },
            "completeness": {
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "source_text": source_text,
            },
            "accuracy": {
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "source_text": source_text,
            },
            "reasonableness": {
                "answer": qa_pair["answer"],
                "source_text": source_text,
            },
            "agnosticism": {"question": qa_pair["question"]},
        }

        prompt_language = resolve_prompt_language(
            qa_pair.get("question", ""),
            qa_pair.get("answer", ""),
            source_text,
        )
        language_instruction = build_language_instruction(prompt_language)
        system_prompt = get_system_prompt(prompt_language, language_instruction)
        evaluation_prompts = get_evaluation_prompts(prompt_language, language_instruction)
        params = get_prompt_params(
            metric, paper_text, evaluation_inputs[metric], config["text_truncate"]
        )
        prompt = _format_prompt_safe(evaluation_prompts[metric], params)

        evaluation_results = []
        evaluation_count = 2
        for eval_attempt in range(evaluation_count):
            raw_response = None
            for attempt in range(config["max_retries"]):
                try:
                    print(
                        f"评估 {metric} 中... (第{eval_attempt+1}/{evaluation_count}次评估，尝试 {attempt+1}/{config['max_retries']})"
                    )
                    raw_response = client.create_chat_completion_text(
                        model=config["model"],
                        messages=[system_prompt, {"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=2000,
                        timeout=float(config["request_timeout"]),
                    )
                    break
                except Exception as e:
                    print(f"调用异常: {str(e)}")
                    time.sleep(2)
            if raw_response:
                raw_history.append(raw_response)
                try:
                    parsed_result = parse_evaluation(raw_response)
                except Exception as parse_exc:
                    print(
                        f"[eval][{metric}] 解析异常: {parse_exc}, raw: {(raw_response or '')[:400]}"
                    )
                    parsed_result = {
                        "score": 0.0,
                        "reasons": f"解析异常: {parse_exc}",
                        "_raw_fragment": (raw_response or "")[:1000],
                    }
                parsed_result["_raw"] = raw_response
                parsed_result["_attempt"] = eval_attempt + 1
                if "score" not in parsed_result:
                    print(
                        f"[eval][{metric}] 未找到 score 字段，raw: {(raw_response or '')[:400]}"
                    )
                    parsed_result["score"] = 0.0
                evaluation_results.append(parsed_result)
            else:
                evaluation_results.append(
                    {
                        "score": 0.0,
                        "reasons": f"评估失败，无法获取响应 (第{eval_attempt+1}次评估)，metric={metric}",
                        "_raw": "",
                        "_attempt": eval_attempt + 1,
                    }
                )
            if eval_attempt < evaluation_count - 1:
                time.sleep(0.5)

        sanitized_results = []
        for r in evaluation_results:
            score_val = r.get("score", 0.0)
            try:
                score_val = float(score_val)
            except Exception:
                print(
                    f"[eval][{metric}] score 解析失败，置为 0，原值: {r.get('score')}, raw: {r.get('_raw','')[:200]}"
                )
                score_val = 0.0
            r["score"] = score_val
            sanitized_results.append(r)

        valid_results = [r for r in sanitized_results if r["score"] > 0]
        if valid_results:
            avg_score = sum(r["score"] for r in valid_results) / len(valid_results)
            closest_result = min(
                valid_results, key=lambda x: abs(x["score"] - avg_score)
            )
            result = closest_result.copy()
        else:
            result = sanitized_results[0].copy()
        result["multi_evaluation"] = True
        result["avg_score"] = (
            avg_score if "avg_score" in locals() else result.get("score", 0.0)
        )
        result["all_scores"] = [r["score"] for r in sanitized_results]
        result["all_responses"] = [
            {
                "attempt": r.get("_attempt"),
                "score": r.get("score"),
                "reasons": r.get("reasons"),
                "_raw": r.get("_raw", ""),
            }
            for r in sanitized_results
        ]
        if "score" not in result or not isinstance(result.get("score"), (int, float)):
            result["score"] = result.get("avg_score", 0.0)
        return result
    except Exception as exc:
        print(f"[eval][{metric}] fatal error: {exc}")
        return {
            "score": 0.0,
            "reasons": f"评估函数异常: {exc}",
            "all_responses": [
                {"attempt": idx + 1, "score": 0.0, "reasons": "exception", "_raw": raw}
                for idx, raw in enumerate(raw_history)
            ],
        }


def evaluate_qa_pair(client, paper_text: str, qa_pair: Dict, config: dict) -> Dict:
    """评估单个QA对的所有指标"""
    result = qa_pair.copy()
    result["evaluation"] = {}
    
    # 评估所有指标
    metrics = ["relevance", "completeness", "accuracy", "reasonableness", "agnosticism"]
    for metric in metrics:
        result["evaluation"][metric] = evaluate_metric(
            client, 
            metric, 
            paper_text, 
            qa_pair, 
            config
        )
    
    return result


def save_results(results: List[Dict], file_path: str):
    """保存评估结果到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存至: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args)
    
    # 初始化客户端（使用统一工厂）
    client = create_vlm_client(
        VLMClientConfig.from_values(
            api_base=config["base_url"],
            model_name=config["model"],
            api_key=config["api_key"],
            api_type=config.get("api_type"),
            model_version=config.get("model_version"),
            timeout_seconds=float(config.get("request_timeout", 120)),
        )
    )
    
    # 测试API连接
    if not test_api_connection(client, config["model"]):
        exit(1)
    
    try:
        print("正在加载文本和QA对...")
        paper_text = load_text(config["input_text"])
        qa_pairs = load_qa_pairs(config["qa_file"])
        
        print(f"成功加载 {len(qa_pairs)} 个QA对，开始评估...")
        start_time = time.time()
        
        # 评估每个QA对
        evaluated_results = []
        for i, qa_pair in enumerate(qa_pairs, 1):
            print(f"\n正在评估第 {i}/{len(qa_pairs)} 个QA对...")
            print(f"问题: {qa_pair['question'][:50]}...")
            
            result = evaluate_qa_pair(client, paper_text, qa_pair, config)
            evaluated_results.append(result)
            
            # 显示评估结果摘要
            scores = {k: v["score"] for k, v in result["evaluation"].items() if k != "_raw"}
            print(f"评估完成，分数: {scores}")
            
            # 每次评估后保存中间结果，防止中断丢失
            save_results(evaluated_results, config["output_file"])
            
            # 避免API限制
            if i < len(qa_pairs):
                time.sleep(2)
        
        elapsed = time.time() - start_time
        print(f"\n评估完成！总耗时: {elapsed:.1f}秒")
        
        # 计算平均分数
        if evaluated_results:
            metrics = ["relevance", "completeness", "accuracy", "reasonableness", "agnosticism"]
            avg_scores = {}
            for metric in metrics:
                scores = [result["evaluation"][metric]["score"] for result in evaluated_results]
                avg_scores[metric] = sum(scores) / len(scores)
            
            print("\n问答对平均评分:")
            for metric, score in avg_scores.items():
                print(f"- {metric}: {score:.2f}")
        
    except KeyboardInterrupt:
        print("\n用户中断操作，保存已完成的评估结果...")
        if 'evaluated_results' in locals():
            save_results(evaluated_results, config["output_file"])
    except Exception as e:
        print(f"\n发生未预期错误: {str(e)}")
        if 'evaluated_results' in locals():
            save_results(evaluated_results, config["output_file"])


if __name__ == "__main__":
    main()

# 添加evaluate_qa_pairs函数，供API调用
def evaluate_qa_pairs(
    qa_file,
    criteria=None,
    max_concurrency: int = 8,
    llm_config: Optional[Dict[str, Any]] = None,
):
    """
    批量评估问答对
    """
    if criteria is None:
        criteria = list(SUPPORTED_METRICS)

    valid_criteria = []
    for metric in criteria:
        metric_normalized = metric.strip().lower()
        for valid_metric in SUPPORTED_METRICS:
            if valid_metric.lower() == metric_normalized:
                valid_criteria.append(valid_metric)
                break

    if not valid_criteria:
        print("警告: 未找到有效的评估标准，使用默认标准")
        valid_criteria = ["relevance", "completeness", "accuracy", "reasonableness"]

    # 合并外部传入的 llm 配置（优先使用传入的 api_key/base_url/model）
    merged_cfg = dict(DEFAULT_CONFIG)
    if llm_config:
        if llm_config.get("api_key"):
            merged_cfg["api_key"] = llm_config["api_key"]
        if llm_config.get("base_url"):
            merged_cfg["base_url"] = llm_config["base_url"]
        if llm_config.get("model"):
            merged_cfg["model"] = llm_config["model"]
        if llm_config.get("max_retries") is not None:
            merged_cfg["max_retries"] = llm_config["max_retries"]
        if llm_config.get("request_timeout") is not None:
            merged_cfg["request_timeout"] = llm_config["request_timeout"]
        if llm_config.get("timeout_seconds") is not None:
            merged_cfg["request_timeout"] = llm_config["timeout_seconds"]
        for key in ("api_type", "model_version"):
            if llm_config.get(key) is not None:
                merged_cfg[key] = llm_config[key]

    client = create_vlm_client(
        VLMClientConfig.from_values(
            api_base=merged_cfg["base_url"],
            model_name=merged_cfg["model"],
            api_key=merged_cfg["api_key"],
            api_type=merged_cfg.get("api_type"),
            model_version=merged_cfg.get("model_version"),
            timeout_seconds=float(merged_cfg.get("request_timeout", 120)),
        )
    )

    try:
        qa_pairs = load_qa_pairs(qa_file)
    except Exception as e:
        return {"error": f"加载QA对失败: {str(e)}"}

    # 尝试加载全局原文内容，作为没有 source 信息时的兜底
    global_paper_text = ""
    try:
        if os.path.exists(DEFAULT_CONFIG["input_text"]):
            global_paper_text = load_text(DEFAULT_CONFIG["input_text"])
        else:
            global_paper_text = ""
    except Exception:
        global_paper_text = ""

    results: List[Dict[str, Any]] = [None] * len(qa_pairs)

    def get_paper_text(qa_pair: Dict[str, Any]) -> str:
        """
        为单个 QA 对选择用于评估的“文章内容”:
        - 优先使用流水线生成时附带的 qa_generation_unit_text / source_fact_text / source_fact / source
        - 若均不存在，则回退到全局文本（如果配置了 input_text）
        """
        return (
            qa_pair.get("qa_generation_unit_text")
            or qa_pair.get("source_fact_text")
            or qa_pair.get("source_fact")
            or qa_pair.get("source")
            or global_paper_text
            or ""
        )

    def evaluate_single(idx: int, qa_pair: Dict[str, Any]):
        print(f"\n正在评估第 {idx + 1}/{len(qa_pairs)} 个QA对...")
        print(f"问题: {qa_pair['question'][:50]}...")
        result = qa_pair.copy()
        result["evaluation"] = {}
        paper_text = get_paper_text(qa_pair)
        for metric in valid_criteria:
            try:
                print(f"  评估 {metric}...")
                eval_result = evaluate_metric(
                    client,
                    metric,
                    paper_text,
                    qa_pair,
                    merged_cfg
                )
                if "avg_score" in eval_result:
                    eval_result["score"] = eval_result["avg_score"]
                result["evaluation"][metric] = eval_result
                score_val = result["evaluation"][metric].get("score", 0.0)
                print(f"  {metric} 得分: {score_val}")
            except Exception as exc:
                print(f"  {metric} 评估失败: {str(exc)}")
                result["evaluation"][metric] = {
                    "score": 0.0,
                    "reasons": f"评估失败: {str(exc)}"
                }
        return idx, result

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(evaluate_single, idx, qa) for idx, qa in enumerate(qa_pairs)]
        for future in as_completed(futures):
            idx, res = future.result()
            results[idx] = res

    final_results = [res for res in results if res is not None]

    summary = {}
    if final_results:
        for metric in valid_criteria:
            scores = [result["evaluation"].get(metric, {}).get("score", 0) for result in final_results]
            if scores:
                summary[metric] = {
                    "average_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores)
                }

    return {
        "results": final_results,
        "summary": summary,
        "evaluated_count": len(final_results),
        "criteria": valid_criteria
    }
