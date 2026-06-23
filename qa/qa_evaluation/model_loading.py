# 文件作用：封装 QA 评估器依赖的本地模型与语法工具加载。
# 关联说明：由 qa_quality_evaluator 组合调用，避免 __init__ 同时承担模型装配与评分逻辑。

from __future__ import annotations

from typing import Any, Tuple

from .language_models import MaskedBert
from .runtime import select_device


def load_semantic_model(resolve_model_reference: Any, default_coverage_model: str, use_local_models: bool, sentence_transformer_cls: Any) -> Any:
    device = select_device()
    if use_local_models:
        local_bge_path = resolve_model_reference(None, default_name=default_coverage_model)
        if local_bge_path and sentence_transformer_cls and hasattr(sentence_transformer_cls, "__call__"):
            import os

            if os.path.exists(local_bge_path):
                print(f"✅ 使用本地BGE-M3模型: {local_bge_path}")
                return sentence_transformer_cls(local_bge_path, device=device)
            print(f"⚠️ 本地BGE-M3模型不存在: {local_bge_path}")
            print("💡 请运行: python model_downloader.py download bge-m3")
            raise FileNotFoundError("BGE-M3模型未找到，请先下载模型")

    print("🌐 使用在线BGE-M3模型")
    return sentence_transformer_cls("BAAI/bge-m3", device=device)


def load_fluency_model(resolve_model_reference: Any, default_fluency_model: str) -> MaskedBert:
    device = select_device()
    fluency_model_path = resolve_model_reference(None, default_name=default_fluency_model)
    import os

    if os.path.exists(fluency_model_path):
        print(f"✅ 使用本地流畅度模型: {fluency_model_path}")
        return MaskedBert.from_pretrained(
            fluency_model_path,
            device=device,
            sentence_length=100,
        )
    print(f"❌ 流畅度模型不存在: {fluency_model_path}")
    print("💡 请运行: python model_downloader.py download chinese-bert-wwm")
    raise FileNotFoundError("流畅度模型未找到，请先下载模型")


def load_grammar_tool() -> Tuple[Any | None, bool]:
    try:
        import language_tool_python

        tool = language_tool_python.LanguageToolPublicAPI("zh")
        print("✅ 语法检查工具已加载")
        return tool, True
    except Exception as exc:
        print(f"⚠️ 语法检查工具加载失败: {str(exc)}")
        print("💡 将使用简化的流畅度评估（仅基于困惑度）")
        return None, False

