# 文件作用：作为文档三级标签分类器的公共 facade。
# 关联说明：聚合 classifier_core 和 service，供 app.services.knowledge_tagging 调用。

from .classifier_core import DocumentRuleClassifier
from .service import TextDocumentLevel3Classifier, build_default_config_path

__all__ = [
    "DocumentRuleClassifier",
    "TextDocumentLevel3Classifier",
    "build_default_config_path",
]
