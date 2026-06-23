# 文件作用：实现知识标签规则分类和规则兜底。
# 关联说明：被 predictor 作为模型预测之外的规则补充。

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuleResult:
    label_path: str
    confidence: float
    reason: str


_RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_RE_ISBN = re.compile(
    r"\bISBN(?:-13)?\s*[:：]?\s*(97[89][- ]?)?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?[\dX]\b",
    re.IGNORECASE,
)

_RE_STD_GB_Z = re.compile(r"\bGB/Z\s*\d", re.IGNORECASE)
_RE_STD_GB_T = re.compile(r"\bGB/T\s*\d", re.IGNORECASE)
_RE_STD_GB = re.compile(r"\bGB\s*\d", re.IGNORECASE)

_RE_STD_T_CEC = re.compile(r"\bT/CEC\b", re.IGNORECASE)
_RE_STD_Q_GDW = re.compile(r"\bQ/GDW\b", re.IGNORECASE)
_RE_STD_Q = re.compile(r"\bQ/[A-Z]{2,}\b|\bQ\b", re.IGNORECASE)

_RE_STD_DB = re.compile(r"\bDB\s*(\d{2})\b", re.IGNORECASE)

_RE_STD_PREFIX = [
    (re.compile(r"\bDL\b", re.IGNORECASE), "标准/行业标准/电力标准", "std:DL"),
    (re.compile(r"\bAQ\b", re.IGNORECASE), "标准/行业标准/安全生产标准", "std:AQ"),
    (re.compile(r"\bHJ\b", re.IGNORECASE), "标准/行业标准/环境保护标准", "std:HJ"),
    (re.compile(r"\bDA\b", re.IGNORECASE), "标准/行业标准/档案标准", "std:DA"),
    (re.compile(r"\bJG\b", re.IGNORECASE), "标准/行业标准/建筑工程标准", "std:JG"),
    (re.compile(r"\bCJJ\b", re.IGNORECASE), "标准/行业标准/城镇建设工程标准", "std:CJJ"),
    (re.compile(r"\bNB\b", re.IGNORECASE), "标准/行业标准/能源标准", "std:NB"),
]

_DB_PROVINCE_CODE_TO_LABEL: dict[str, str] = {
    "11": "标准/地方标准/北京市",
    "12": "标准/地方标准/天津市",
    "13": "标准/地方标准/河北省",
    "14": "标准/地方标准/山西省",
    "15": "标准/地方标准/内蒙古自治区",
    "21": "标准/地方标准/辽宁省",
    "22": "标准/地方标准/吉林省",
    "23": "标准/地方标准/黑龙江省",
    "31": "标准/地方标准/上海市",
    "32": "标准/地方标准/江苏省",
    "33": "标准/地方标准/浙江省",
    "34": "标准/地方标准/安徽省",
    "35": "标准/地方标准/福建省",
    "36": "标准/地方标准/江西省",
    "37": "标准/地方标准/山东省",
    "41": "标准/地方标准/河南省",
    "42": "标准/地方标准/湖北省",
    "43": "标准/地方标准/湖南省",
    "44": "标准/地方标准/广东省",
    "45": "标准/地方标准/广西壮族自治区",
    "46": "标准/地方标准/海南省",
    "50": "标准/地方标准/重庆市",
    "51": "标准/地方标准/四川省",
    "52": "标准/地方标准/贵州省",
    "53": "标准/地方标准/云南省",
    "54": "标准/地方标准/西藏自治区",
    "61": "标准/地方标准/陕西省",
    "62": "标准/地方标准/甘肃省",
    "63": "标准/地方标准/青海省",
    "64": "标准/地方标准/宁夏回族自治区",
    "65": "标准/地方标准/新疆维吾尔自治区",
}


def classify_with_rules(text: str) -> Optional[RuleResult]:
    """
    High-precision rules only. Returns None if no confident rule match.
    """

    if not text or not text.strip():
        return None

    hay = " ".join(text.strip().split())

    if _RE_STD_GB_Z.search(hay):
        return RuleResult("标准/国家标准/指导性技术文件", 0.98, "std:GB_Z")
    if _RE_STD_GB_T.search(hay):
        return RuleResult("标准/国家标准/推荐性标准", 0.98, "std:GB_T")
    if _RE_STD_GB.search(hay) and not re.search(r"\bGB/(?:T|Z)\b", hay, re.IGNORECASE):
        return RuleResult("标准/国家标准/强制性标准", 0.98, "std:GB")

    if _RE_STD_T_CEC.search(hay):
        return RuleResult("标准/团体标准/中国电力企业联合会", 0.97, "std:T_CEC")
    if _RE_STD_Q_GDW.search(hay):
        return RuleResult("标准/企业标准/国家电网公司", 0.97, "std:Q_GDW")
    if _RE_STD_DB.search(hay):
        code = _RE_STD_DB.search(hay).group(1)
        label = _DB_PROVINCE_CODE_TO_LABEL.get(code, "标准/地方标准/其他地方标准")
        return RuleResult(label, 0.96, f"std:DB{code}")
    for reg, label, reason in _RE_STD_PREFIX:
        if reg.search(hay):
            return RuleResult(label, 0.96, reason)
    if _RE_STD_Q.search(hay):
        return RuleResult("标准/企业标准/其他", 0.9, "std:Q_other")

    return None
