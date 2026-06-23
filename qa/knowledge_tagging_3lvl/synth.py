# 文件作用：合成知识三级标签训练样本。
# 关联说明：依赖 taxonomy/text_cleaning，为 build_dataset 脚本合成样本。

from __future__ import annotations

import random
from typing import Iterable

from .taxonomy import LeafLabel
from .text_cleaning import normalize_whitespace

_CN_YEAR = [2020, 2021, 2022, 2023, 2024, 2025]
_CN_NAMES = ["张伟", "王芳", "李娜", "刘洋", "陈强", "赵敏", "周杰", "孙磊"]
_CN_ORGS = ["国家电网公司", "省公司", "市公司", "国务院办公厅", "国家发展改革委", "能源局"]
_CN_CITIES = ["北京", "上海", "南京", "杭州", "武汉", "成都", "西安", "广州"]

_PROVINCE_TO_DB_CODE: dict[str, str] = {
    "北京市": "11",
    "天津市": "12",
    "河北省": "13",
    "山西省": "14",
    "内蒙古自治区": "15",
    "辽宁省": "21",
    "吉林省": "22",
    "黑龙江省": "23",
    "上海市": "31",
    "江苏省": "32",
    "浙江省": "33",
    "安徽省": "34",
    "福建省": "35",
    "江西省": "36",
    "山东省": "37",
    "河南省": "41",
    "湖北省": "42",
    "湖南省": "43",
    "广东省": "44",
    "广西壮族自治区": "45",
    "海南省": "46",
    "重庆市": "50",
    "四川省": "51",
    "贵州省": "52",
    "云南省": "53",
    "西藏自治区": "54",
    "陕西省": "61",
    "甘肃省": "62",
    "青海省": "63",
    "宁夏回族自治区": "64",
    "新疆维吾尔自治区": "65",
}

_STANDARD_PREFIX_BY_INDUSTRY_LEAF3: dict[str, str] = {
    "电力标准": "DL/T",
    "安全生产标准": "AQ/T",
    "环境保护标准": "HJ",
    "档案标准": "DA/T",
    "建筑工程标准": "JG",
    "城镇建设工程标准": "CJJ",
    "能源标准": "NB/T",
}


def _rand_std_no(rng: random.Random, prefix: str) -> str:
    num = rng.randint(1000, 99999)
    year = rng.choice(_CN_YEAR)
    return f"{prefix} {num}-{year}"


def _rand_isbn(rng: random.Random) -> str:
    parts = [str(rng.randint(100, 999)), str(rng.randint(1, 99)), str(rng.randint(1000, 9999)), str(rng.randint(1, 9))]
    return "ISBN: 978-" + "-".join(parts)


def _rand_doi(rng: random.Random) -> str:
    return f"DOI: 10.{rng.randint(1000,9999)}/{rng.choice(['abc','sgcc','nlp','grid'])}.{rng.randint(1000,99999)}"


def _pick_keywords(desc: str, limit: int = 6) -> list[str]:
    if not desc:
        return []
    tokens: list[str] = []
    for part in desc.replace("，", " ").replace("。", " ").replace("；", " ").replace("、", " ").split():
        part = part.strip().strip('"“”')
        if len(part) >= 2 and not part.startswith("其他"):
            tokens.append(part)
    uniq: list[str] = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
    return uniq[:limit]


def _format_keywords(keywords: list[str]) -> str:
    if not keywords:
        return ""
    return "；".join(keywords[:6])


def synthesize_examples(
    leaves: Iterable[LeafLabel],
    per_label: int,
    seed: int = 42,
) -> list[dict]:
    """
    Generate synthetic labeled samples for every leaf label so the supervised
    model has coverage (even when no real dataset exists yet).
    """

    rng = random.Random(seed)
    out: list[dict] = []

    for leaf in leaves:
        keywords = _pick_keywords(leaf.description)
        kw_line = _format_keywords(keywords)

        for _i in range(per_label):
            year = rng.choice(_CN_YEAR)
            serial = rng.randint(1, 999)
            city = rng.choice(_CN_CITIES)
            person = rng.choice(_CN_NAMES)
            org = rng.choice(_CN_ORGS)

            doc_style = leaf.level1
            title = ""
            body = ""

            if doc_style == "标准":
                # Prefer consistent standard numbering by the taxonomy branch.
                if leaf.level3 == "标准发布公告":
                    title = f"关于发布{leaf.level2}标准的公告（{year}）"
                    body = "文档名可能包含“标准发布”，内容只有公告描述，没有标准正文。"
                else:
                    std_no = ""
                    if leaf.level2 == "国家标准":
                        if leaf.path.endswith("强制性标准"):
                            std_no = _rand_std_no(rng, "GB")
                        elif leaf.path.endswith("推荐性标准"):
                            std_no = _rand_std_no(rng, "GB/T")
                        elif leaf.path.endswith("指导性技术文件"):
                            std_no = _rand_std_no(rng, "GB/Z")
                        else:
                            std_no = _rand_std_no(rng, "GB/T")
                    elif leaf.level2 == "地方标准":
                        code = _PROVINCE_TO_DB_CODE.get(leaf.level3)
                        if code:
                            std_no = _rand_std_no(rng, f"DB{code}/T")
                        else:
                            std_no = _rand_std_no(rng, "DB/T")
                    elif leaf.level2 == "行业标准":
                        prefix = _STANDARD_PREFIX_BY_INDUSTRY_LEAF3.get(leaf.level3, "DL/T")
                        std_no = _rand_std_no(rng, prefix)
                    elif leaf.level2 == "团体标准":
                        if leaf.level3 == "中国电力企业联合会":
                            std_no = _rand_std_no(rng, "T/CEC")
                        else:
                            std_no = _rand_std_no(rng, "T/XXX")
                    elif leaf.level2 == "企业标准":
                        if leaf.level3 == "国家电网公司":
                            std_no = _rand_std_no(rng, "Q/GDW")
                        else:
                            std_no = _rand_std_no(rng, "Q/XXX")
                    else:
                        std_no = _rand_std_no(rng, "GB/T")

                    title = f"{std_no} {leaf.level2}{leaf.level3}相关要求（{year}版）"
                    if leaf.level2 == "地方标准" and leaf.level3 in _PROVINCE_TO_DB_CODE:
                        body = (
                            f"本标准为{leaf.level3}地方标准，规定了{leaf.level3}相关的术语、分类、技术要求与试验方法。"
                            f"适用于{leaf.level3}范围内相关单位参考执行。{kw_line}"
                        )
                    else:
                        body = (
                            f"本标准规定了{leaf.level2}{leaf.level3}相关的术语、分类、技术要求与试验方法。"
                            f"适用于相关单位参考执行。{kw_line}"
                        )
            elif doc_style == "学术论文":
                title = f"{leaf.level3}相关研究进展与方法分析（{year}）"
                body = (
                    f"摘要：本文围绕{leaf.level2}-{leaf.level3}展开研究，提出若干方法并进行实验验证。"
                    f"关键词：{kw_line or (leaf.level3)}。{_rand_doi(rng)}"
                )
            elif doc_style == "图书":
                title = f"{leaf.level2}：{leaf.level3}入门与实践（{year}第{rng.randint(1,5)}版）"
                body = (
                    f"出版社：电力出版社。{_rand_isbn(rng)}。"
                    f"作者：{person}。内容简介：本书系统介绍{leaf.level2}-{leaf.level3}的基础概念与应用案例。{kw_line}"
                )
            elif doc_style == "法律法规":
                title = f"关于{leaf.level3}的{rng.choice(['规定','解释','办法','决定'])}（{year}年修订）"
                body = (
                    f"第一条 为规范{leaf.level2}{leaf.level3}相关活动，制定本文件。"
                    f"第二条 本文件自发布之日起施行（第{serial}号）。{kw_line}"
                )
            elif doc_style == "公司制度":
                title = f"{org}{leaf.level2}{leaf.level3}{rng.choice(['管理办法','实施细则','工作手册'])}（{year}）"
                body = (
                    f"编号：国网〔{year}〕{serial}号。为加强{leaf.level2}{leaf.level3}工作，明确职责与流程，特制定本制度。{kw_line}"
                )
            elif doc_style == "公司发文":
                title = f"{org}关于{leaf.level3}的{rng.choice(['通知','方案','意见'])}（{year}）"
                body = f"编号：国网〔{year}〕{serial}号。为推进{leaf.level2}{leaf.level3}相关工作，现提出如下要求与安排。{kw_line}"
            elif doc_style == "政府发文":
                title = f"关于推进{leaf.level2}{leaf.level3}工作的{rng.choice(['意见','通知','公告','规划'])}（{year}）"
                body = f"文号：国办发〔{year}〕{serial}号。为落实国家部署，现就{leaf.level2}{leaf.level3}提出政策措施。{kw_line}"
            elif doc_style == "业务通识":
                title = f"{leaf.level2}{leaf.level3}培训要点（{year}）"
                body = f"讲师：{person}。本资料用于{leaf.level3}培训，包含背景介绍、流程步骤与注意事项。{kw_line}"
            else:
                title = f"{leaf.level2}{leaf.level3}相关文档（{year}）"
                body = f"文件编号：{year}-{serial:03d}。本文档涉及{leaf.level2}{leaf.level3}内容。{kw_line}"

            text = normalize_whitespace(f"{title}\n{body}")
            out.append(
                {
                    "text": text,
                    "title": title,
                    "source": {"name": "synthetic", "url": ""},
                    "label_path": leaf.path,
                    "label_confidence": 0.35,
                    "label_reason": "synthetic",
                }
            )

    return out
