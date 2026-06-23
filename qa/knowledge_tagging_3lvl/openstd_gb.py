# 文件作用：抓取国家标准公开信息并生成标准类样本。
# 关联说明：与 govcn.py 并列，为标准类标签生成种子样本。

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import requests
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


@dataclass(frozen=True)
class OpenStdItem:
    std_no: str
    std_name_zh: str
    detail_url: str

    @property
    def text(self) -> str:
        if self.std_name_zh:
            return f"{self.std_no} {self.std_name_zh}"
        return self.std_no


def _request(session: requests.Session, url: str, timeout_s: int = 30) -> str:
    resp = session.get(url, timeout=timeout_s)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def crawl_openstd_gb(
    std_type: int,
    max_items: int,
    page_size: int = 50,
    sleep_s: float = 0.2,
) -> list[OpenStdItem]:
    """
    Crawl `openstd.samr.gov.cn` national standards list.

    std_type:
      1 -> GB (mandatory)
      2 -> GB/T (recommended)
      3 -> GB/Z (guidance)
    """

    if std_type not in (1, 2, 3):
        raise ValueError("std_type must be 1, 2, or 3")

    if BeautifulSoup is None:
        raise RuntimeError(
            "openstd crawler needs BeautifulSoup. Install with: pip install beautifulsoup4"
        )

    base = "http://openstd.samr.gov.cn"
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; knowledge-tagger/0.1)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
        }
    )

    items: list[OpenStdItem] = []
    page = 1
    while len(items) < max_items:
        url = f"{base}/bzgk/gb/std_list?p.p1={std_type}&page={page}&pageSize={page_size}"
        html = _request(session, url)
        soup = BeautifulSoup(html, "lxml")

        page_rows = 0
        for row in soup.find_all("tr"):
            a_no = row.find("a", onclick=lambda v: bool(v) and "showInfo(" in v)
            if not a_no:
                continue
            std_no = a_no.get_text(strip=True)
            onclick = a_no.get("onclick") or ""
            hcno = ""
            if "'" in onclick:
                parts = onclick.split("'")
                if len(parts) >= 2:
                    hcno = parts[1]
            if not hcno:
                continue

            tds = row.find_all("td")
            std_name = ""
            if len(tds) >= 4:
                std_name = tds[3].get_text(" ", strip=True)

            detail_url = f"{base}/bzgk/gb/newGbInfo?hcno={hcno}"
            items.append(OpenStdItem(std_no=std_no, std_name_zh=std_name, detail_url=detail_url))
            page_rows += 1
            if len(items) >= max_items:
                break

        if page_rows == 0:
            break

        page += 1
        time.sleep(sleep_s)

    return items


def iter_openstd_seed_examples(max_per_type: int = 200) -> Iterable[dict]:
    """
    Generate labeled examples from openstd (3 national standard types).
    """

    type_to_label = {
        1: "标准/国家标准/强制性标准",
        2: "标准/国家标准/推荐性标准",
        3: "标准/国家标准/指导性技术文件",
    }
    for std_type in (1, 2, 3):
        for item in crawl_openstd_gb(std_type=std_type, max_items=max_per_type):
            yield {
                "text": item.text,
                "title": item.text,
                "source": {"name": "openstd.samr.gov.cn", "url": item.detail_url},
                "label_path": type_to_label[std_type],
                "label_confidence": 0.99,
                "label_reason": f"openstd:p.p1={std_type}",
            }
