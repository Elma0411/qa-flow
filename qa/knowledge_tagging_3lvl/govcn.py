# 文件作用：从政府网站样例中构造知识标签训练样本。
# 关联说明：与 openstd_gb.py 并列，为 synth/build_dataset 提供外部样本来源。

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

from .text_cleaning import clean_for_model, normalize_whitespace

_ZUIXIN_BASE = "https://www.gov.cn/zhengce/zuixin/"


def _request(session: requests.Session, url: str, timeout_s: int = 30) -> str:
    resp = session.get(url, timeout=timeout_s)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def _normalize_https(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def _is_gov_cn_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in {"www.gov.cn", "gov.cn"}


def _extract_article_urls(html: str, base_url: str) -> list[str]:
    if BeautifulSoup is None:
        raise RuntimeError("gov.cn crawler needs BeautifulSoup. Install with: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = _normalize_https(urljoin(base_url, href))
        if not _is_gov_cn_url(abs_url):
            continue
        path = urlparse(abs_url).path
        if not re.search(r"^/zhengce/content/\d{6}/content_\d+\.htm$", path):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append(abs_url)

    return out


def iter_govcn_zuixin_urls(max_pages: int = 20, sleep_s: float = 0.2) -> Iterable[str]:
    """
    Crawl policy list pages under `gov.cn/zhengce/zuixin/` and yield article URLs.
    """

    if max_pages < 1:
        return

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; knowledge-tagger/0.1)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
        }
    )

    seen = set()
    for page in range(max_pages):
        suffix = "home.htm" if page == 0 else f"home_{page}.htm"
        url = urljoin(_ZUIXIN_BASE, suffix)
        html = _request(session, url)
        urls = _extract_article_urls(html, base_url=url)
        if not urls:
            break
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            yield u
        time.sleep(sleep_s)


@dataclass(frozen=True)
class GovCnArticle:
    url: str
    title: str
    paragraphs: list[str]

    @property
    def snippet(self) -> str:
        return "\n".join(self.paragraphs)


def _extract_title(html: str) -> str:
    if BeautifulSoup is None:
        raise RuntimeError("gov.cn crawler needs BeautifulSoup. Install with: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1:
        t = normalize_whitespace(h1.get_text(" ", strip=True))
        if t:
            return t

    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta and meta.get("content"):
        t = normalize_whitespace(meta["content"])
        if t:
            return t

    title_tag = soup.find("title")
    if title_tag:
        t = normalize_whitespace(title_tag.get_text(" ", strip=True))
        if t:
            t = t.split("_", 1)[0]
            return t

    return ""


def _select_content_node(soup) -> Optional[object]:
    selectors = [
        "div.pages_content",
        "div#UCAP-CONTENT",
        "div.TRS_Editor",
        "div#Zoom",
        "div.article",
        "div.content",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            return node
    return None


def _extract_paragraphs(html: str) -> list[str]:
    if BeautifulSoup is None:
        raise RuntimeError("gov.cn crawler needs BeautifulSoup. Install with: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    node = _select_content_node(soup) or soup

    paras: list[str] = []
    for p in node.find_all("p"):
        txt = normalize_whitespace(p.get_text(" ", strip=True))
        if not txt:
            continue
        if txt.startswith("来源：") or txt.startswith("责任编辑："):
            continue
        paras.append(txt)
    return paras


def fetch_govcn_article(
    session: requests.Session,
    url: str,
    max_paragraphs: int = 3,
    max_chars: int = 1200,
) -> GovCnArticle:
    html = _request(session, url)
    title = _extract_title(html)
    paras = _extract_paragraphs(html)

    clipped: list[str] = []
    total = 0
    for p in paras:
        if max_paragraphs and len(clipped) >= max_paragraphs:
            break
        if max_chars and total >= max_chars:
            break
        p = clean_for_model(p, max_chars=max_chars)
        if not p:
            continue
        clipped.append(p)
        total += len(p)

    return GovCnArticle(url=url, title=title, paragraphs=clipped)


def _guess_label(title: str, snippet: str) -> tuple[str, float, str]:
    text = f"{title}\n{snippet}"

    if re.search(r"^中华人民共和国.+法", title) or ("主席令" in text):
        if "公告" in title and ("第一条" not in snippet):
            return ("法律法规/国家法律/法律公告", 0.93, "govcn:national_law_announcement")
        if "修订" in title and ("第一条" not in snippet):
            return ("法律法规/国家法律/修订说明", 0.92, "govcn:national_law_revision_note")
        return ("法律法规/国家法律/其他", 0.95, "govcn:national_law")

    if ("国务院令" in text) or ("国令" in text) or title.endswith("条例"):
        if "公告" in title and ("第一条" not in snippet):
            return ("法律法规/行政法规/法规公告", 0.9, "govcn:admin_reg_announcement")
        return ("法律法规/行政法规/其他", 0.93, "govcn:admin_reg")

    if ("最高人民法院" in text) or ("最高人民检察院" in text):
        if "批复" in title:
            return ("法律法规/司法解释/个案批复", 0.92, "govcn:judicial_reply")
        if any(k in title for k in ("程序", "诉讼", "执行")):
            return ("法律法规/司法解释/程序规范", 0.88, "govcn:judicial_procedure")
        return ("法律法规/司法解释/其他", 0.9, "govcn:judicial_interp")

    if ("中共中央" in text) or ("党中央" in text):
        return ("法律法规/党内法规/其他", 0.9, "govcn:party_reg")

    if "政府工作报告" in title or title.endswith("工作报告"):
        return ("政府发文/国家工作报告/其他", 0.9, "govcn:work_report")

    if re.search(r"(十四五|十三五|十五五|五年规划|第十[一二三四五六七八九]个五年规划)", title):
        return ("政府发文/国家政策文件/五年规划", 0.9, "govcn:five_year_plan")

    if "规划" in title:
        return ("政府发文/国家政策文件/发展规划", 0.86, "govcn:plan")

    if "公告" in title and ("第一条" not in snippet):
        return ("政府发文/国家政策公告/其他", 0.86, "govcn:policy_announcement")

    return ("政府发文/国家政策文件/其他", 0.92, "govcn:zuixin_default")


def iter_govcn_seed_examples(
    max_pages: int = 20,
    max_items: int = 300,
    sleep_s: float = 0.2,
    max_paragraphs: int = 3,
    max_chars: int = 1200,
) -> Iterable[dict]:
    """
    Generate labeled examples from gov.cn latest policy list pages.
    """

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; knowledge-tagger/0.1)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
        }
    )

    n = 0
    for url in iter_govcn_zuixin_urls(max_pages=max_pages, sleep_s=sleep_s):
        if n >= max_items:
            break
        art = fetch_govcn_article(
            session=session,
            url=url,
            max_paragraphs=max_paragraphs,
            max_chars=max_chars,
        )
        if not art.title:
            continue
        text = normalize_whitespace(f"{art.title}\n{art.snippet}".strip())
        label_path, conf, reason = _guess_label(art.title, art.snippet)
        yield {
            "text": text,
            "title": art.title,
            "source": {"name": "gov.cn", "url": art.url},
            "label_path": label_path,
            "label_confidence": conf,
            "label_reason": reason,
        }
        n += 1
        time.sleep(sleep_s)

