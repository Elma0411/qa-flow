"""Score the committed BGE relevance fixture and report threshold behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qa.retrieval.reranker import RerankerService


DEFAULT_FIXTURE = ROOT_DIR / "tests" / "testdata" / "bge_reranker_relevance_cases.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a relevant case is below the threshold or a negative is admitted.",
    )
    return parser.parse_args()


def _load_fixture(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("fixture must contain a cases list")
    return payload


def main() -> int:
    args = _parse_args()
    payload = _load_fixture(args.fixture)
    threshold = float(payload.get("minimum_logit") or 0.0)
    maximum_primary_source_drop = float(payload.get("maximum_primary_source_drop") or 0.0)
    cases = [case for case in payload["cases"] if isinstance(case, dict)]
    service = RerankerService(args.model_path)
    relevant_scores: Dict[str, float] = {}
    for case in cases:
        if str(case.get("label") or "") != "relevant":
            continue
        query = str(case.get("query") or "")
        score = service.score(query, [str(case.get("passage") or "")])[0]
        relevant_scores[query] = max(float(score), relevant_scores.get(query, float("-inf")))
    results: List[Dict[str, Any]] = []
    for case in cases:
        score = service.score(str(case.get("query") or ""), [str(case.get("passage") or "")])[0]
        label = str(case.get("label") or "")
        expected_admitted = label == "relevant"
        primary_source_score = relevant_scores.get(str(case.get("query") or ""))
        admitted = score >= threshold and (
            primary_source_score is None
            or primary_source_score - score <= maximum_primary_source_drop
        )
        matches_expectation = admitted == expected_admitted
        results.append(
            {
                "id": case.get("id"),
                "label": label,
                "score": round(float(score), 6),
                "admitted": admitted,
                "expected_admitted": expected_admitted,
                "primary_source_score": (
                    round(primary_source_score, 6)
                    if primary_source_score is not None
                    else None
                ),
                "matches_expectation": matches_expectation,
            }
        )
    summary = {
        "fixture": str(args.fixture),
        "model_path": str(service.model_path),
        "minimum_logit": threshold,
        "maximum_primary_source_drop": maximum_primary_source_drop,
        "cases": len(results),
        "matches": sum(item["matches_expectation"] is True for item in results),
        "mismatches": sum(item["matches_expectation"] is False for item in results),
        "results": results,
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for item in results:
            marker = "PASS" if item["matches_expectation"] else "FAIL"
            print(f"{marker}\t{item['score']:.6f}\t{item['label']}\t{item['id']}")
        print(
            f"threshold={threshold:.3f} cases={summary['cases']} "
            f"matches={summary['matches']} mismatches={summary['mismatches']}"
        )
    return 1 if args.strict and summary["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
