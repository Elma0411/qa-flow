from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qa.chunking import (
    get_capabilities,
    manual_split,
    preview_split_points,
    preprocess_file,
    save_to_separate_files,
    split_file,
)


def _decode_escapes(value: str) -> str:
    return (
        str(value or "")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
    )


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if args.type:
        config["splitType"] = args.type
    if args.min is not None:
        config["textSplitMinLength"] = int(args.min)
    if args.max is not None:
        config["textSplitMaxLength"] = int(args.max)
    if args.chunk_size is not None:
        config["chunkSize"] = int(args.chunk_size)
    if args.chunk_overlap is not None:
        config["chunkOverlap"] = int(args.chunk_overlap)
    if args.separator is not None:
        config["separator"] = _decode_escapes(args.separator)
    if args.separators is not None:
        config["separators"] = [
            _decode_escapes(item.strip())
            for item in str(args.separators).split(",")
            if item.strip()
        ]
    if args.language is not None:
        config["splitLanguage"] = args.language
    if args.custom_separator is not None:
        config["customSeparator"] = _decode_escapes(args.custom_separator)
    return config


def _write_json(payload: Any, output_path: str | None) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote result to {output_path}")
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="easy_dataset_chunk",
        description="Python reconstruction of easy-dataset chunk capabilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("capabilities")

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--file", required=True)
    preprocess_parser.add_argument("--out")

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--file", required=True)
    split_parser.add_argument("--type")
    split_parser.add_argument("--min", type=int)
    split_parser.add_argument("--max", type=int)
    split_parser.add_argument("--chunk-size", type=int)
    split_parser.add_argument("--chunk-overlap", type=int)
    split_parser.add_argument("--separator")
    split_parser.add_argument("--separators")
    split_parser.add_argument("--language")
    split_parser.add_argument("--custom-separator")
    split_parser.add_argument("--out")
    split_parser.add_argument("--parts-base")

    manual_parser = subparsers.add_parser("manual-split")
    manual_parser.add_argument("--file", required=True)
    manual_parser.add_argument("--points", required=True)
    manual_parser.add_argument("--out")

    args = parser.parse_args()

    if args.command == "capabilities":
        _write_json(get_capabilities(), None)
        return

    if args.command == "preprocess":
        _write_json(preprocess_file(args.file), args.out)
        return

    if args.command == "split":
        result = split_file(file_path=args.file, config=_build_config(args))
        _write_json(result, args.out)
        if args.parts_base:
            parts_result = save_to_separate_files(result.get("chunks") or [], args.parts_base)
            print(
                f"Wrote {int(parts_result.get('count') or 0)} parts to {parts_result.get('outputDir')}"
            )
        return

    if args.command == "manual-split":
        split_points: List[Dict[str, Any]] = []
        for index, raw in enumerate(str(args.points).split(","), start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            split_points.append({"id": index, "position": int(stripped)})

        content = Path(args.file).read_text(encoding="utf-8")
        payload = {
            "preview": preview_split_points(content, split_points),
            "chunks": manual_split(
                content=content,
                file_name=Path(args.file).name,
                split_points=split_points,
            ),
        }
        _write_json(payload, args.out)
        return


if __name__ == "__main__":
    main()
