# 文件作用：读取上传文件、写出 CSV/JSON 产物并控制批处理并发。
# 关联说明：依赖 paths.py 解析落盘位置，为 pipeline_batch_routes 读取上传文件。

import csv
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

from .paths import get_output_path


def read_uploaded_file_content(upload_file: UploadFile) -> str:
    """Read text content from an UploadFile without touching disk."""
    try:
        content = upload_file.file.read()
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("gbk")
            except UnicodeDecodeError:
                return content.decode("latin-1")
    finally:
        upload_file.file.seek(0)


def read_uploaded_json_file(upload_file: UploadFile) -> list:
    """Parse JSON payload from an UploadFile."""
    content = read_uploaded_file_content(upload_file)
    return json.loads(content)


def save_temp_csv_file(upload_file: UploadFile) -> str:
    """Persist a CSV upload to a temporary file for downstream scripts."""
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".csv", delete=False)
    try:
        content = upload_file.file.read()
        temp_file.write(content)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()
        upload_file.file.seek(0)


def read_multiple_uploaded_files(upload_files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Read many uploaded text files."""
    results: List[Dict[str, Any]] = []
    for file in upload_files:
        try:
            content = read_uploaded_file_content(file)
            results.append(
                {
                    "filename": file.filename,
                    "content": content,
                    "size": len(content),
                    "status": "success",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "filename": file.filename,
                    "content": None,
                    "size": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return results


def read_multiple_uploaded_json_files(upload_files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Read many uploaded JSON files."""
    results: List[Dict[str, Any]] = []
    for file in upload_files:
        try:
            data = read_uploaded_json_file(file)
            results.append(
                {
                    "filename": file.filename,
                    "data": data,
                    "count": len(data) if isinstance(data, list) else 1,
                    "status": "success",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "filename": file.filename,
                    "data": None,
                    "count": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return results


def save_batch_results(
    results: List[Dict[str, Any]],
    save_mode: str,
    prefix: str,
    ext: str,
) -> Dict[str, Any]:
    """
    Persist batch processing outputs according to the requested save mode.

    - save_mode == "unified": merge all successful results into a single file.
    - save_mode == "separate": one output per input file.
    """
    if save_mode == "unified":
        output_file = get_output_path(f"{prefix}_batch", ext)
        if ext == ".json":
            all_data: List[Any] = []
            for result in results:
                if result.get("status") == "success" and result.get("data"):
                    data = result["data"]
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            return {
                "mode": "unified",
                "file": output_file,
                "total_items": len(all_data),
                "source_files": [
                    r["filename"] for r in results if r.get("status") == "success"
                ],
            }
        if ext == ".csv":
            total_rows = 0
            with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
                writer: Optional[csv.DictWriter] = None
                for result in results:
                    if result.get("status") != "success" or not result.get("data"):
                        continue
                    for item in result["data"]:
                        if writer is None:
                            fieldnames = list(item.keys())
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                        writer.writerow(item)
                        total_rows += 1
            return {
                "mode": "unified",
                "file": output_file,
                "total_rows": total_rows,
                "source_files": [
                    r["filename"] for r in results if r.get("status") == "success"
                ],
            }
        # Unsupported extension for unified mode
        return {
            "mode": "unified",
            "file": output_file,
            "total_items": 0,
            "source_files": [],
        }

    # Default: separate mode
    saved_files: List[Dict[str, Any]] = []
    for result in results:
        if result.get("status") != "success" or not result.get("data"):
            continue
        original_name = os.path.splitext(result["filename"])[0]
        output_file = get_output_path(f"{prefix}_{original_name}", ext)
        data = result["data"]
        if ext == ".json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif ext == ".csv":
            with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
                fieldnames: List[str] = list(data[0].keys()) if data else []
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in data:
                    writer.writerow(item)
        saved_files.append(
            {
                "source_file": result["filename"],
                "output_file": output_file,
                "item_count": len(data) if isinstance(data, list) else 1,
            }
        )
    return {"mode": "separate", "files": saved_files, "total_files": len(saved_files)}

__all__ = [
    'read_multiple_uploaded_files',
    'read_multiple_uploaded_json_files',
    'read_uploaded_file_content',
    'read_uploaded_json_file',
    'save_batch_results',
    'save_temp_csv_file',
]
