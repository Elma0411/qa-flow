# 文件作用：聚合存储路径、上传解析和结果合并服务。
# 关联说明：聚合同目录存储能力，是 __init__.py facade 的主要来源。

from .consolidation import build_consolidated_entry
from .csv_export import write_consolidated_csv
from .merge import merge_consolidated_entries
from .paths import (
    cleanup_outputs,
    get_output_path,
    resolve_batch_concurrency,
    sanitize_filename,
    write_status_file,
)
from .uploads import (
    read_multiple_uploaded_files,
    read_multiple_uploaded_json_files,
    read_uploaded_file_content,
    read_uploaded_json_file,
    save_batch_results,
    save_temp_csv_file,
)

__all__ = [
    'build_consolidated_entry',
    'cleanup_outputs',
    'get_output_path',
    'merge_consolidated_entries',
    'read_multiple_uploaded_files',
    'read_multiple_uploaded_json_files',
    'read_uploaded_file_content',
    'read_uploaded_json_file',
    'resolve_batch_concurrency',
    'sanitize_filename',
    'save_batch_results',
    'save_temp_csv_file',
    'write_consolidated_csv',
    'write_status_file',
]
