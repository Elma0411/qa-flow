# 文件作用：聚合管理端问答查询和写入能力。
# 关联说明：作为同目录 qa_query 与 qa_write 的聚合层，被 __init__.py facade 导出。

from .qa_common import AdminMilvusError
from .qa_query import get_qa_item, list_qa_items
from .qa_write import (
    batch_update_fields,
    export_items_to_json,
    fetch_records_by_ids,
    hard_delete,
    replace_records,
    update_qa_item_fields,
)

__all__ = [
    "AdminMilvusError",
    "batch_update_fields",
    "export_items_to_json",
    "fetch_records_by_ids",
    "get_qa_item",
    "hard_delete",
    "list_qa_items",
    "replace_records",
    "update_qa_item_fields",
]
