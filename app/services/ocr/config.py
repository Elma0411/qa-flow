# 文件作用：维护 OCR 配置档案服务的 facade。
# 关联说明：对外保持原有函数接口，内部由 store.py 的类化实现承接状态管理。

from .store import (
    activate_profile,
    delete_profile,
    get_active_profile,
    get_active_vlm_defaults,
    get_profile,
    list_profiles,
    upsert_profile,
)
