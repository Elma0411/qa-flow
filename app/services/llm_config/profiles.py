# 文件作用：维护大模型配置档案的 facade。
# 关联说明：对外保持原有函数接口，内部由 store.py 的类化实现承接状态管理。

from .store import activate_profile, delete_profile, list_profiles, upsert_profile

