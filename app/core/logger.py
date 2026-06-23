# 文件作用：配置服务统一日志格式和 logger 实例。
# 关联说明：被 routers 和 services 共享使用，与 config/time_utils 一起支撑运行观测。

import logging


def get_logger(name: str = "api_server") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger


logger = get_logger()

__all__ = ["logger", "get_logger"]
