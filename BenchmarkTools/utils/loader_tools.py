from importlib import import_module
from typing import Any

from BenchmarkTools import logger


def load_object(import_name: str, import_from: str) -> Any:
    """
    This helper function loads dynamically a class or function.

    It executes the following statement:
    > from `import_from` import `import_name`

    Args:
        import_name: str
        import_from: str

    Returns:
        obj
    """
    logger.debug(f'Try to execute command: from {import_from} import {import_name}')
    module = import_module(import_from)
    obj = getattr(module, import_name)
    return obj
