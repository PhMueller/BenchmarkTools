from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Tuple

from hydra import initialize_config_dir, compose
from omegaconf import DictConfig

from BenchmarkTools import logger


def load_object(import_name: str, import_from: str, **kwargs: Dict) -> Any:
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


def load_benchmark_settings(benchmark_name: str) -> DictConfig:
    config_path = Path(__file__).parent.resolve().absolute().parent / 'hydra_settings'
    with initialize_config_dir(config_dir=str(config_path / 'benchmarks'), job_name='benchmark', version_base=None):
        benchmark_settings = compose(config_name=benchmark_name, overrides=[])
    return benchmark_settings


def load_optimizer_settings(optimizer_name: str) -> DictConfig:
    config_path = Path(__file__).parent.resolve().absolute().parent / 'hydra_settings'
    with initialize_config_dir(config_dir=str(config_path / 'optimizers'), job_name='optimizers', version_base=None):
        optimizer_settings = compose(config_name=optimizer_name, overrides=[])
    return optimizer_settings


def load_benchmark_and_optimizer_conf(benchmark_name: str, optimizer_name: str) -> Tuple[DictConfig, DictConfig]:
    """
    Use the hydra interface to load the benchmark and optimizer settings that are defined in the settings directory.
    Args:
        benchmark_name: str
        optimizer_name: str
    Returns:
        DictConfig, DictConfig
    """
    benchmark_settings = load_benchmark_settings(benchmark_name)
    optimizer_settings = load_optimizer_settings(optimizer_name)

    return benchmark_settings, optimizer_settings
