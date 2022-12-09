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


def load_benchmark(benchmark_name, import_from, use_local: bool) -> Any:
    """
    Load the benchmark object.
    If not `use_local`:  Then load a container from a given source, defined in the HPOBench.
    Import via command from hpobench.[container.]benchmarks.<import_from> import <benchmark_name>
    Parameters
    ----------
    benchmark_name : str
    import_from : str
    use_local : bool
        By default this value is set to false.
        In this case, a container will be downloaded. This container includes all necessary files for the experiment.
        You don't have to install something.
        If true, use the experiment locally. Therefore the experiment has to be installed.
        See the experiment description in the HPOBench.
    Returns
    -------
    Benchmark
    """
    import_str = 'hpobench.' + ('container.' if not use_local else '') + 'benchmarks.' + import_from
    logger.debug(f'Try to execute command: from {import_str} import {benchmark_name}')

    module = import_module(import_str)
    benchmark_obj = getattr(module, benchmark_name)
    logger.debug(f'Benchmark {benchmark_name} successfully loaded')

    return benchmark_obj