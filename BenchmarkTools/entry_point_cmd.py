import argparse
from pathlib import Path
from typing import Tuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from BenchmarkTools.run_experiment import run


def load_omega_conf(benchmark_name: str, optimizer_name: str) -> Tuple[DictConfig, DictConfig]:
    """
    Use the hydra interface to load the benchmark and optimizer settings that are defined in the settings directory.
    Args:
        benchmark_name: str
        optimizer_name: str
    Returns:
        DictConfig, DictConfig
    """
    config_path = Path(__file__).parent.resolve().absolute() / 'hydra_settings'
    with initialize_config_dir(config_dir=str(config_path / 'benchmarks'), job_name='benchmark', version_base=None):
        benchmark_settings = compose(config_name=benchmark_name, overrides=[])

    with initialize_config_dir(config_dir=str(config_path / 'optimizers'), job_name='optimizers', version_base=None):
        optimizer_settings = compose(config_name=optimizer_name, overrides=[])

    return benchmark_settings, optimizer_settings


def main_cmd(args: argparse.Namespace):

    benchmark_settings, optimizer_settings = load_omega_conf(
        benchmark_name=args.benchmark_name, optimizer_name=args.optimizer_name
    )

    run(
        optimizer_name=args.optimizer_name,
        optimizer_settings=optimizer_settings,
        benchmark_name=args.benchmark_name,
        benchmark_settings=benchmark_settings,
        run_id=args.run_id,
        output_path=Path(args.output_dir),
        debug=args.debug,
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('')
    parser.add_argument('--optimizer_name', type=str)
    parser.add_argument('--benchmark_name', type=str)
    parser.add_argument('--run_id', type=int)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    main_cmd(args)
