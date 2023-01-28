import argparse
from pathlib import Path

from BenchmarkTools.run_experiment import run
from BenchmarkTools.utils.loader_tools import load_benchmark_and_optimizer_conf


def main_cmd(args: argparse.Namespace):

    benchmark_settings, optimizer_settings = load_benchmark_and_optimizer_conf(
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
