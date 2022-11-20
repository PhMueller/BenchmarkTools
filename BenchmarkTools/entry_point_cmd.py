from pathlib import Path

from BenchmarkTools.run_experiment import run


def main_cmd(args):
    run(
        optimizer_name=args.optimizer_name,
        optimizer_settings={},
        benchmark_name=args.benchmark_name,
        benchmark_settings={},
        run_id=0,
        output_path=Path('C:/Users/Philipp/PycharmProjects/BenchmarkTools/Results')
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('')
    parser.add_argument('--optimizer_name')
    parser.add_argument('--benchmark_name')
    args = parser.parse_args()
    main_cmd(args)
