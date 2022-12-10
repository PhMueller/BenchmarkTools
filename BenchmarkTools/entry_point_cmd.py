from BenchmarkTools.run_experiment import run


def load_single_hydra_config(optimizer, benchmark):
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    initialize(config_path="hydra_settings", version_base=None)
    optimizer_settings = compose(config_name=f'optimizers/{optimizer}')['optimizers']
    benchmark_settings = compose(config_name=f'benchmarks/{benchmark}')['benchmarks']
    print(OmegaConf.to_yaml(optimizer_settings))
    print(OmegaConf.to_yaml(benchmark_settings))
    return optimizer_settings, benchmark_settings


def main_cmd(args):
    optimizer_settings, benchmark_settings = load_single_hydra_config(
        optimizer=args.optimizer_name, benchmark=args.benchmark_name
    )

    run(
        optimizer_name=args.optimizer_name,
        optimizer_settings=optimizer_settings,
        benchmark_name=args.benchmark_name,
        benchmark_settings=benchmark_settings,
        run_id=args.run_id,
        output_path=args.output_path,
        debug=args.debug,
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('')
    parser.add_argument('--optimizer_name', type=str)
    parser.add_argument('--benchmark_name', type=str)
    parser.add_argument('--run_id', type=int)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    main_cmd(args)
