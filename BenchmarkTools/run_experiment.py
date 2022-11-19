from typing import Dict
from pathlib import Path
from BenchmarkTools import logger
from BenchmarkTools.utils.exceptions import AlreadyFinishedException
from BenchmarkTools.utils.constants import BenchmarkToolsConstants

from BenchmarkTools.benchmarks.toy_benchmark import BOTestFunctionBenchmark
from BenchmarkTools.optimizers.random_search import RandomSearchOptimizer
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment


def run(benchmark_name: str,
        benchmark_settings: Dict,
        optimizer_name: str,
        optimizer_settings: Dict,
        run_id,
        output_path,
        debug=False,
        ):

    # -------------------- CREATE THE OUTPUT DIRECTORY -----------------------------------------------------------------
    output_path = Path(output_path) / benchmark_name / optimizer_name / str(run_id)
    if not debug:
        if (output_path / BenchmarkToolsConstants.FINISHED_FLAG).exists():
            logger.warning('The Directory already exists and has already finished')
            raise AlreadyFinishedException(f'The run directory exists and is not empty. {output_path}')
    Path(output_path).mkdir(exist_ok=True, parents=True)
    # -------------------- CREATE THE OUTPUT DIRECTORY -----------------------------------------------------------------
    benchmark_settings = {
        'objectives':
            [
                {
                    'name': 'branin',
                    'threshold': 350,
                    'limits': [0, 350],
                    'lower_is_better': True,
                },
                {
                    'name': 'currin',
                    'threshold': 12,
                    'limits': [0, 12],
                    'lower_is_better': True,
                }
            ],

        'track_metrics': [],

        'benchmark_parameters': {
            'function_name': 'BraninCurrin',
            'function_kwargs': {
                'negate': False
            }
        },

        'optimization_parameters': {
            'tae_limit': 100,
            'wallclock_limit_in_s': 1800,
            'estimated_cost_limit_in_s': 1800,
            'is_surrogate': True,
        },
    }

    # -------------------- PREPARE STEPS -------------------------------------------------------------------------------
    benchmark = BOTestFunctionBenchmark(**benchmark_settings['benchmark_parameters'])
    configuration_space = benchmark.get_configuration_space(seed=run_id)

    optimizer = RandomSearchOptimizer(
        optimizer_settings={}, benchmark_settings={}, configuration_space=configuration_space
    )

    experiment = MultiObjectiveExperiment(
        benchmark_settings=benchmark_settings,
        benchmark=benchmark,
        optimizer=optimizer,
        output_path=output_path,
        seed=run_id,
    )

    # -------------------- PREPARE STEPS -------------------------------------------------------------------------------
    experiment.setup()

    budget_limit_reached, error = experiment.run()

    if error is not None:
        raise error

    # -------------------- CLEAN UP -------------------------------------------------------------------------------
    try:
        benchmark.__del__()
    except Exception:
        pass

    logger.info('Finished Experiment')
