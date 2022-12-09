from pathlib import Path
from typing import Dict

from BenchmarkTools import logger
from BenchmarkTools.benchmarks.toy_benchmark import BOTestFunctionBenchmark
from BenchmarkTools.benchmarks.hpobench_container_interface import HPOBenchContainerInterface
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment
from BenchmarkTools.optimizers.random_search import RandomSearchOptimizer
from BenchmarkTools.utils.constants import BenchmarkToolsConstants
from BenchmarkTools.utils.exceptions import AlreadyFinishedException
from BenchmarkTools.utils.loader_tools import load_object


def run(benchmark_name: str,
        benchmark_settings: Dict,
        optimizer_name: str,
        optimizer_settings: Dict,
        run_id,
        output_path,
        debug=False,
        ):

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

        # 'benchmark_type': 'HPOBenchContainer',
        'benchmark_type': 'HPOBenchLocal',

        'optimization_parameters': {
            'tae_limit': 100,
            'wallclock_limit_in_s': 1800,
            'estimated_cost_limit_in_s': 1800,
            'is_surrogate': True,
        },
    }
    benchmark_settings = {
        'objectives':
            [
                {
                    'name': 'acc',
                    'threshold': 1.0,
                    'limits': [0, 1.0],
                    'lower_is_better': False,
                },
                {
                    'name': 'memory',
                    'threshold': 21.0,
                    'limits': [0, 21.],
                    'lower_is_better': True,
                }
            ],

        'track_metrics': [],

        'benchmark_type': 'HPOBenchContainer',
        # 'benchmark_type': 'HPOBenchLocal',

        'benchmark_import': {
            'import_from': 'surrogates.yahpo_gym',
            'benchmark_name': 'YAHPOGymMOBenchmark',
            'use_local': False,
        },

        'benchmark_parameters': {
            'scenario': 'rbv2_xgboost',
            'instance': '28',
            'multi_thread': False,
        },

        'optimization_parameters': {
            'tae_limit': 500,
            'wallclock_limit_in_s': 1800,
            'estimated_cost_limit_in_s': 1800,
            'is_surrogate': True,
        },
    }

    optimizer_settings = {
        # 'optimizer_import': {
        #     'import_from': 'BenchmarkToolsOptimizers.optimizers.random_search.random_search',
        #     'import_name': 'RandomSearchOptimizer',
        # }
        'optimizer_import': {
            'import_from': 'BenchmarkToolsOptimizers.optimizers.bayesian_optimizer.simple_gp',
            'import_name': 'SimpleBOOptimizer',
        }
    }

    benchmark_settings['name'] = benchmark_name
    optimizer_settings['name'] = optimizer_name

    # -------------------- CREATE THE OUTPUT DIRECTORY -----------------------------------------------------------------
    output_path = Path(output_path) / benchmark_name / optimizer_name / str(run_id)
    if not debug:
        if (output_path / BenchmarkToolsConstants.FINISHED_FLAG).exists():
            logger.warning('The Directory already exists and has already finished')
            raise AlreadyFinishedException(f'The run directory exists and is not empty. {output_path}')
    Path(output_path).mkdir(exist_ok=True, parents=True)
    # -------------------- CREATE THE OUTPUT DIRECTORY -----------------------------------------------------------------

    # -------------------- PREPARE STEPS -------------------------------------------------------------------------------
    benchmark_object = load_object(**benchmark_settings['benchmark_import'])
    if benchmark_settings['benchmark_import']['benchmark_name'] == 'BOTestFunctionBenchmark':
        benchmark: BOTestFunctionBenchmark = benchmark_object(
            **benchmark_settings['benchmark_parameters']
        )

    elif benchmark_settings['benchmark_type'] == 'HPOBenchLocal':
        raise NotImplementedError()

    elif benchmark_settings['benchmark_import']['benchmark_name'] == 'HPOBenchContainerInterface':
        main_benchmark = benchmark_object(
            settings=benchmark_settings, rng=run_id, keep_alive=True
        )
        main_benchmark.init_benchmark()
        benchmark: HPOBenchContainerInterface = benchmark_object(
            settings=benchmark_settings, rng=run_id,
            socket_id=main_benchmark.socket_id, keep_alive=False
        )
    else:
        raise ValueError('Unknown Benchmark Type')

    configuration_space = benchmark.get_configuration_space(seed=run_id)

    optimizer_type = load_object(**optimizer_settings['optimizer_import'])
    optimizer = optimizer_type(
        optimizer_settings=optimizer_settings, benchmark_settings=benchmark_settings, configuration_space=configuration_space
    )

    experiment = MultiObjectiveExperiment(
        benchmark_settings=benchmark_settings,
        benchmark=benchmark,
        optimizer_settings=optimizer_settings,
        optimizer=optimizer,
        output_path=output_path,
        run_id=run_id,
    )

    # -------------------- PREPARE STEPS -------------------------------------------------------------------------------
    experiment.setup()

    budget_limit_reached, error = experiment.run()

    if error is not None:
        raise error

    # -------------------- CLEAN UP ------------------------------------------------------------------------------------
    try:
        benchmark.__del__()
    except Exception:
        pass

    try:
        main_benchmark.__del__()
    except Exception:
        pass

    logger.info('Finished Experiment')
    # -------------------- CLEAN UP ------------------------------------------------------------------------------------

    # -------------------- EVALUATION ----------------------------------------------------------------------------------
    trials_dataframe = experiment.study.trials_dataframe()
    trials_dataframe.to_csv(output_path / 'optimization_history.csv')
    # -------------------- EVALUATION ----------------------------------------------------------------------------------

    # -------------------- VISUALIZATION -------------------------------------------------------------------------------
    import optuna
    fig = optuna.visualization.plot_pareto_front(experiment.study, target_names=experiment.objective_names)
    fig.show()
    fig = optuna.visualization.plot_optimization_history(experiment.study, target=lambda trial: trial.values[0], target_name=experiment.objective_names[0])
    fig.show()
    # -------------------- VISUALIZATION -------------------------------------------------------------------------------
