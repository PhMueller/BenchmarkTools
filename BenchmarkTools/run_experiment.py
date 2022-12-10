from pathlib import Path
from typing import Dict, Union
from omegaconf import DictConfig

from BenchmarkTools import logger
from BenchmarkTools.benchmarks.toy_benchmark import BOTestFunctionBenchmark
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment
from BenchmarkTools.optimizers.random_search import RandomSearchOptimizer
from BenchmarkTools.utils.constants import BenchmarkToolsConstants
from BenchmarkTools.utils.exceptions import AlreadyFinishedException
from BenchmarkTools.utils.loader_tools import load_object


def run(benchmark_name: str,
        benchmark_settings: Union[Dict, DictConfig],
        optimizer_name: str,
        optimizer_settings: Union[Dict, DictConfig],
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

    # -------------------- PREPARE STEPS -------------------------------------------------------------------------------
    if benchmark_settings['benchmark_import']['import_from'].startswith('hpobench.container'):
        from BenchmarkTools.benchmarks.hpobench_container_interface import HPOBenchContainerInterface
        main_benchmark = HPOBenchContainerInterface(
            settings=benchmark_settings, rng=run_id, keep_alive=True
        )
        main_benchmark.init_benchmark()
        benchmark: HPOBenchContainerInterface = HPOBenchContainerInterface(
            settings=benchmark_settings, rng=run_id,
            socket_id=main_benchmark.socket_id, keep_alive=False
        )
    else:
        benchmark_object = load_object(**benchmark_settings['benchmark_import'])
        benchmark: BOTestFunctionBenchmark = benchmark_object(
            **benchmark_settings['benchmark_parameters']
        )

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
    rename_dict = {f'values_{i}': obj_name for i, obj_name in enumerate(experiment.objective_names)}
    trials_dataframe = trials_dataframe.rename(columns=rename_dict)
    trials_dataframe.to_csv(output_path / BenchmarkToolsConstants.OPT_HISTORY_NAME)
    # -------------------- EVALUATION ----------------------------------------------------------------------------------

    # -------------------- VISUALIZATION -------------------------------------------------------------------------------
    import optuna
    fig = optuna.visualization.plot_pareto_front(
        experiment.study, target_names=experiment.objective_names
    )
    fig.show()

    for i in range(len(experiment.objective_names)):
        fig = optuna.visualization.plot_optimization_history(
            experiment.study, target=lambda trial: trial.values[i], target_name=experiment.objective_names[i]
        )
        fig.show()
    # -------------------- VISUALIZATION -------------------------------------------------------------------------------
