from pathlib import Path
from typing import Dict, Union

from omegaconf import DictConfig

from BenchmarkTools import logger
from BenchmarkTools.benchmarks.hpobench_container_interface import HPOBenchContainerInterface
from BenchmarkTools.benchmarks.botorch_black_box_interface import BotorchBlackBoxBenchmark
from BenchmarkTools.core.constants import BenchmarkToolsConstants, BenchmarkTypes
from BenchmarkTools.core.exceptions import AlreadyFinishedException
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment
from BenchmarkTools.core.run_tracking import wrapper_track_run_stats
from BenchmarkTools.utils.loader_tools import load_object


@wrapper_track_run_stats
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
    assert benchmark_settings['benchmark_type'] in [t.name for t in BenchmarkTypes], \
        f'BenchmarkType has to be in {[t.name for t in BenchmarkTypes]} but was {benchmark_settings["benchmark_type"]}'

    if BenchmarkTypes[benchmark_settings['benchmark_type']] is BenchmarkTypes.BOTORCH_BLACK_BOX:
        benchmark_object = load_object(**benchmark_settings['benchmark_import'])
        benchmark: BotorchBlackBoxBenchmark = benchmark_object(
            benchmark_settings=benchmark_settings,
        )

    elif BenchmarkTypes[benchmark_settings['benchmark_type']] is BenchmarkTypes.HPOBENCH_CONTAINER:
        main_benchmark = HPOBenchContainerInterface(
            benchmark_settings=benchmark_settings, rng=run_id, keep_alive=True
        )
        main_benchmark.init_benchmark()
        benchmark: HPOBenchContainerInterface = HPOBenchContainerInterface(
            benchmark_settings=benchmark_settings, rng=run_id,
            socket_id=main_benchmark.socket_id, keep_alive=False
        )
    else:
        raise ValueError('Unknown Benchmark Type')

    # Init the optimizer. It has a standard form.
    # TODO: Write the optimizer so that it supports parallel executions
    configuration_space = benchmark.get_configuration_space(seed=run_id)

    optimizer_type = load_object(**optimizer_settings['optimizer_import'])
    optimizer = optimizer_type(
        optimizer_settings=optimizer_settings,
        benchmark_settings=benchmark_settings,
        configuration_space=configuration_space
    )

    # The experiment host all HPO related information, e.g. time limits. It stops the HPO run if the HPO has hit the
    # run limits. It also takes care of the result logging.
    experiment = MultiObjectiveExperiment(
        benchmark_settings=benchmark_settings,
        benchmark=benchmark,
        optimizer_settings=optimizer_settings,
        optimizer=optimizer,
        output_path=output_path,
        run_id=run_id,
    )

    experiment.setup()
    # -------------------- PREPARE STEPS -------------------------------------------------------------------------------

    # -------------------- EXECUTE BENCHMARK ---------------------------------------------------------------------------
    budget_limit_reached, error = experiment.run()

    if error is not None:
        raise error
    # -------------------- EXECUTE BENCHMARK ---------------------------------------------------------------------------

    # -------------------- CLEAN UP ------------------------------------------------------------------------------------
    def clean_up():
        try:
            benchmark.__del__()
        except Exception:
            pass

        try:
            main_benchmark.__del__()
        except Exception:
            pass

    clean_up()
    logger.info('Finished Experiment')
    # -------------------- CLEAN UP ------------------------------------------------------------------------------------

    # -------------------- EVALUATION ----------------------------------------------------------------------------------
    trials_dataframe = experiment.study.trials_dataframe()
    rename_dict = {
        **{f'values_{i}': obj_name for i, obj_name in enumerate(experiment.objective_names)},
        **{f'params_{f_name}': f'fidelity_{f_name}' for f_name in experiment.fidelity_names}
    }
    trials_dataframe = trials_dataframe.rename(columns=rename_dict)
    trials_dataframe.to_csv(output_path / BenchmarkToolsConstants.OPT_HISTORY_NAME)
    logger.info(f'Run results exported to {output_path / "optimization_history.csv"}')
    # -------------------- EVALUATION ----------------------------------------------------------------------------------

    # -------------------- VISUALIZATION -------------------------------------------------------------------------------
    # import optuna
    # fig = optuna.visualization.plot_pareto_front(
    #     experiment.study, target_names=experiment.objective_names
    # )
    # fig.show()

    # for i in range(len(experiment.objective_names)):
    #     fig = optuna.visualization.plot_optimization_history(
    #         experiment.study, target=lambda trial: trial.values[i], target_name=experiment.objective_names[i]
    #     )
    #     fig.show()
    # -------------------- VISUALIZATION -------------------------------------------------------------------------------
