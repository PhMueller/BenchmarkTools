from pathlib import Path
from typing import Dict, Union, List

import ray

from omegaconf import DictConfig

from BenchmarkTools import logger
from BenchmarkTools.core.constants import BenchmarkToolsConstants
from BenchmarkTools.core.exceptions import AlreadyFinishedException
from BenchmarkTools.core.multi_objective_experiment import MultiObjectiveExperiment
from BenchmarkTools.core.run_tracking import wrapper_track_run_stats
from BenchmarkTools.utils.loader_tools import load_object
from BenchmarkTools.core.ray_scheduler import Scheduler
from BenchmarkTools.core.ray_worker import BenchmarkWorker


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

    class RayBenchmarkRunner(object):
        def __init__(self, benchmark_settings: Dict, run_id: int, num_workers: int = 1):
            self.benchmark_settings: Dict = benchmark_settings
            self.run_id = run_id
            self.num_workers: int = num_workers

            self.scheduler: Union[Scheduler, None] = None
            self.workers: Union[List[BenchmarkWorker], None] = None

        def start(self):
            self.start_workers()
            self.start_scheduler()

        def start_workers(self):
            self.workers = [
                BenchmarkWorker.remote(benchmark_settings=self.benchmark_settings, worker_id=i, rng=self.run_id)
                for i in range(self.num_workers)
            ]
            # Give the workers some time to start and wait until they are alive and responsible!
            wait_until_alive_futures = [w.is_alive.remote() for w in self.workers]
            ray.wait(wait_until_alive_futures, num_returns=self.num_workers, timeout=None)

        def start_scheduler(self):
            # Init the scheduler and start the submit-jobs- and fetching-results-threads
            self.scheduler = Scheduler(self.workers)
            self.scheduler.run()

        def get_configuration_space(self, seed: int = 0):
            configspace_cs = ray.get(self.workers[0].get_configuration_space.remote(seed=seed))
            return configspace_cs

        def get_fidelity_space(self, seed: int = 0):
            configspace_cs = ray.get(self.workers[0].get_fidelity_space.remote(seed=seed))
            return configspace_cs

    runner = RayBenchmarkRunner(benchmark_settings=benchmark_settings, run_id=run_id, num_workers=1)
    runner.start()

    # Init the optimizer. It has a standard form.
    configuration_space = runner.get_configuration_space(seed=run_id)
    print('test')

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
        optimizer_settings=optimizer_settings,
        optimizer=optimizer,
        runner=runner,
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
            runner.scheduler.shutdown()
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
