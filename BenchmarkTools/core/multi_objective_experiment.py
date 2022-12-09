import datetime
import os
from pathlib import Path
from time import time
from typing import Dict

import ConfigSpace as CS
import optuna

from BenchmarkTools import logger
from BenchmarkTools.utils.constants import BenchmarkToolsConstants
from BenchmarkTools.utils.exceptions import BudgetExhaustedException


class MultiObjectiveExperiment:
    def __init__(
            self,
            optimizer,
            optimizer_settings: Dict,
            benchmark,
            benchmark_settings: Dict,
            output_path: Path,
            run_id: int = 0,
    ):
        self.output_path = Path(output_path)
        self.optimizer = optimizer
        self.optimizer_settings = optimizer_settings
        self.benchmark = benchmark
        self.benchmark_settings = benchmark_settings
        self.run_id = run_id
        self.study_name = self.benchmark_settings['name'] + '_' + self.optimizer_settings['name'] + '_' + str(self.run_id)

        self.search_space = benchmark.get_configuration_space(seed=run_id)
        self.optuna_distributions = self._configuration_space_cs_optuna_distributions(self.search_space)

        self.initial_time = time()
        self.num_configs_evaluated = -1
        self.accumulated_surrogate_cost = 0
        self.time_for_saving = 0
        self.used_wallclock_time = 0
        self.used_total_cost = 0

        self.tae_limit = self.benchmark_settings['optimization_parameters']['tae_limit']
        self.wallclock_limit_in_s = self.benchmark_settings['optimization_parameters']['wallclock_limit_in_s']
        self.estimated_cost_limit_in_s = self.benchmark_settings['optimization_parameters']['estimated_cost_limit_in_s']
        self.is_surrogate = self.benchmark_settings['optimization_parameters']['is_surrogate']

        self.study: [optuna.Study, None] = None
        self.storage_path = self.output_path / BenchmarkToolsConstants.DATABASE_NAME

        self.objective_names = [obj['name'] for obj in self.benchmark_settings['objectives']]
        self.directions = [
            'minimize' if obj['lower_is_better'] else 'maximize'
            for obj in self.benchmark_settings['objectives']
        ]

    def setup(self):
        self.try_start_study()
        self.start_time()
        self.optimizer.link_experiment(experiment=self)

    def start_time(self):
        self.initial_time = time()
        self.used_wallclock_time = 0
        self.used_total_cost = 0
        self.accumulated_surrogate_cost = 0

    def try_start_study(self):
        """
        Start a study if it does not exist. If it exist already, connect to it.
        """
        # TODO: Adaptive storage dir
        # TODO: Adaptive study name
        load_if_exists = False

        if os.name == 'posix':
            storage = f"sqlite:////{self.storage_path.resolve()}"
        else:
            storage = f"sqlite:///{self.storage_path.resolve()}"

        logger.info(f'Try to connect to study {self.study_name} at {storage}')
        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                directions=self.directions,
                load_if_exists=load_if_exists,
            )
        except optuna.exceptions.DuplicatedStudyError:
            optuna.delete_study(
                study_name=self.study_name,
                storage=storage,
            )
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                directions=self.directions,
                load_if_exists=load_if_exists,
            )

        # Add experiment information to the study
        experiment_information = {
            'benchmark_name': self.benchmark_settings['name'],
            'optimizer_name': self.optimizer_settings['name'],
            'run_id': self.run_id,
            'objective_names': self.objective_names,
            'directions': self.directions,
        }
        for k, v in experiment_information.items():
            self.study.set_user_attr(k, v)

        # optuna.delete_study
        # TODO: When continue: update optimization limits

    def evaluate_configuration(
            self,
            configuration: Dict,
            fidelity: Dict = None,
    ) -> Dict:

        """
        This function evaluates a configuration with a given fidelity (optional).
        It tracks all trials / results using the optuna study.

        Args:
            configuration: Dict
            fidelity: Dict

        Returns:
            Result object is in HPBench Result format.
            Dict:
                function_value - Dict
                cost - int, float
                info - Dict

        """
        if isinstance(configuration, CS.Configuration):
            configuration = configuration.get_dictionary()
        # --------------------------- UPDATE TIMER ---------------------------------------------------------------------
        self.used_wallclock_time = time() - self.initial_time
        # --------------------------- UPDATE TIMER ---------------------------------------------------------------------

        # --------------------------- CHECK RUN LIMITS -----------------------------------------------------------------
        # Check that the wallclock time limit is not reached
        if self.used_wallclock_time >= self.wallclock_limit_in_s:
            logger.warning(f'Wallclock Time Limit Reached')
            raise BudgetExhaustedException(f'Wallclock Time Limit Reached: {self.wallclock_limit_in_s}')

        # If we evaluate a surrogate model, then we also care about the predicted costs
        self.used_total_cost = self.used_wallclock_time + self.accumulated_surrogate_cost
        if self.is_surrogate and (self.used_total_cost >= self.estimated_cost_limit_in_s):
            logger.warning(f'Surrogate Costs Limit Reached')
            raise BudgetExhaustedException(f'Total Time Limit Reached: {self.estimated_cost_limit_in_s}')

        # By default, the tae limit is not active. If it is set to -1, 1000 * #hps is used.
        if self.tae_limit is not None and self.num_configs_evaluated > self.tae_limit:
            logger.warning('Target Execution limit is reached.')
            raise BudgetExhaustedException(
                f'Total Number of Target Executions Limit is reached. {self.num_configs_evaluated}'
            )
        # --------------------------- CHECK RUN LIMITS -----------------------------------------------------------------

        # --------------------------- PREPARE CONFIG AND FIDELITY ------------------------------------------------------
        if fidelity is not None:
            configuration = {**configuration, **fidelity}

        # # We have to add missing parameters (even if they are not active), since ax will raise an
        # # error otherwise. HPObench will remove them before running the configuration, therefore
        # # we should be okay with this solution. (more or less)
        # for key in self.cs_search_space.get_hyperparameter_names():
        #     if key not in configuration:
        #         missing_hp = self.cs_search_space.get_hyperparameter(key)
        #         configuration[key] = missing_hp.default_value
        # --------------------------- PREPARE CONFIG AND FIDELITY ------------------------------------------------------

        # --------------------------- EVALUATE TRIAL AND LOG RESULTS ---------------------------------------------------
        self.num_configs_evaluated += 1

        # Create a trial that is assigned to the function evaluation
        trial = optuna.create_trial(
            state=optuna.trial.TrialState.RUNNING,
            params=configuration,
            distributions={name: dist for name, dist in self.optuna_distributions.items() if name in configuration.keys()},
        )
        # TODO: Assigning a run_id might be better with a lock! In case of parallel optimization
        trial.number = self.num_configs_evaluated

        # Query the benchmark. The Output should be in the return format of the HPOBench-Experiments
        result_dict: Dict = self.benchmark.objective_function(configuration=configuration, fidelity=fidelity)

        # Log the results to the optuna study
        trial.values = [result_dict['function_value'][obj_name] for obj_name in self.objective_names]
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.datetime_complete = datetime.datetime.now()
        self.study.add_trial(trial)
        # --------------------------- EVALUATE TRIAL AND LOG RESULTS ---------------------------------------------------

        # --------------------------- UPDATE TIMER --------------------------------
        if self.is_surrogate:
            self.accumulated_surrogate_cost += result_dict['cost']
        # --------------------------- UPDATE TIMER --------------------------------

        # --------------------------- POST PROCESS --------------------------------
        eval_interval = 100 if self.is_surrogate else 10
        if (self.num_configs_evaluated % eval_interval) == 0:
            remaining_time = self.wallclock_limit_in_s - self.used_wallclock_time
            logger.info(f'WallClockTime left: {remaining_time:10.4f}s ({remaining_time/3600:.4f}h)')
            if self.is_surrogate:
                remaining_time = self.estimated_cost_limit_in_s - self.used_total_cost
                logger.info(f'EstimatedTime left: {remaining_time:10.4f}s ({remaining_time/3600:.4f}h)')
            if self.tae_limit is not None:
                logger.info(f'Number of TAE: {self.num_configs_evaluated:10d}|{self.tae_limit}')
            else:
                logger.info(f'Number of TAE: {self.num_configs_evaluated:10d}| INF')

        # TODO: Callbacks
        #  Example: Add Save-Callback
        # save_interval = 10000 if self.is_surrogate else 10
        # if (self.num_configs_evaluated % save_interval) == 0:
        #     # Saving intermediate results is pretty expensive
        #     t = time()
        #     save_output(self, self.output_path, finished=False, surrogate=self.is_surrogate)
        #     time_for_saving = time() - t
        #     logger.info(f'Saved Experiment to Pickle took {time_for_saving:.2f}s')

        return result_dict

    def run(self):
        budget_limit_reached = True
        try:
            self.optimizer.run()
        except BudgetExhaustedException as err:
            logger.info(f'Budget Limit has been reached: str(err)')

        except Exception as err:
            logger.warning(f'The optimization run has crashed with error: {err}')
            budget_limit_reached = False
            return budget_limit_reached, err

        logger.info(f'Optimization procedure has finished.')

        logger.info('Stop the experiment gracefully.')
        logger.info('Create finish-flag')
        flag = self.output_path / str(BenchmarkToolsConstants.FINISHED_FLAG)
        flag.touch(exist_ok=True)

        return budget_limit_reached, None

    @staticmethod
    def _configuration_space_cs_optuna_distributions(configuration_space: CS.ConfigurationSpace) \
            -> Dict[str, optuna.distributions.BaseDistribution]:
        """
        Derive the optuna distributions per parameter from the CS.ConfigurationSpace.

        Args:
            configuration_space:  CS.ConfigurationSpace
                Searchspace of the benchmark

        Returns:
            Dict[str, BaseDistribution]
                Distributions of the hyperparameters (boundaries)
                    key: name of the hyperparameter
                    value: optuna.distributions.BaseDistribution
        """
        mapping = {
            CS.UniformIntegerHyperparameter: optuna.distributions.IntDistribution,
            CS.UniformFloatHyperparameter: optuna.distributions.FloatDistribution,
            CS.CategoricalHyperparameter: optuna.distributions.CategoricalDistribution,
            CS.OrdinalHyperparameter: optuna.distributions.CategoricalDistribution,
            CS.Constant: optuna.distributions.CategoricalDistribution,
        }

        distributions = {}
        for hp in configuration_space.get_hyperparameters():
            if isinstance(hp, (CS.UniformIntegerHyperparameter, CS.UniformFloatHyperparameter)):
                dist = mapping[type(hp)](low=hp.lower, high=hp.upper, log=hp.log)
            elif isinstance(hp, (CS.CategoricalHyperparameter, CS.OrdinalHyperparameter)):
                dist = mapping[type(hp)](choices=hp.choices)
            elif isinstance(hp, CS.Constant):
                dist = mapping[type(hp)](choices=[hp.value])
            else:
                raise NotImplementedError('Unknown HP Type')
            distributions[hp.name] = dist
        return distributions

