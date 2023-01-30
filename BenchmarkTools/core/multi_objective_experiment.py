import datetime
import os
from pathlib import Path
from typing import Dict

import ConfigSpace as CS
import optuna

from BenchmarkTools import logger
from BenchmarkTools.core.constants import BenchmarkToolsConstants
from BenchmarkTools.core.exceptions import BudgetExhaustedException


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

        """
        This object contains all necessary information to run an experiment.
        It combines tracking limits, calling the benchmark, and executing the optimizer.

        Args:
            optimizer: obj
            optimizer_settings: Dict
            benchmark: obj
            benchmark_settings: Dict
            output_path: Path
            run_id: int
        """

        self.output_path = Path(output_path)
        self.optimizer = optimizer
        self.optimizer_settings = optimizer_settings
        self.benchmark = benchmark
        self.benchmark_settings = benchmark_settings
        self.run_id = run_id
        self.study_name = self.benchmark_settings['name'] + '_' + self.optimizer_settings['name'] + '_' + str(self.run_id)

        self.search_space = benchmark.get_configuration_space(seed=run_id)
        self.fidelity_space = benchmark.get_fidelity_space(seed=run_id)
        self.fidelity_names = [hp.name for hp in self.fidelity_space.get_hyperparameters()]

        self.optuna_distributions = self._configuration_space_cs_optuna_distributions(self.search_space)
        self.optuna_distributions_fs = self._configuration_space_cs_optuna_distributions(self.fidelity_space)
        self.optuna_distributions = {**self.optuna_distributions, **self.optuna_distributions_fs}

        from BenchmarkTools.core.bookkeeper import FileBookKeeper
        self.bookkeeper = FileBookKeeper(
            benchmark_settings=self.benchmark_settings,
            lock_dir=self.output_path / 'lock_dir',
            resource_file_dir=self.output_path
        )

        self.study: [optuna.Study, None] = None
        self.storage_path = self.output_path / BenchmarkToolsConstants.DATABASE_NAME

        self.is_surrogate = self.benchmark_settings['optimization_parameters']['is_surrogate']
        self.objective_names = [obj['name'] for obj in self.benchmark_settings['objectives']]
        self.directions = [
            'minimize' if obj['lower_is_better'] else 'maximize'
            for obj in self.benchmark_settings['objectives']
        ]

    def setup(self):
        self.try_start_study()
        self.bookkeeper.start_timer()
        self.optimizer.link_experiment(experiment=self)
        self.optimizer.init(seed=self.run_id)

    def try_start_study(self):
        """
        Start a study if it does not exist. If it exist already, connect to it.
        """
        # TODO: Adaptive storage dir
        # TODO: Adaptive study name
        # TODO: Add function to continue a run?
        # What happens if a study is already present? -> Delete it or continue it?
        # For now delete it. But a continue option would be nice.
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

    def evaluate_configuration(
            self,
            configuration: Dict,
            fidelity: Dict = None,
    ) -> Dict:

        """
        # TODO: Make this function multi-thread safe:
        #       1) Extract tracking of time limits: Maybe write a SQLite database or in a file with a lock.
        #       2) Extract tracking of run results (this should actually already be multi-thread safe by optuna)

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

        # --------------------------- CHECK RUN LIMITS -----------------------------------------------------------------
        self.bookkeeper.check_optimization_limits()
        # --------------------------- CHECK RUN LIMITS -----------------------------------------------------------------

        # --------------------------- PREPARE CONFIG AND FIDELITY ------------------------------------------------------
        # We combine the configuration and the fidelity dict to make it trackable for optuna.
        if fidelity is not None:
            optuna_configuration = {**configuration, **fidelity}
        else:
            optuna_configuration = configuration
        # --------------------------- PREPARE CONFIG AND FIDELITY ------------------------------------------------------

        # --------------------------- EVALUATE TRIAL AND LOG RESULTS ---------------------------------------------------
        new_trial_number = self.bookkeeper.increase_num_tae_calls_and_get_num_tae_calls(delta_tae_calls=1)

        # Create a trial that is assigned to the function evaluation
        trial = optuna.create_trial(
            state=optuna.trial.TrialState.RUNNING,
            params=optuna_configuration,
            distributions={
                name: dist for name, dist in self.optuna_distributions.items() if name in optuna_configuration.keys()
            },
        )
        trial.number = new_trial_number

        # Query the benchmark. The Output should be in the return format of the HPOBench-Experiments
        result_dict: Dict = self.benchmark.objective_function(configuration=configuration, fidelity=fidelity)

        # Log the results to the optuna study
        trial.values = [result_dict['function_value'][obj_name] for obj_name in self.objective_names]

        # TODO: Set additional information!
        additional_info = result_dict['info']
        additional_info.update(**{'cost': result_dict['cost']})
        for k, v in additional_info.values:
            trial.set_user_attr(f'info_{k}', v)

        trial.state = optuna.trial.TrialState.COMPLETE
        trial.datetime_complete = datetime.datetime.now()
        self.study.add_trial(trial)
        # --------------------------- EVALUATE TRIAL AND LOG RESULTS ---------------------------------------------------

        # --------------------------- UPDATE TIMER --------------------------------
        if self.is_surrogate:
            self.bookkeeper.increase_used_resources(delta_surrogate_cost=result_dict['cost'])
        # --------------------------- UPDATE TIMER --------------------------------

        # --------------------------- POST PROCESS --------------------------------
        eval_interval = 100 if self.is_surrogate else 10
        if (new_trial_number % eval_interval) == 0:
            self.bookkeeper.log_currently_used_resources()

        # TODO: Callbacks
        #  Example: Add Save-Callback
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
