from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from hpobench.abstract_benchmark import AbstractBenchmark
from torch import Tensor

from BenchmarkTools.utils.loader_tools import load_object


# from MOHPOBenchExperimentUtils.core.multiobjective_experiment_new import MultiObjectiveSimpleExperiment
# from MOHPOBenchExperimentUtils.core.target_normalization import TargetScaler, get_scaler
# from MOHPOBenchExperimentUtils.utils.ax_utils import get_ax_metrics_from_metric_dict
# from MOHPOBenchExperimentUtils.utils.hpobench_utils import load_benchmark, HPOBenchMetrics
# from MOHPOBenchExperimentUtils.utils.search_space_utils import convert_config_space_to_ax_space, \
#     wrapper_change_hp_in_configspace


class BOTestFunctionBenchmark(AbstractBenchmark):
    def __init__(self, function_name: str, function_kwargs: Dict, **kwargs):

        self.function_name = function_name
        self.function = load_object(import_from='botorch.test_functions.multi_objective', import_name=self.function_name)
        self.function: MultiObjectiveTestProblem = self.function(**function_kwargs)

        super(BOTestFunctionBenchmark, self).__init__(**kwargs)

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                f'x{i+1}',
                lower=self.function.bounds[0, i],
                upper=self.function.bounds[1, i],
                default_value=self.function.bounds[0, i],
            ) for i in range(self.function.bounds.shape[1])
        ])
        return cs  # 2.0.0 for smac

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(
            CS.Constant('fidelity', 1.0)
            # CS.UniformFloatHyperparameter('fidelity', lower=0.0, upper=1.0, default_value=1.0)
        )
        return cs

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        cs = self.get_configuration_space()
        config = [configuration[hp_name] for hp_name in cs.get_hyperparameter_names()]
        config = np.array(config)
        config = config.reshape((-1, len(config)))

        objective_value = self.function.evaluate_true(Tensor(config))

        branin = objective_value[..., 0]
        currin = objective_value[..., 1]

        cost = np.sum(config)

        return {'function_value': {'branin': branin.item(), 'currin': currin.item()},
                'cost': cost.item(),
                'info': {}}

    def objective_function_test(self, **kwargs) -> Dict:
        return self.objective_function(**kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {}
