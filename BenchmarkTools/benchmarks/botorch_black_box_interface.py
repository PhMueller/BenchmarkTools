from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
import torch
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from hpobench.abstract_benchmark import AbstractBenchmark
from torch import Tensor, normal

from BenchmarkTools.utils.loader_tools import load_object
from BenchmarkTools.core.constants import BenchmarkToolsTrackMetrics


class BotorchBlackBoxBenchmark(AbstractBenchmark):
    def __init__(self, benchmark_settings: Dict,  **kwargs):

        self.benchmark_settings: Dict = benchmark_settings
        self.function_name: str = benchmark_settings['benchmark_parameters']['function_name']
        self.function_kwargs: Dict = benchmark_settings['benchmark_parameters'].get('function_kwargs', None)
        if self.function_kwargs is None:
            self.function_kwargs = {}

        function_obj = load_object(import_from='botorch.test_functions.multi_objective', import_name=self.function_name)
        self.function: MultiObjectiveTestProblem = function_obj(**self.function_kwargs)

        self.input_dim = self.function.bounds.shape[1]
        self.output_dim = self.function.num_objectives

        # Following https://arxiv.org/pdf/2105.08195.pdf, the authors add zero-mean gaussian noise to the objectives
        # depending on their ranges
        # TODO: It is not entirely clear how to set the ranges.
        self.objective_noise = self.benchmark_settings['benchmark_parameters'].get('objective_noise', None)
        if self.objective_noise == 0:
            self.objective_noise = None

        if self.objective_noise is not None:
            bounds_diff = torch.Tensor([m['limits'][1] - m['limits'][0] for m in self.benchmark_settings['objectives']])
            self.objective_noise = self.objective_noise * bounds_diff

        super(BotorchBlackBoxBenchmark, self).__init__(**kwargs)

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                f'x{i+1}',
                lower=self.function.bounds[0, i],
                upper=self.function.bounds[1, i],
                default_value=self.function.bounds[0, i],
            ) for i in range(self.input_dim)
        ])
        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=seed)
        #  TODO: This is a single fidelity benchmark. Does that work?
        #        Alternative: Use cosnt fidelity
        # cs.add_hyperparameter(
        #     CS.Constant('fidelity', 1.0)
        #     CS.UniformFloatHyperparameter('fidelity', lower=0.0, upper=1.0, default_value=1.0)
        # )
        return cs

    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[Dict, CS.Configuration, None] = None,
            rng: Union[np.random.RandomState, int, None] = None,
            **kwargs
    ) -> Dict:

        configuration_np = [configuration[f'x{i+1}'] for i in range(self.input_dim)]
        configuration_np = np.array(configuration_np).reshape((-1, len(configuration_np)))
        configuration_torch = Tensor(configuration_np)

        objective_value = self.function.evaluate_true(configuration_torch)

        if self.objective_noise is not None:
            additional_noise = normal(torch.zeros(self.output_dim), self.objective_noise)
            objective_value = objective_value + additional_noise

        function_values = {
            metric['name']: objective_value[..., i].item()
            for i, metric in enumerate(self.benchmark_settings['objectives'])
        }

        # These botorch functions are currently not MF benchmarks and they do not have a cost value.
        # We set it to 1, so that the cost value corresponds to the number of function evaluations
        # TODO: think about that one again.
        cost = 1

        # TODO: Do we have some information for the info field?
        return {
            BenchmarkToolsTrackMetrics.FUNCTION_VALUE_FIELD: function_values,
            BenchmarkToolsTrackMetrics.COST: cost,
            BenchmarkToolsTrackMetrics.INFO_FIELD: {}
        }

    def objective_function_test(self, **kwargs) -> Dict:
        return self.objective_function(**kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {
            'name': 'Botorch Test Functions',
            'code': 'https://github.com/pytorch/botorch/blob/main/botorch/test_functions',
            'reference':
                'S. Daulton, M. Balandat, and E. Bakshy. '
                'Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. '
                'Advances in Neural Information Processing Systems 34, 2021'
        }
