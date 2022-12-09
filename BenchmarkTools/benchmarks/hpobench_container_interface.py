from typing import Union, Dict, List

import ConfigSpace as CS
import numpy as np
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from hpobench.config import config_file
from hpobench.abstract_benchmark import AbstractSingleObjectiveBenchmark, AbstractMultiObjectiveBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient, AbstractMOBenchmarkClient
from torch import Tensor

from BenchmarkTools.utils.loader_tools import load_object, load_benchmark

# from MOHPOBenchExperimentUtils.core.multiobjective_experiment_new import MultiObjectiveSimpleExperiment
# from MOHPOBenchExperimentUtils.core.target_normalization import TargetScaler, get_scaler
# from MOHPOBenchExperimentUtils.utils.ax_utils import get_ax_metrics_from_metric_dict
# from MOHPOBenchExperimentUtils.utils.hpobench_utils import load_benchmark, HPOBenchMetrics
# from MOHPOBenchExperimentUtils.utils.search_space_utils import convert_config_space_to_ax_space, \
#     wrapper_change_hp_in_configspace


class HPOBenchContainerInterface(AbstractMultiObjectiveBenchmark):
    def __init__(self, settings: Dict, rng: int, socket_id=None, keep_alive=True, **kwargs):
        self.settings = settings
        self.socket_id = socket_id
        self.rng = rng
        self.benchmark = None
        self.keep_alive = keep_alive
        # super(HPOBenchContainerInterface, self).__init__(**kwargs)

    def init_benchmark(self) -> None:
        if self.benchmark is None:
            benchmark_object = load_benchmark(**self.settings['benchmark_import'])
            self.benchmark: Union[AbstractMOBenchmarkClient, AbstractBenchmarkClient] = benchmark_object(
                container_source=config_file.container_source,
                rng=self.rng,
                **self.settings.get('benchmark_parameters', {}),
                socket_id=self.socket_id,
            )
            self.socket_id = self.benchmark.socket_id

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        self.init_benchmark()
        cs = self.benchmark.get_configuration_space(seed=seed)
        if not self.keep_alive:
            self.benchmark = None
        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        self.init_benchmark()
        fs = self.benchmark.get_fidelity_space(seed=seed)
        if not self.keep_alive:
            self.benchmark = None
        return fs

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        self.init_benchmark()
        results = self.benchmark.objective_function(configuration=configuration, fidelity=fidelity, rng=rng, **kwargs)
        if not self.keep_alive:
            self.benchmark = None
        return results
    
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        self.init_benchmark()
        return self.benchmark.objective_function_test(configuration=configuration, fidelity=fidelity, rng=rng, **kwargs)

    def get_meta_information(self) -> Dict:
        self.init_benchmark()
        meta = self.benchmark.get_meta_information()
        if not self.keep_alive:
            self.benchmark = None
        return meta

    def get_objective_names(self) -> List:
        self.init_benchmark()
        obj_names = self.benchmark.get_objective_names()
        if not self.keep_alive:
            self.benchmark = None
        return obj_names
