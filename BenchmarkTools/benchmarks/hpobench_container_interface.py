from typing import Union, Dict, List

import ConfigSpace as CS
import numpy as np
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark
from hpobench.config import config_file
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient, AbstractMOBenchmarkClient

from BenchmarkTools.utils.loader_tools import load_object


class HPOBenchContainerInterface(AbstractMultiObjectiveBenchmark):
    """
    This is a wrapper around an HPOBench Container.
    The idea is to first create a benchmark object that lives in the main process.
    This interface here stores then the socket_id / address to the container. When the interface is queried, it
    establishes a connection to the benchmark.

    That allows us later to easily pass the container to optimizers even though it requires to serialize this particular
    interface/class to send it via pickle to a different process.

    Also, the benchmark has only to be initialized once in the beginning.
    """
    def __init__(self, benchmark_settings: Dict, rng: int, socket_id=None, keep_alive=True, **kwargs):
        """

        Args:
            benchmark_settings:  Dict

            rng: int
                seeding for the benchmark

            socket_id: str, None
                address of the benchmark.
                * If you call this function with `keep_alive=True`, then socket_id has to be `None`
                * otherwise, pass the `socket_id` from the always-running-container.

            keep_alive: bool
                flag indicting if the container should be kept alive.

            **kwargs:
        """
        assert not (not keep_alive and socket_id is None), \
            'You have to provide a socket id, if you only want connect to the benchmark.'

        self.benchmark_settings = benchmark_settings
        self.socket_id = socket_id
        self.rng = rng
        self.benchmark = None
        self.keep_alive = keep_alive
        # super(HPOBenchContainerInterface, self).__init__(**kwargs)

    def init_benchmark(self) -> None:
        if self.benchmark is None:
            benchmark_object = load_object(**self.benchmark_settings['benchmark_import'])
            self.benchmark: Union[AbstractMOBenchmarkClient, AbstractBenchmarkClient] = benchmark_object(
                container_source=config_file.container_source,
                rng=self.rng,
                **self.benchmark_settings.get('benchmark_parameters', {}),
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
