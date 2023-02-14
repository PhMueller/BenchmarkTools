from typing import Dict

import ray

from BenchmarkTools import logger
from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.constants import BenchmarkTypes
from BenchmarkTools.benchmarks.hpobench_container_interface import HPOBenchContainerInterface
from BenchmarkTools.benchmarks.botorch_black_box_interface import BotorchBlackBoxBenchmark


class Worker(object):
    def __init__(self, worker_id: int = 0, **kwargs):
        self.worker_id = worker_id
        logger.info(f'Worker {self.worker_id} instantiated')

    def compute(self, configuration: Dict, fideltiy: Dict) -> Dict:
        raise NotImplementedError()

    def is_alive(self):
        return True

    def stop(self):
        """ You can add clean up functionality to stop the worker here! """
        pass

    def _process_job(self, job: Job):
        """
        The worker receives a job object that stores the configuration, the fidelity as well as the results.
        It also tracks some time information.

        This function extract the necessary information from the job object and passes it to the compute() function.
        """

        job.start_job()
        logger.debug(f'Worker {self.worker_id}: Obj Func: Start - config {job.configuration}')

        result_dict = self.compute(job.configuration, job.fidelity)
        job.register_result(result_dict=result_dict)

        logger.debug(f'Worker {self.worker_id}: Obj Func: return {result_dict}')
        return job

    def __del__(self):
        self.stop()


@ray.remote
class BenchmarkWorker(Worker):
    """
    Example class for a custom worker.
    It needs a re-implementation of the compute- and (sometimes) stop-function.
    """
    def __init__(self, benchmark_settings: Dict, worker_id: int = 0, rng: int = 0, **kwargs):
        super().__init__(worker_id=worker_id, **kwargs)

        self.rng = rng
        self.benchmark_settings = benchmark_settings
        self.benchmark = None

        # initialize the benchmark
        self._load_benchmark()

    def compute(self, configuration: Dict, fideltiy: Dict) -> Dict:
        result = self.benchmark.objective_function(configuration, fideltiy)
        return result

    def get_configuration_space(self, seed: int = None):
        return self.benchmark.get_configuration_space(seed=seed)

    def get_fidelity_space(self, seed: int = None):
        return self.benchmark.get_fidelity_space(seed=seed)

    def stop(self):
        """ Stop function to gracefully stop the worker. Clean up references to the (pyro/hpobench) benchmarks """
        try:
            self.benchmark.__del__()
        except Exception as e:
            print(f'Shutdown - {self.worker_id} - Exception {e} ')

        try:
            # If it is a hpobenchcontainer benchmark, we also have a ref to a background benchmark.
            self._main_benchmark.__del__()
        except Exception:
            pass

        print(f'Shutdown - {self.worker_id} - Done')

    def _load_benchmark(self):
        """Load the correct benchmark and initialize it. """
        assert self.benchmark_settings['benchmark_type'] in [t.name for t in BenchmarkTypes], \
            f'BenchmarkType has to be in {[t.name for t in BenchmarkTypes]}, ' \
            f'but was {self.benchmark_settings["benchmark_type"]}'

        self.benchmark = None

        if BenchmarkTypes[self.benchmark_settings['benchmark_type']] is BenchmarkTypes.BOTORCH_BLACK_BOX:
            self.benchmark = BotorchBlackBoxBenchmark(
                benchmark_settings=self.benchmark_settings,
            )

        elif BenchmarkTypes[self.benchmark_settings['benchmark_type']] is BenchmarkTypes.HPOBENCH_CONTAINER:
            self._main_benchmark = HPOBenchContainerInterface(
                benchmark_settings=self.benchmark_settings, rng=self.rng, keep_alive=True
            )
            self._main_benchmark.init_benchmark()
            self.benchmark = HPOBenchContainerInterface(
                benchmark_settings=self.benchmark_settings, rng=self.rng,
                socket_id=self._main_benchmark.socket_id, keep_alive=False
            )
            self.benchmark.init_benchmark()

        else:
            raise ValueError('Unknown Benchmark Type')
