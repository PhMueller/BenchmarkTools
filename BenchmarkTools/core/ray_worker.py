from typing import Dict

import ray

from BenchmarkTools import logger
from BenchmarkTools.core.ray_job import Job


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

        job.start_job()
        logger.debug(f'Worker {self.worker_id}: Obj Func: Start - config {job.configuration}')

        result_dict = self.compute(job.configuration, job.fidelity)
        job.register_result(result_dict=result_dict)

        logger.debug(f'Worker {self.worker_id}: Obj Func: return {result_dict}')
        return job


@ray.remote
class BenchmarkWorker(Worker):
    """
    Example class for a custom worker.
    It needs a re-implementation of the compute and maybe stop function.
    """
    def __init__(self, benchmark_parameters: Dict, worker_id: int = 0):
        super().__init__(worker_id=worker_id)

        from hpobench.container.benchmarks.surrogates.yahpo_gym import YAHPOGymMOBenchmark
        self.benchmark = YAHPOGymMOBenchmark(**benchmark_parameters)

    def compute(self, configuration: Dict, fideltiy: Dict) -> Dict:
        result = self.benchmark.objective_function(configuration, fideltiy)
        return result

    def get_configuration_space(self):
        return self.benchmark.get_configuration_space()

    def stop(self):
        try:
            self.benchmark._shutdown()
        except Exception as e:
            print(f'Shutdown - {self.worker_id} - Exception {e} ')
        print(f'Shutdown - {self.worker_id} - Done')

    def __del__(self):
        self.stop()
