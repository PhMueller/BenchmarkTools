"""
Idea: Use ray actors as workers.

MainObject: Schedules tasks to the workers and receives their results
Worker:
 - init: instantiate a benchmark function
 - get_objective_function(config) -> result
"""

import ray
from time import sleep
from loguru import logger

from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.ray_scheduler import Scheduler
from BenchmarkTools.core.ray_worker import BenchmarkWorker


if __name__ == '__main__':
    benchmark_parameters = dict(scenario='rbv2_xgboost', instance='28', multi_thread=False, container_tag='0.0.2')
    workers = [BenchmarkWorker.options(name=f'W{i}').remote(benchmark_parameters=benchmark_parameters, worker_id=i) for i in range(1)]
    cs = ray.get(workers[0].get_configuration_space.remote())
    configs = cs.sample_configuration(100)

    scheduler = Scheduler(workers)
    scheduler.run()

    sleep(1)

    scheduler.add_jobs([Job(job_id=i, configuration=c) for i, c in enumerate(configs)])

    finished_jobs = []
    while scheduler.get_num_finished_jobs() + scheduler.get_num_running_jobs() + scheduler.get_num_pending_jobs() > 0:
        sleep(1)
        finished_jobs.extend(scheduler.get_finished_jobs())
        logger.info(f'Finished Jobs: {len(finished_jobs)}')

    logger.info(f'Finished Jobs: {len(finished_jobs)}')
    scheduler.stop_workers()
