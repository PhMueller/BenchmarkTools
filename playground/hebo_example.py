"""
Idea: Use ray actors as workers.

MainObject: Schedules tasks to the workers and receives their results
Worker:
 - init: instantiate a benchmark function
 - get_objective_function(config) -> result
"""

from time import time

import numpy as np
import pandas as pd
import ray
from hebo.optimizers.hebo import HEBO
from loguru import logger

from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.ray_scheduler import Scheduler
from BenchmarkTools.core.ray_worker import BenchmarkWorker
from BenchmarkToolsOptimizers.optimizers.hebo.configspace_tools import \
    hebo_config_to_configspace_config, configspace_cs_to_hebo_cs

MAX_VALUE = 2**32 - 1


if __name__ == '__main__':

    ray.init(dashboard_host='localhost', dashboard_port=8265)
    benchmark_parameters = dict(scenario='rbv2_xgboost', instance='28', multi_thread=False, container_tag='0.0.2')

    num_workers = 2
    workers = [
        BenchmarkWorker.remote(benchmark_parameters=benchmark_parameters, worker_id=i)
        for i in range(num_workers)
    ]

    # Give the workers some time to start
    wait_until_alive_futures = [w.is_alive.remote() for w in workers]
    job_ref, remaining_refs = ray.wait(wait_until_alive_futures, num_returns=num_workers, timeout=None)

    # Init the scheduler and start the submit-jobs- and fetching-results-threads
    scheduler = Scheduler(workers)
    scheduler.run()

    configspace_cs = ray.get(workers[0].get_configuration_space.remote())

    # ############################## OPTIMIZER.INIT() ##################################################################
    # Configure the HEBO optimizer
    hebo_cs = configspace_cs_to_hebo_cs(configspace_cs=configspace_cs)
    opt = HEBO(hebo_cs)

    # Mapping from job_id to (hebo config; configspace config)
    job_dict = {}
    # ############################## OPTIMIZER.INIT() ##################################################################

    config_id = 0
    finished_jobs = []
    sleep_interval_in_s = 0.001
    print_every_k_sec = 5  # make print statements every k seconds

    budget_exhausted = False

    while config_id < 100:

        if budget_exhausted: break

        scheduler.wait_until_ready_for_new_configs(sleep_interval_in_s=sleep_interval_in_s, log_every_k_seconds=5)

        # ############################## OPTIMIZER.ASK() ###############################################################
        jobs = []
        t = time()
        logger.info('Start Suggest a new configuration')
        rec = opt.suggest(n_suggestions=4)
        logger.info(f'Done Suggest a new configuration, took {time() - t:4f}s')

        # Cast hebo configs to jobs that are schedulable
        for hebo_config in rec.iterrows():
            _, hebo_config = hebo_config
            configspace_config = hebo_config_to_configspace_config(hebo_config.to_dict(), configspace_cs)
            job = Job(job_id=config_id, configuration=configspace_config)
            jobs.append(job)
            job_dict[config_id] = {'hebo_config': hebo_config, 'configspace_config': configspace_config}
            config_id += 1
        # ############################## OPTIMIZER.ASK() ###############################################################

        scheduler.add_jobs(jobs)

        logger.info('Query finished jobs')
        new_finished_jobs = scheduler.get_finished_jobs()

        if len(new_finished_jobs) == 0:
            continue

        logger.info(f'--> {len(finished_jobs)+len(new_finished_jobs):4d} Jobs have been finished. +{len(new_finished_jobs):3d}')
        finished_jobs.extend(new_finished_jobs)

        # ############################## OPTIMIZER.TELL() ##############################################################
        # pass the jobs to the scheduler
        results = np.array([j.result_dict['function_value']['logloss'] for j in new_finished_jobs]).reshape((-1, 1))
        configs = [job_dict[j.job_id]['hebo_config'] for j in new_finished_jobs]
        configs = pd.DataFrame(configs)
        opt.observe(configs, results)
        # ############################## OPTIMIZER.TELL() ##############################################################

    print('test')
    scheduler.stop_workers()
