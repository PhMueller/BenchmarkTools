from threading import Thread, Lock
from time import sleep, time
from typing import List, Dict, Union, Deque
from collections import deque

import ray

from BenchmarkTools import logger
from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.ray_worker import Worker


class Scheduler(object):
    """ A job scheduler with a fix number of persistent workers"""
    def __init__(self, workers: List):

        # Collection of workers that process a given configuration.
        self.workers_dict: Dict[str, Worker] = {}
        for worker in workers:
            self.add_worker(worker)

        # Store the names of idle workers
        self.free_workers_queue: Deque[str] = deque(self.workers_dict.keys())

        # This queue does contain all jobs that are not scheduled yet
        self.job_queue: Deque[Job] = deque()

        # This queue stores the references to the currently running jobs
        self.running_jobs: List[str] = []

        # this queue stores the returned results from the workers
        self.finished_jobs: List[Job] = []

        # Submitting and Fetching is organized in threads. Thus, we need some flags to be able to cancel these processes
        # as well as some locks to make it concurrently working
        self.submit_is_running = False
        self.fetch_is_running = False
        self.submit_thread = Thread(target=self.loop_submit_job)
        self.fetch_thread = Thread(target=self.loop_fetch_results)

        self.job_queue_lock = Lock()
        self.free_worker_lock = Lock()
        self.running_jobs_lock = Lock()
        self.finished_jobs_queue_lock = Lock()

    def add_worker(self, worker: Worker):
        """ Add a new worker to the scheduler. """
        worker_name = worker._actor_id.hex()
        if worker_name in self.workers_dict:
            logger.info(f'Worker {worker_name} already in mapping. Skip.')
            return

        self.workers_dict[worker_name] = worker
        logger.info(f'Worker {worker_name} added to scheduler.')

    def add_jobs(self, jobs: Union[List[Job], Deque[Job], Job]):

        if isinstance(jobs, Job):
            jobs = [jobs]

        with self.job_queue_lock:
            self.job_queue.extend(jobs)

    def get_num_finished_jobs(self) -> int:
        with self.finished_jobs_queue_lock:
            return len(self.finished_jobs)

    def get_num_pending_jobs(self) -> int:
        with self.job_queue_lock:
            return len(self.job_queue)

    def get_num_running_jobs(self) -> int:
        with self.running_jobs_lock:
            return len(self.running_jobs)

    def get_num_free_workers(self) -> int:
        with self.free_worker_lock:
            return len(self.free_workers_queue)

    def get_finished_jobs(self) -> List[Job]:
        with self.finished_jobs_queue_lock:
            finished_jobs = self.finished_jobs.copy()
            self.finished_jobs = []
            return finished_jobs

    def run(self):
        """ Start submitting and fetching potential jobs in the background. """
        self.submit_is_running = True
        self.fetch_is_running = True
        self.submit_thread.start()
        self.fetch_thread.start()

    def loop_submit_job(self):
        # If there are free workers: submit a new job
        logger.info('Start submit-loop')

        while self.submit_is_running:
            if len(self.free_workers_queue) != 0 and len(self.job_queue) != 0:
                logger.info(f'Free Workers: {len(self.free_workers_queue):2d} - Jobs to assign: {len(self.job_queue):3d}')

                # Get a new configuration from the job queue and assign the job to a free worker
                with self.job_queue_lock:
                    job = self.job_queue.popleft()

                with self.free_worker_lock:
                    free_worker_id = self.free_workers_queue.popleft()
                    free_worker = self.workers_dict[free_worker_id]
                    job_ref = free_worker._process_job.remote(job)  # reference to future result object.

                logger.debug(f'Assign job {job_ref.hex()[:6]} to worker {free_worker_id[:6]}')

                with self.running_jobs_lock:
                    self.running_jobs.append(job_ref)

            else:
                sleep(0.0001)
        logger.info('End submit-loop')

    def loop_fetch_results(self):
        """ This is the function for fetching results and add them to a result queue. """
        logger.info('Start result-fetching-loop')

        while self.fetch_is_running:
            if len(self.running_jobs) != 0:

                # Might break here otherwise: New job is added, but we override here the remaining jobs!
                with self.running_jobs_lock:
                    copy_running_jobs = self.running_jobs.copy()

                # Either no or at max 1 finished job is returned here...
                # wait_interval_in_s = 1
                # job_ref, remaining_refs = ray.wait(copy_running_jobs, num_returns=1, timeout=wait_interval_in_s)
                job_ref, remaining_refs = ray.wait(copy_running_jobs, num_returns=1, timeout=None)

                # If no result is available, skip the remaining loop
                # if len(job_ref) == 0:
                #     continue

                job_ref = job_ref[0]
                finished_job: Job = ray.get(job_ref)

                # Look for the worker id that has executed the job
                worker_id = job_ref.task_id().actor_id().hex()
                logger.debug(f'Get job {job_ref.hex()[:6]} from {worker_id[:6]}')
                logger.debug(f'Received result {finished_job.result_dict}')

                with self.finished_jobs_queue_lock:
                    self.finished_jobs.append(finished_job)

                with self.free_worker_lock:
                    logger.debug(f'Add worker {worker_id[:6]} to free worker queue')
                    self.free_workers_queue.append(worker_id)

                with self.running_jobs_lock:
                    # Better: Remove the finished job from the running jobs - queue
                    self.running_jobs = [j for j in self.running_jobs if j.hex() != job_ref.hex()]
        logger.info('End result-fetching-loop')

    def stop_background_threads(self):
        """ Stop the background processes of submitting and fetching. """
        logger.info('Stop background threads')
        self.submit_is_running = False
        self.fetch_is_running = False
        self.submit_thread.join()
        self.fetch_thread.join()

    def stop_workers(self):
        logger.info('Call stop function of workers')
        for worker in self.workers_dict.values():
            try:
                ray.get(worker.stop.remote())
                ray.kill(worker)
            except Exception as e:
                pass

    def shutdown(self):
        self.stop_background_threads()
        self.stop_workers()

    def __del__(self):
        self.shutdown()


def is_ready_for_new_configuration(scheduler: Scheduler, show_log_msg: bool = False) -> bool:
    num_pending_jobs = scheduler.get_num_pending_jobs()
    num_running_jobs = scheduler.get_num_running_jobs()
    num_free_workers = scheduler.get_num_free_workers()

    if show_log_msg:
        logger.info(f'Currently Free Workers: {num_free_workers:3d}')
        logger.info(f'Currently Pending jobs: {num_pending_jobs:3d}')
        logger.info(f'Currently Running jobs: {num_running_jobs:3d}')

    # Check if some workers have not assigned work yet
    free_workers_available_1 = num_free_workers > 0

    # It might be the case that a worker is not assigned yet but there are enough pending jobs to be assigned soon
    free_workers_available_2 = (num_running_jobs + num_pending_jobs) < len(scheduler.workers_dict)

    # We can schedule a new configuration if both conditions are fulfilled.
    return free_workers_available_1 and free_workers_available_2


def wait_until_ready_for_new_configs(scheduler: Scheduler, sleep_interval: float = 0.001, log_every_k_seconds: int = 5):
    last_print = time()
    show_log_msg = True

    # If there are no free workers or no job to potentially schedule to an empty worker, wait.
    while not is_ready_for_new_configuration(scheduler, show_log_msg=show_log_msg):

        # Only print every `log_every_k_seconds`
        if show_log_msg:
            last_print = time()  # update
            logger.info(f'No Free Worker Available. Sleep for {sleep_interval}s.')

        sleep(sleep_interval)

        show_log_msg = time() - last_print > log_every_k_seconds
