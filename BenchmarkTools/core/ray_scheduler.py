from threading import Thread, Lock
from time import sleep
from typing import List, Dict, Union

import ray

from BenchmarkTools import logger
from BenchmarkTools.core.ray_job import Job
from BenchmarkTools.core.ray_worker import Worker


class Scheduler(object):
    """ A job scheduler with a fix number of persistent workers"""
    def __init__(self, workers: List):

        # Collection of workers that process a given configuration.
        self.workers: Dict[str, Worker] = {}
        for worker in workers:
            self.add_worker(worker)

        # Store the names of idle workers
        self.free_workers: List[str] = list(self.workers.keys())

        # This queue does contain all jobs that are not scheduled yet
        self.job_queue: List[Job] = []

        # This queue stores the references to the currently running jobs
        self.running_jobs: List[str] = []

        # this queue stores the returned results from the workers
        self.finished_jobs_queue: List[Job] = []

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
        if worker_name in self.workers:
            logger.info(f'Worker {worker_name} already in mapping. Skip.')
            return

        self.workers[worker_name] = worker
        logger.info(f'Worker {worker_name} added to scheduler.')

    def add_jobs(self, jobs: Union[List[Job], Job]):

        if isinstance(jobs, Job):
            jobs = [jobs]

        with self.job_queue_lock:
            self.job_queue.extend(jobs)

    def get_num_finished_jobs(self) -> int:
        with self.finished_jobs_queue_lock:
            return len(self.finished_jobs_queue)

    def get_num_pending_jobs(self) -> int:
        with self.job_queue_lock:
            return len(self.job_queue)

    def get_num_running_jobs(self) -> int:
        with self.running_jobs_lock:
            return len(self.running_jobs)

    def get_finished_jobs(self) -> List[Job]:
        with self.finished_jobs_queue_lock:
            results = self.finished_jobs_queue.copy()
            self.finished_jobs_queue = []
            return results

    def run(self):
        """ Start submitting and fetching potential jobs in the background. """
        self.submit_is_running = True
        self.fetch_is_running = True
        self.submit_thread.start()
        self.fetch_thread.start()

    def stop_background_threads(self):
        """ Stop the background processes of submitting and fetching. """
        self.submit_is_running = False
        self.fetch_is_running = False
        self.submit_thread.join()
        self.fetch_thread.join()

    def loop_submit_job(self):
        # If there are free workers: submit a new job
        logger.info(f'Free Workers: {len(self.free_workers):2d} - Jobs to assign: {len(self.job_queue):3d}')

        while self.submit_is_running:
            if len(self.free_workers) != 0 and len(self.job_queue) != 0:
                logger.info(f'Free Workers: {len(self.free_workers):2d} - Jobs to assign: {len(self.job_queue):3d}')

                # Get a new configuration from the job queue and assign the job to a free worker
                with self.job_queue_lock:
                    job = self.job_queue.pop()

                with self.free_worker_lock:
                    free_worker_id = self.free_workers.pop()
                    free_worker = self.workers[free_worker_id]
                    job_ref = free_worker._process_job.remote(job)

                with self.running_jobs_lock:
                    self.running_jobs.append(job_ref)

            else:
                sleep(0.0001)

    def loop_fetch_results(self):
        """ This is the function for fetching results and add them to a result queue. """
        while self.fetch_is_running:
            if len(self.running_jobs) != 0:

                # Might break here otherwise: New job is added but we override here the remaining jobs!
                with self.running_jobs_lock:
                    copy_running_jobs = self.running_jobs.copy()

                job_ref, remaining_refs = ray.wait(copy_running_jobs, num_returns=1, timeout=None)
                job_ref = job_ref[0]
                finished_job: Job = ray.get(job_ref)

                # Look for the worker id that has executed the job
                worker_id = job_ref.task_id().actor_id().hex()
                logger.info(f'Fetch Result: Worker {worker_id} - Result {finished_job.result_dict}')

                with self.finished_jobs_queue_lock:
                    self.finished_jobs_queue.append(finished_job)

                with self.free_worker_lock:
                    self.free_workers.append(worker_id)

                with self.running_jobs_lock:
                    # Better: Remove the finished job from the running jobs - queue
                    self.running_jobs = [j for j in self.running_jobs if j.hex() != job_ref.hex()]

    def stop_workers(self):
        for worker in self.workers.values():
            try:
                ray.get(worker.stop.remote())
                ray.kill(worker)
            except Exception as e:
                pass

    def __del__(self):
        self.stop_background_threads()
        self.stop_workers()
