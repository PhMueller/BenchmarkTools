from time import time
from typing import Dict


class Job(object):
    def __init__(self, job_id: int, configuration: Dict, fidelity: Dict = None):
        self.job_id = job_id
        self.configuration = configuration
        self.fidelity = fidelity if fidelity is not None else {}
        self.result_dict = None
        self.start_time = None
        self.finish_time = None

    def start_job(self):
        self.start_time = time()

    def register_result(self, result_dict: Dict):
        self.result_dict = result_dict
        self.finish_time = time()
