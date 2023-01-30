import abc
from abc import ABC
from time import time
from typing import Dict, Optional
from BenchmarkTools import logger
from BenchmarkTools.core.exceptions import BudgetExhaustedException
from pathlib import Path
from oslo_concurrency import lockutils
import json


class BookKeeper(abc.ABC):
    """
    AbstractClass: This class takes care of managing the optimization limits.
    """
    def __init__(self, benchmark_settings, lock_dir: Path, **kwargs):
        self.benchmark_settings = benchmark_settings
        self.lock_dir = Path(lock_dir)
        self.lock_name = 'resource_lock'

        self.lock_dir.mkdir(exist_ok=True, parents=True)

        self.tae_limit = self.benchmark_settings['optimization_parameters']['tae_limit']
        self.wallclock_limit_in_s = self.benchmark_settings['optimization_parameters']['wallclock_limit_in_s']
        self.estimated_cost_limit_in_s = self.benchmark_settings['optimization_parameters']['estimated_cost_limit_in_s']
        self.is_surrogate = self.benchmark_settings['optimization_parameters']['is_surrogate']

    def get_resource_lock(self):
        return lockutils.lock(
            name=self.lock_name, external=True, do_log=False, lock_path=str(self.lock_dir), delay=0.001
        )

    def get_used_resources(self) -> Dict:
        resource_lock = self.get_resource_lock()
        with resource_lock:
            resources = self._get_used_resources()
        return resources

    @abc.abstractmethod
    def _get_used_resources(self) -> Dict:
        raise NotImplementedError()

    def set_used_resources(
            self,
            initial_time: Optional[float] = None,
            num_tae_calls: Optional[float] = None,
            sum_surrogate_cost: Optional[float] = None,
            sum_wallclock_time: Optional[float] = None,
            sum_total_costs: Optional[float] = None
    ):
        resource_lock = self.get_resource_lock()
        with resource_lock:
            self._set_used_resources(
                initial_time, num_tae_calls, sum_surrogate_cost, sum_wallclock_time, sum_total_costs
            )

    @abc.abstractmethod
    def _set_used_resources(
            self,
            initial_time: Optional[float] = None,
            num_tae_calls: Optional[float] = None,
            sum_surrogate_cost: Optional[float] = None,
            sum_wallclock_time: Optional[float] = None,
            sum_total_costs: Optional[float] = None
    ):
        NotImplementedError()

    def increase_used_resources(
            self,
            delta_tae_calls: Optional[float] = 0,
            delta_surrogate_cost: Optional[float] = 0,
            delta_wallclock_time: Optional[float] = 0,
            delta_total_costs: Optional[float] = 0
    ):
        resource_lock = self.get_resource_lock()
        with resource_lock:
            self._increase_used_resources(
                delta_tae_calls, delta_surrogate_cost, delta_wallclock_time, delta_total_costs
            )

    def _increase_used_resources(
            self,
            delta_tae_calls: Optional[float] = 0,
            delta_surrogate_cost: Optional[float] = 0,
            delta_wallclock_time: Optional[float] = 0,
            delta_total_costs: Optional[float] = 0
    ):
        used_resources = self._get_used_resources()
        used_resources['num_tae_calls'] += delta_tae_calls
        used_resources['sum_surrogate_cost'] += delta_surrogate_cost
        used_resources['sum_wallclock_time'] += delta_wallclock_time
        used_resources['sum_total_costs'] += delta_total_costs
        self._set_used_resources(**used_resources)

    def start_timer(self):
        resource_lock = self.get_resource_lock()
        with resource_lock:
            self._start_timer()

    def _start_timer(self):
        self._set_used_resources(
            initial_time=time(),  num_tae_calls=-1, sum_surrogate_cost=0, sum_wallclock_time=0, sum_total_costs=0,
        )

    def check_optimization_limits(self):
        resource_lock = self.get_resource_lock()
        with resource_lock:
            self._check_optimization_limits()

    def _check_optimization_limits(self):
        used_resources = self._get_used_resources()

        # --------------------------- UPDATE TIMER ---------------------------------------------------------------------
        # TODO: Write a function that writes the resources to file without loading them. This makes a speed up by
        #       reducing unnecessary file access calls.
        sum_wallclock_time = time() - used_resources['initial_time']
        sum_total_costs = sum_wallclock_time + used_resources['sum_surrogate_cost']
        self._set_used_resources(
            sum_wallclock_time=sum_wallclock_time,
            sum_total_costs=sum_total_costs,
        )
        # --------------------------- UPDATE TIMER ---------------------------------------------------------------------

        # --------------------------- CHECK RUN LIMITS -----------------------------------------------------------------
        # Check that the wallclock time limit is not reached
        if sum_wallclock_time >= self.wallclock_limit_in_s:
            logger.warning(f'Wallclock Time Limit Reached: {self.wallclock_limit_in_s}')
            raise BudgetExhaustedException(f'Wallclock Time Limit Reached: {self.wallclock_limit_in_s}')

        # If we evaluate a surrogate model, then we also care about the predicted costs
        if self.is_surrogate and (sum_total_costs >= self.estimated_cost_limit_in_s):
            logger.warning(f'Surrogate Time Limit Reached: {self.estimated_cost_limit_in_s}')
            raise BudgetExhaustedException(f'Surrogate Time Limit Reached: {self.estimated_cost_limit_in_s}')

        # By default, the tae limit is not active. If it is set to -1, 1000 * #hps is used.
        if self.tae_limit is not None and used_resources['num_tae_calls'] > self.tae_limit:
            logger.warning(f'Total Number of Target Executions Limit is reached. {used_resources["num_tae_calls"]}')
            raise BudgetExhaustedException(
                f'Total Number of Target Executions Limit is reached. {used_resources["num_tae_calls"]}'
            )
        # --------------------------- CHECK RUN LIMITS -----------------------------------------------------------------

    def increase_num_tae_calls_and_get_num_tae_calls(self, delta_tae_calls=1):
        # TODO: same here. reduce file access calls.
        with self.get_resource_lock():
            self._increase_used_resources(delta_tae_calls=delta_tae_calls)
            num_tae_calls = self._get_used_resources()['num_tae_calls']
        return num_tae_calls

    def log_currently_used_resources(self):
        """
        Logging function: Write the currently used resources to the logger.
        TODO: Add callbacks to the class and execute the callbacks here.
        """
        used_resources = self.get_used_resources()
        remaining_time = self.wallclock_limit_in_s - used_resources['sum_wallclock_time']
        logger.info(f'WallClockTime left: {remaining_time:10.4f}s ({remaining_time / 3600:.4f}h)')
        if self.is_surrogate:
            remaining_time = self.estimated_cost_limit_in_s - used_resources['sum_total_costs']
            logger.info(f'EstimatedTime left: {remaining_time:10.4f}s ({remaining_time / 3600:.4f}h)')
        if self.tae_limit is not None:
            logger.info(f'Number of TAE: {used_resources["num_tae_calls"]:10d}|{self.tae_limit}')
        else:
            logger.info(f'Number of TAE: {used_resources["num_tae_calls"]:10d}| INF')


class MemoryBookkeeper(BookKeeper):

    def __init__(self, benchmark_settings: Dict,lock_dir: Path):
        """
        This bookkeeper lives in the memory.

        Args:
            benchmark_settings: Dict
            lock_dir: Path
                Directory to store the file lock
        """

        super(MemoryBookkeeper, self).__init__(benchmark_settings=benchmark_settings, lock_dir=lock_dir)

        # Store the limit information in this class.
        self.initial_time = time()
        self.num_tae_calls = -1
        self.sum_surrogate_cost = 0
        self.sum_wallclock_time = 0
        self.sum_total_costs = 0

    def _set_used_resources(
            self,
            initial_time: Optional[float] = None,
            num_tae_calls: Optional[float] = None,
            sum_surrogate_cost: Optional[float] = None,
            sum_wallclock_time: Optional[float] = None,
            sum_total_costs: Optional[float] = None
    ):
        self.initial_time = initial_time or self.initial_time
        self.num_tae_calls = num_tae_calls or self.num_tae_calls
        self.sum_surrogate_cost = sum_surrogate_cost or self.sum_surrogate_cost
        self.sum_wallclock_time = sum_wallclock_time or self.sum_wallclock_time
        self.sum_total_costs = sum_total_costs or self.sum_total_costs

    def _get_used_resources(self) -> Dict:
        return {
            'initial_time': self.initial_time,
            'num_tae_calls': self.num_tae_calls,
            'sum_surrogate_cost': self.sum_surrogate_cost,
            'sum_wallclock_time': self.sum_wallclock_time,
            'sum_total_costs': self.sum_total_costs,
        }


class FileBookKeeper(BookKeeper):

    def __init__(self, benchmark_settings: Dict, lock_dir: Path, resource_file_dir: Path):
        """
        This object stores the optimization limits per run in a json - file.

        Args:
            benchmark_settings:
            lock_dir:
            resource_file_dir:
        """
        super(FileBookKeeper, self).__init__(benchmark_settings=benchmark_settings, lock_dir=lock_dir)

        self.resource_file_dir = Path(resource_file_dir)
        self.resources_file = self.resource_file_dir / 'resources.json'

    def _get_used_resources(self) -> Dict:
        if not self.resources_file.exists():
            return {
                'initial_time': time(),
                'num_tae_calls': -1,
                'sum_surrogate_cost': 0,
                'sum_wallclock_time': 0,
                'sum_total_costs': 0,
            }

        with self.resources_file.open('r') as fh:
            used_resources = json.load(fh)
        return used_resources

    def _set_used_resources(
            self,
            initial_time: Optional[float] = None,
            num_tae_calls: Optional[float] = None,
            sum_surrogate_cost: Optional[float] = None,
            sum_wallclock_time: Optional[float] = None,
            sum_total_costs: Optional[float] = None
    ):
        old_data = self._get_used_resources()

        data_obj = {
            'initial_time': initial_time if initial_time is not None else old_data['initial_time'],
            'num_tae_calls': num_tae_calls if num_tae_calls is not None else old_data['num_tae_calls'],
            'sum_surrogate_cost': sum_surrogate_cost if sum_surrogate_cost is not None else old_data['sum_surrogate_cost'],
            'sum_wallclock_time': sum_wallclock_time if sum_wallclock_time is not None else old_data['sum_wallclock_time'],
            'sum_total_costs': sum_total_costs if sum_total_costs is not None else old_data['sum_total_costs'],
        }

        with self.resources_file.open('w') as fp:
            json.dump(obj=data_obj, fp=fp, indent=2)
