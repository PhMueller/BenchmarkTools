from pathlib import Path
from datetime import datetime
from oslo_concurrency import lockutils
import pandas as pd
from BenchmarkTools import logger
import pickle
import shutil
from pathlib import Path
from typing import Any, Union, Dict
from pprint import pformat
import traceback

import numpy as np
from importlib import import_module

from tqdm import tqdm
from BenchmarkTools.core.exceptions import AlreadyFinishedException, BudgetExhaustedException


def wrapper_track_run_stats(func_to_wrap):
    def track_run_stats(experiment_name, experiment_settings, optimizer_name, optimizer_settings,
                        run_id, output_path, debug, *args, **kwargs):

        run_state_dir = Path(output_path) / '_run_states'
        lock_dir = run_state_dir / 'lock_collect_runs'
        lock_dir.mkdir(exist_ok=True, parents=True)
        state_file = run_state_dir / 'run_states.csv'
        start_time = datetime.now()

        run_dir = Path(output_path) / experiment_name / optimizer_name / str(run_id)

        @lockutils.synchronized('not_thread_process_safe', external=True, lock_path=str(lock_dir), delay=0.01)
        def update_state_file(entry, state_file, col_names, check_duplicates_on):
            new_df = pd.DataFrame([entry])

            # When a run has finished check if it was broken before and remove that broken one from the run stats
            if state_file.exists():
                old_df = pd.read_csv(state_file, delimiter=';', header=None, names=col_names)
                new_df = pd.concat([old_df, new_df], axis=0).reset_index(drop=True)
            new_df = new_df.drop_duplicates(subset=check_duplicates_on, keep='last')
            new_df.to_csv(state_file, sep=';', header=None, index=False)

        skip_writing = False
        crash = False
        exception = None
        tb = None
        try:
            logger.debug('Start evaluation')
            func_to_wrap(experiment_name, experiment_settings, optimizer_name, optimizer_settings,
                        run_id, output_path, debug, *args, **kwargs)
            logger.info('Finished without exception')
        except Exception as e:
            logger.info('Exception caught: -> Going to save the run state')
            tb = traceback.format_exc()
            exception = e
            if not isinstance(e, AlreadyFinishedException):
                crash = True

        col_names = ['time_str', 'wallclock_time_in_s', 'result_dir', 'benchmark', 'optimizer', 'run_id', 'status', 'run_dir', 'exception']
        check_duplicates_on = [                         'result_dir', 'benchmark', 'optimizer', 'run_id']
        entry = {
            'time_str': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4],
            'wallclock_time_in_s': (datetime.now() - start_time).seconds,
            'result_dir': output_path,
            'benchmark': experiment_name,
            'optimizer': optimizer_name,
            'run_id': int(run_id),
            'status': 'CRASH' if crash else 'SUCCESS',
            'run_dir': str(run_dir),
            'exception': str(tb) if crash else 'no exception',
        }
        logger.info(f'Wrapper: Finished: {pformat(entry)}')
        logger.info(f'Write to status file: {state_file}')
        update_state_file(entry, state_file, col_names, check_duplicates_on)

        if exception is not None:
            logger.info('During the execution of the optimization following exception occured:')
            raise exception

    return track_run_stats
