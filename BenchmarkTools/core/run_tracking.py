import traceback
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import List

import pandas as pd
from oslo_concurrency import lockutils

from BenchmarkTools import logger
from BenchmarkTools.core.exceptions import AlreadyFinishedException


def wrapper_track_run_stats(func_to_wrap):

    def track_run_stats(benchmark_name, benchmark_settings, optimizer_name, optimizer_settings,
                        run_id, output_path, debug, *args, **kwargs):
        """
        This wrapper tracks the experimental progress and makes it easy to observe what runs have been executed or crashed.

        Args:
            benchmark_name: str
            benchmark_settings: Dict
            optimizer_name: str
            optimizer_settings: Dict
            run_id: int
            output_path: Path
            debug: bool
            *args: List
            **kwargs: Dict
        """
        run_state_dir = Path(output_path) / '_run_states'
        lock_dir = run_state_dir / 'lock_run_states'
        lock_dir.mkdir(exist_ok=True, parents=True)
        state_file = run_state_dir / 'run_states.csv'
        start_time = datetime.now()

        run_dir = Path(output_path) / benchmark_name / optimizer_name / str(run_id)

        @lockutils.synchronized('not_thread_process_safe', external=True, lock_path=str(lock_dir), delay=0.001)
        def update_state_file(entry: Dict, state_file: Path, col_names: List[str], check_duplicates_on: List[str]):
            """Helperfunction: Write the tracking stats to a csv file. """
            new_df = pd.DataFrame([entry])

            # When a run has finished check if it was broken before and remove that broken one from the run stats
            if state_file.exists():
                old_df = pd.read_csv(state_file, delimiter=';', header=None, names=col_names)
                new_df = pd.concat([old_df, new_df], axis=0).reset_index(drop=True)
            new_df = new_df.drop_duplicates(subset=check_duplicates_on, keep='last')
            new_df.to_csv(state_file, sep=';', header=None, index=False)

        crash = False
        exception = None
        tb = None
        try:
            # Execute the optimization function:
            logger.debug('Start evaluation')
            func_to_wrap(
                benchmark_name, benchmark_settings, optimizer_name, optimizer_settings,
                run_id, output_path, debug, *args, **kwargs
            )
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
            'benchmark': benchmark_name,
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
