import time
from pathlib import Path
from typing import Union, Optional, List

import optuna
import optuna.study
from loguru import logger
from optuna.trial import TrialState
from tqdm import tqdm
import multiprocessing as mp

from BenchmarkTools.core.constants import BenchmarkToolsConstants


class DataContainer:
    def __init__(self, study_name: Optional[str] = None):
        self.study_name = study_name  # corresponds to the benchmark name
        self.optimizer = None
        self.run_id = None
        self.study = None


class DataContainerFromStudy(DataContainer):
    def __init__(self, study: optuna.Study, study_name: Optional[str] = None):
        super(DataContainerFromStudy, self).__init__(study_name=study_name)
        self.optimizer = None
        self.run_id = None
        self.study = study


class DataContainerFromSQLite(DataContainer):
    def __init__(self, storage_path: Union[str, Path], study_name: Optional[str] = None):
        super(DataContainerFromSQLite, self).__init__(study_name=study_name)
        self.directory = Path(storage_path).parent
        self.storage_path = f"sqlite:///{Path(storage_path).resolve()}"
        self.optimizer = storage_path.parent.parent.name
        self.run_id = storage_path.parent.name
        self.study_name = study_name if study_name is not None else self.__try_extract_study_name()
        self.study = self._try_load_study()

    def _try_load_study(self):
        logger.debug(f'Try to connect to study {self.study_name} at {self.storage_path}')
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage_path,
        )
        return study

    def __try_extract_study_name(self):
        """Try to retrieve the available study names in the data base. """
        summaries = optuna.get_all_study_summaries(str(self.storage_path))
        if len(summaries) == 1:
            return summaries[0].study_name

        elif len(summaries) == 0:
            raise optuna.exceptions.OptunaError(f'Study (here: {self.storage_path}) is empty.')

        elif len(summaries) > 1:
            available_names = [s.study_name for s in summaries]
            logger.warning('More than 1 study is in the database. Please provide a study name.')
            logger.warning(f'Available study names: {available_names}')
            raise optuna.exceptions.OptunaError('Not unique study.')
        else:
            raise optuna.exceptions.OptunaError(f'Unclear State. {summaries}')


def create_container(db_file: Path):
    """ Helperfunction: Create a data container from a file path. This is used to use it in multiprocessing.Pool. """
    return DataContainerFromSQLite(db_file)


def load_data_containers_from_directory(experiment_result_dir: Union[str, Path]) -> List[DataContainerFromSQLite]:
    """
    Search in a `experiment_result_dir`  recursively all available runhistories.

    Args:
        experiment_result_dir: str, Path
            Dir containing runhistories. Can be the directory of a benchmark or a specific optimizer.
    Returns:

    """
    experiment_result_dir = Path(experiment_result_dir)

    # Load the run histories and combine them into a single study
    db_files = list(experiment_result_dir.rglob(BenchmarkToolsConstants.DATABASE_NAME.value))
    # TODO: Make that step parallel.

    # start_time = time.time()
    # with mp.Pool() as pool:
    #     data_containers = pool.map(create_container, db_files)
    # duration_1 = time.time() - start_time


    # start_time = time.time()
    data_containers = []
    for db_file in tqdm(db_files, desc='Load run histories'):
        data_containers.append(DataContainerFromSQLite(storage_path=db_file))
    # duration_2 = time.time() - start_time

    # print(f'Time MP: {duration_1}')
    # print(f'Time SP: {duration_2}')

    return data_containers


def combine_multiple_data_container(
        data_containers: List[DataContainerFromSQLite],
        same_optimizer: bool = True) -> DataContainerFromStudy:
    """
    This function does only work if the data container belong to the same benchmark (experiment)

    Args:
        data_containers: DataContainer
            Stores the runhistory and additional information
        same_optimizer: bool
    Returns:
        DataContainer holding the combined runhistory
    """
    trials = []
    user_attrs = []
    benchmark_name = None
    directions = None
    for data_container in data_containers:
        if benchmark_name is None:
            benchmark_name = data_container.study.user_attrs['benchmark_name']
            directions = data_container.study.user_attrs['directions']

        # Make sure that every runhistory is from the same experiment
        assert data_container.study.user_attrs['benchmark_name'] == benchmark_name
        assert data_container.study.user_attrs['directions'] == directions

        trials.extend(data_container.study.get_trials(states=[TrialState.COMPLETE]))
        user_attrs.append(data_container.study.user_attrs)

    combined_study: optuna.study.Study = optuna.study.create_study(
        study_name=benchmark_name,
        directions=directions,
    )
    combined_study.add_trials(trials=trials)

    optimizer = set()
    run_ids = []

    for i, _user_attrs in enumerate(user_attrs):
        optimizer.add(_user_attrs.get('optimizer_name', 'NotDefined'))
        run_ids.append(_user_attrs.get('run_id', 'NotDefined'))
        if same_optimizer:
            for k, v in _user_attrs.items():
                combined_study.set_user_attr(k, v)
        else:
            combined_study.set_user_attr(key=str(i), value=_user_attrs)

    data_container = DataContainerFromStudy(study=combined_study, study_name=benchmark_name)

    if same_optimizer:
        assert len(optimizer) == 1
        data_container.optimizer = optimizer.pop()
    else:
        data_container.optimizer = list(optimizer)

    run_ids.sort()
    data_container.run_id = run_ids
    return data_container
