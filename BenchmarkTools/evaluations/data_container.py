from pathlib import Path
from typing import Union, Optional, List

import optuna
import optuna.study
from loguru import logger
from optuna.trial import TrialState
from tqdm import tqdm

from BenchmarkTools.utils.constants import BenchmarkToolsConstants


class DataContainer:
    def __init__(self, storage_path: Union[str, Path], study_name: Optional[str] = None):
        self.directory = Path(storage_path).parent
        self.storage_path = f"sqlite:///{Path(storage_path).resolve()}"
        self.optimizer = storage_path.parent.parent.name
        self.run_id = storage_path.parent.name
        self.study_name = study_name if study_name is not None else self.__try_extract_study_name()
        self.study = self._try_load_study()

    def _try_load_study(self):
        logger.info(f'Try to connect to study {self.study_name} at {self.storage_path}')
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


def load_data_containers_from_directory(experiment_result_dir: Union[str, Path]) -> List[DataContainer]:
    experiment_result_dir = Path(experiment_result_dir)

    # Load the run histories and combine them into a single study
    db_files = list(experiment_result_dir.rglob(BenchmarkToolsConstants.DATABASE_NAME.value))
    data_containers = []
    for db_file in tqdm(db_files, desc='Load run histories'):
        data_containers.append(DataContainer(storage_path=db_file))
    return data_containers


def combine_multiple_data_container(data_containers: List[DataContainer]) -> optuna.study.Study:
    """
    This function does only work if the data container belong to the same benchmark (experiment)

    Args:
        data_containers: DataContainer
            Stores the runhistory and additional information

    Returns:
        Combined Runhistory
    """
    trials = []
    user_attrs = []
    benchmark_name = None
    directions = None
    for data_container in data_containers:
        if benchmark_name is None:
            benchmark_name = data_container.study.user_attrs['benchmark_name']

        if directions is None:
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
    for i, _user_attrs in enumerate(user_attrs):
        combined_study.set_user_attr(key=str(i), value=_user_attrs)
    return combined_study
