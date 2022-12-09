import optuna
from pathlib import Path
from typing import Optional, Union

from BenchmarkTools import logger


class DataContainer:
    def __init__(self, storage_path: Union[str, Path], study_name: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path = f"sqlite:///{self.storage_path.resolve()}"

        if study_name is None:
            study_name = self.__try_extract_study_name()
        self.study_name = study_name
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





if __name__ == '__main__':
    storage_path = Path('/home/pm/Dokumente/Code/BenchmarkTools/Results/yahpo/rs/0/run_storage.db')
    data_container = DataContainer(storage_path=storage_path)
    print(data_container.study.user_attrs)
    print('test')