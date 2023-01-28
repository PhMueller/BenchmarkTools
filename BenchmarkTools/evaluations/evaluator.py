from pathlib import Path
from typing import Union

from BenchmarkTools.evaluations.data_container import (
    DataContainerFromSQLite,
    load_data_containers_from_directory,
    combine_multiple_data_container
)
from BenchmarkTools.evaluations.empirical_pareto_front import get_empirical_pareto_front


def prepare_multi_objective_evaluation(experiment_result_dir: Union[Path, str], output_dir: Union[Path, str]):
    """
    This function performs following steps:
        1) Empirical Pareto Front:
            - Read in all runhistories of an experiment
            - Creates the empirical pareto front over the entire experiment
            - Compute statistics: Best and nadir point of the PF
            - Store the information in yaml format
    """
    output_dir = Path(output_dir)
    experiment_result_dir = Path(experiment_result_dir)
    experiment_name = experiment_result_dir.name

    data_containers = load_data_containers_from_directory(experiment_result_dir)
    combined_study = combine_multiple_data_container(data_containers)

    emp_pf_stats = get_empirical_pareto_front(
        experiment_name=experiment_name,
        experiment_result_dir=experiment_result_dir,
        combined_study=combined_study,
        output_dir=output_dir,
    )

    # Compute the normalized hypervolume over time
    anytime_hv_dir = output_dir / 'norm_hv' / experiment_name
    anytime_hv_dir.mkdir(exist_ok=True, parents=True)
    data_container = data_containers[0]

    # TODO Extract code from MO HPOBench.

    # for data_container in data_containers:

    print('done')


if __name__ == '__main__':

    prepare_multi_objective_evaluation(
        experiment_result_dir='C:\\Users\\Philipp\\PycharmProjects\\BenchmarkTools\\Results\\BraninCurrin',
        output_dir='C:\\Users\\Philipp\\PycharmProjects\\BenchmarkTools\\Results\\Analysis',
    )

    # storage_path = Path('/home/pm/Dokumente/Code/BenchmarkTools/Results/yahpo/rs/0/run_storage.db')
    # data_container = DataContainer(storage_path=storage_path)
    # print(data_container.study.user_attrs)
    # print('test')
