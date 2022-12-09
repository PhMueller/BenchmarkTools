"""
This script extracts the empirical pareto front from a benchmark.
It scraps all results in a certain directory and computes the pareto front across the combination of the evaluated
configurations.
"""

from typing import Union, List
from pathlib import Path

import optuna.visualization

from BenchmarkTools.utils.constants import BenchmarkToolsConstants
from BenchmarkTools.evaluations.evaluator import DataContainer
from optuna.trial import TrialState

from optuna._hypervolume import WFG
from optuna.study._multi_objective import _get_pareto_front_trials, _normalize_value
import numpy as np


def _combine_multiple_data_container(data_containers: List[DataContainer]) -> optuna.study.Study:
    trials = []
    user_attrs = []
    for data_container in data_containers:
        trials.extend(data_container.study.get_trials(states=[TrialState.COMPLETE]))
        user_attrs.append(data_container.study.user_attrs)

    combined_study: optuna.study.Study = optuna.study.create_study(
        study_name=data_container.study_name,
        directions=data_container.study.user_attrs['directions'],
    )
    combined_study.add_trials(trials=trials)
    for i, _user_attrs in enumerate(user_attrs):
        combined_study.set_user_attr(key=str(i), value=_user_attrs)
    return combined_study


def get_empirical_pareto_front(path_to_exp_results: [str, Path], output_file: Union[str, Path]):
    path_to_exp_results = Path(path_to_exp_results)
    db_files = list(path_to_exp_results.rglob(BenchmarkToolsConstants.DATABASE_NAME.value))
    data_containers = [DataContainer(storage_path=db_file) for db_file in db_files]
    combined_study = _combine_multiple_data_container(data_containers)

    pareto_trials = _get_pareto_front_trials(combined_study)
    pareto_points = []
    for t in pareto_trials:
        # Cast everything to a min problem.
        values = [_normalize_value(v, d) for v, d in zip(t.values, combined_study.directions)]
        pareto_points.append(values)
    pareto_points = np.array(pareto_points)

    non_pareto_points = []
    pareto_trials_indices = [t.number for t in pareto_trials]
    for t in combined_study.trials:
        if t.number not in pareto_trials_indices:  # TODO: adapt the trial number to be unique
            values = [_normalize_value(v, d) for v, d in zip(t.values, combined_study.directions)]
            non_pareto_points.append(values)

    nadir = np.max(pareto_points, axis=0)
    utopic = np.min(pareto_points, axis=0)  # Either theoretical or empirical

    # Compute Hypervolume inidcator + distance to hv area.
    reference_point = np.array([1, 1])

    # normalize with nadir and utopic:
    pareto_points -= utopic
    pareto_points /= nadir - utopic

    non_pareto_points -= utopic
    non_pareto_points /= nadir - utopic

    hypervolume = WFG().compute(solution_set=pareto_points, reference_point=reference_point)
    min_distance = np.min(non_pareto_points - reference_point, axis=1)

    fig = optuna.visualization.plot_pareto_front(
        combined_study,
        target_names=combined_study.user_attrs['0']['objective_names'],
        include_dominated_trials=False
    )
    fig.show()

    print('test')



if __name__ == '__main__':
    get_empirical_pareto_front(
        '/home/pm/Dokumente/Code/BenchmarkTools/Results/yahpo',
        output_file='/home/pm/Dokumente/Code/BenchmarkTools/Results/yahpo/empricial_pareto_front.csv'
    )