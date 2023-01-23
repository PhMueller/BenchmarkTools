"""
This script extracts the empirical pareto front from a benchmark.
It scraps all results in a certain directory and computes the pareto front across the combination of the evaluated
configurations.
"""

from typing import Union
from pathlib import Path

import optuna.visualization
import tqdm
import yaml

from BenchmarkTools.utils.constants import BenchmarkToolsConstants
from BenchmarkTools.evaluations.data_container import DataContainer, combine_multiple_data_container

from optuna._hypervolume import WFG
from optuna.study._multi_objective import _get_pareto_front_trials, _normalize_value
import numpy as np


def get_empirical_pareto_front(
        experiment_name: str,
        experiment_result_dir: [str, Path],
        combined_study: Union[None, optuna.study.Study],
        output_dir: Union[str, Path]
        ):

    output_dir = Path(output_dir)
    output_file = output_dir / BenchmarkToolsConstants.MO_EMP_PF_SUMMARY_FILE_NAME

    statistics = {}
    if output_file.exists():
        with open(output_file, 'r') as fh:
            statistics = yaml.full_load(fh)

    if experiment_name in statistics:
        return statistics[experiment_name]

    if combined_study is None:
        from BenchmarkTools.evaluations.data_container import load_data_containers_from_directory
        data_containers = load_data_containers_from_directory(experiment_result_dir)
        combined_study = combine_multiple_data_container(data_containers)

    # Extract the pareto front over the combined pareto front
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
        if t.number not in pareto_trials_indices:
            values = [_normalize_value(v, d) for v, d in zip(t.values, combined_study.directions)]
            non_pareto_points.append(values)

    nadir = np.max(pareto_points, axis=0)
    ideal = np.min(pareto_points, axis=0)  # != utopic

    # Compute hypervolume indicator + distance to hv area.
    reference_point = np.array([1, 1])

    # normalize with nadir and utopic:
    norm_pareto_points = (pareto_points - ideal) / (nadir - ideal)
    norm_non_pareto_points = (non_pareto_points - ideal) / (nadir - ideal)

    hypervolume = WFG().compute(solution_set=norm_pareto_points, reference_point=reference_point)

    summary = {experiment_name: {
        'best_point': ideal.tolist(),
        'nadir_point': nadir.tolist(),
        'hypervolume': hypervolume,
        'num_points_in_pf': len(pareto_points),
        'num_observations': len(pareto_points) + len(non_pareto_points)
    }}
    statistics.update(summary)

    with open(output_file, 'w') as fh:
        statistics = yaml.dump(statistics, fh, yaml.Dumper)

    fig = optuna.visualization.plot_pareto_front(
        combined_study,
        target_names=combined_study.user_attrs['0']['objective_names'],
        include_dominated_trials=False
    )
    fig.show()

    return statistics[experiment_name]
