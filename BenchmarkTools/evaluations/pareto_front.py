from pathlib import Path
from typing import Union

from BenchmarkTools.evaluations.data_container import (
    DataContainerFromSQLite,
    load_data_containers_from_directory,
    combine_multiple_data_container,
    DataContainer,
)

from BenchmarkTools import logger
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict
from BenchmarkTools.evaluations.data_container import combine_multiple_data_container
import optuna
from BenchmarkTools.utils.loader_tools import load_optimizer_settings
from optuna.visualization._pareto_front import _get_pareto_front_info, plot_pareto_front
from plotly.graph_objects import Figure
import plotly.graph_objects as go

from omegaconf import DictConfig


def _make_marker(
    plotting_settings: DictConfig,
    dominated_trials: bool = False,
) -> Dict[str, Any]:

    if dominated_trials:
        return {
            "line": {"width": 0.5, "color": 'DarkSlateGrey', },
            "color": plotting_settings.color,
            'opacity': 0.5
        }
    else:
        return {
            "line": {"width": 0.5, "color": 'DarkSlateGrey'},
            "color": plotting_settings.color
        }


def plot_pareto_front_one_experiment(data_containers: List[DataContainer], output_dir: Path, benchmark_name: str):

    plot_file = output_dir / f'pareto_front_{benchmark_name}.html'

    fig = Figure(layout=go.Layout(width=1000, height=800))

    for data_container in data_containers_combined:
        plotting_settings = load_optimizer_settings(data_container.optimizer).plotting

        non_dominated_marker = _make_marker(plotting_settings=plotting_settings, dominated_trials=False)
        dominated_marker = _make_marker(plotting_settings=plotting_settings, dominated_trials=True)

        pf_infos = _get_pareto_front_info(
            study=data_container.study,
            target_names=data_container.study.user_attrs['objective_names'],
            include_dominated_trials=True,
        )

        x = np.array([values[pf_infos.axis_order[0]] for _, values in pf_infos.best_trials_with_values])
        y = np.array([values[pf_infos.axis_order[1]] for _, values in pf_infos.best_trials_with_values])
        sorting = np.argsort(x)

        # Plot the pareto points. Add a line that visualizes the pareto front.
        fig.add_trace(go.Scatter(
            # Depending if the objective on the x axis is a min or max problem, do the step before (vh) or after (hv).
            # https://github.com/plotly/plotly.js/issues/51#issuecomment-160771872
            line={'shape': 'hv' if data_container.study.directions[0].name == 'MINIMIZE' else 'vh'},
            x=x[sorting],
            y=y[sorting],
            marker=non_dominated_marker,
            mode='lines+markers',
            showlegend=True,
            name=plotting_settings.display_name
        ))

        # Add the dominated points as well.
        fig.add_trace(go.Scatter(
            x=[values[pf_infos.axis_order[0]] for _, values in pf_infos.non_best_trials_with_values],
            y=[values[pf_infos.axis_order[1]] for _, values in pf_infos.non_best_trials_with_values],
            marker=dominated_marker,
            mode='markers',
            showlegend=False,
            name=plotting_settings.display_name
        ))

    fig.update_layout(
        title=f"Pareto Front: {benchmark_name}",
        xaxis_title=data_containers[0].study.user_attrs['objective_names'][0],
        yaxis_title=data_containers[0].study.user_attrs['objective_names'][1],
        legend_title="Optimizers",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    fig.write_html(plot_file, include_mathjax='cdn')
    logger.info(f'Pareto-front plot saved to {plot_file}')


if __name__ == '__main__':

    benchmark_name = 'YAHPO_RBV2_28'
    benchmark_result_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results'
    output_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results/Plots/ParetoFront'

    benchmark_result_dir = Path(benchmark_result_dir) / benchmark_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all run histories
    data_containers = load_data_containers_from_directory(experiment_result_dir=benchmark_result_dir)

    # Group them by optimizer
    data_containers_by_optimizer = defaultdict(lambda: [])
    for data_container in data_containers:
        data_containers_by_optimizer[data_container.optimizer].append(data_container)

    # Aggregate the runhistories of each optimizer
    data_containers_combined = []
    for optimizer in data_containers_by_optimizer.keys():
        dc = combine_multiple_data_container(data_containers_by_optimizer[optimizer], same_optimizer=True)
        data_containers_combined.append(dc)

    plot_pareto_front_one_experiment(
        data_containers=data_containers_combined,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
    )
