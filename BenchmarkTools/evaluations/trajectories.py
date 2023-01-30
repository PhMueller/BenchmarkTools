from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from BenchmarkTools import logger
from BenchmarkTools.evaluations.data_container import (
    load_data_containers_from_directory,
    DataContainer,
)
from BenchmarkTools.evaluations.plotting_utils import color_to_rgba_str, make_marker
from BenchmarkTools.utils.loader_tools import load_optimizer_settings


def plot_trajectories_per_objective(
        data_containers_by_optimizer: Dict[str, DataContainer],
        output_dir: Path,
        benchmark_name: str
):
    n_objectives = None
    figures = None
    # TODO: Rename that feature and make it a parameter to the function!
    x_axis = 'number'  # Time

    # Aggregate the runhistories of each optimizer
    for i_optimizer, optimizer in enumerate(data_containers_by_optimizer.keys()):
        plotting_settings = None
        directions = None
        objective_names = None
        color_rgba = None
        color_rgba_transparent = None

        # Aggregate the runs per optimizer
        dfs_per_optimizer = []
        for data_container in data_containers_by_optimizer[optimizer]:
            if n_objectives is None:
                n_objectives = len(data_container.study.directions)
            if plotting_settings is None:
                plotting_settings = load_optimizer_settings(data_container.optimizer).plotting
                color_rgba = color_to_rgba_str(plotting_settings.color, opacity=1.0)
                color_rgba_transparent = color_to_rgba_str(plotting_settings.color, opacity=0.3)

            if objective_names is None:
                objective_names = data_container.study.user_attrs['objective_names']
                directions = [direction.name for direction in data_container.study.directions]

            trials_df = data_container.study.trials_dataframe()
            trials_df = trials_df[trials_df.state == 'COMPLETE']

            # TODO: Add surrogate cost column if it is a surrogate benchmark!
            objective_columns = [col for col in trials_df.columns if col.startswith('values_')]
            select_cols = ['number', 'datetime_complete'] + objective_columns
            trials_df = trials_df.loc[:, select_cols]
            trials_df.loc[:, 'run_id'] = data_container.run_id
            dfs_per_optimizer.append(trials_df)

        # TODO: Ich möchte hier:
        # a) x = Zeit b = Objective i
        # b ) x = TAE  y = Objective i
        dfs_per_optimizer = pd.concat(dfs_per_optimizer, axis=0)

        if x_axis == 'number':
            pivot = dfs_per_optimizer.pivot(index=x_axis, columns='run_id', values=objective_columns)
        else:
            # TODO: write that for time
            raise NotImplementedError()

        if figures is None:
            figures: List[go.Figure] = [Figure(layout=go.Layout(width=1000, height=800)) for _ in range(n_objectives)]

        for i_obj_name, obj_name in enumerate(objective_names):

            # Select the correct figure object
            fig: go.Figure = figures[i_obj_name]

            corresponding_col = f'values_{i_obj_name}'
            sub_pivot = pivot.loc[:, corresponding_col]

            # Determine the runtime per run (index of last not-nan entry)
            last_valid_ids = sub_pivot.apply(pd.Series.last_valid_index).tolist()
            sub_pivot = sub_pivot.ffill()

            if directions[i_obj_name] == 'MAXIMIZE':
                sub_pivot = sub_pivot.cummax(axis=0)
            else:
                sub_pivot = sub_pivot.cummin(axis=0)

            mean = sub_pivot.mean(axis=1)
            std = sub_pivot.std(axis=1)

            fig.add_trace(
                go.Scatter(
                    x=sub_pivot.index,
                    y=mean,
                    mode='lines',
                    line={'dash': plotting_settings.linestyle},
                    marker=make_marker(plotting_settings=plotting_settings, opacity=1.0),
                    showlegend=True,
                    name=plotting_settings.display_name
                ),
            )

            # Plot the confidence bounds
            fig.add_trace(
                go.Scatter(
                    name='Upper Bound',
                    x=sub_pivot.index,
                    y=mean + std,
                    mode='lines',
                    marker={
                        "line": {"width": 0.5, "color": 'DarkSlateGrey', },
                        "color": color_rgba,
                        'opacity': 0.1
                    },
                    line={'dash': plotting_settings.linestyle, 'width': 1},
                    showlegend=False
                    ),
            )
            fig.add_trace(
                go.Scatter(
                    name='Lower Bound',
                    x=sub_pivot.index,
                    y=mean - std,
                    marker={
                        "line": {"width": 0.5, "color": 'DarkSlateGrey', },
                        "color": color_rgba,
                        'opacity': 0.1
                    },
                    line={'dash': plotting_settings.linestyle, 'width': 1},
                    mode='lines',
                    fillcolor=color_rgba_transparent,
                    fill='tonexty',
                    showlegend=False
                ),
            )

            # Plot the points where a run has had its last observation
            fig.add_trace(
                go.Scatter(
                    x=last_valid_ids,
                    y=mean.iloc[last_valid_ids],
                    mode='markers',
                    showlegend=False,
                    marker=make_marker(plotting_settings, opacity=1.0),
                ),
            )

            # Do this only once: Set the correct label names and update the layout.
            if i_optimizer == 0:
                fig.update_layout(
                    title_text=f"Trajectories: {benchmark_name}",
                    xaxis_title='TAE' if x_axis == 'number' else 'Wallclock Time',
                    yaxis_title=obj_name,
                    legend_title="Optimizers",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="RebeccaPurple"
                    )
                )

    for objective_name, fig in zip(objective_names, figures):
        plot_file = output_dir / f'trajectory_{benchmark_name}_{objective_name}_{x_axis}.html'
        fig.write_html(plot_file, include_mathjax='cdn')
        logger.info(f'Pareto-front plot saved to {plot_file}')


if __name__ == '__main__':

    benchmark_name = 'YAHPO_RBV2_28'
    benchmark_result_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results'
    output_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results/Plots/TrajectoriesPerObjective'

    benchmark_result_dir = Path(benchmark_result_dir) / benchmark_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all run histories
    data_containers = load_data_containers_from_directory(experiment_result_dir=benchmark_result_dir)

    # Group them by optimizer
    data_containers_by_optimizer = defaultdict(lambda: [])
    for data_container in data_containers:
        data_containers_by_optimizer[data_container.optimizer].append(data_container)

    plot_trajectories_per_objective(
        data_containers_by_optimizer=data_containers_by_optimizer,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
    )
