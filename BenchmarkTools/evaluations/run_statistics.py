from pathlib import Path
from pathlib import Path
from typing import List

import pandas as pd

from BenchmarkTools.evaluations.data_container import load_data_containers_from_directory, DataContainer


def run_statistics(
    data_containers: List[DataContainer],
    output_dir: Path,
    benchmark_name: str,
):

    stats_file = output_dir / 'ExperimentStatistics.csv'

    new_data = []
    for dc in data_containers:

        df = dc.study.trials_dataframe()
        df = df[df.state == 'COMPLETE']
        entry = {
            'benchmark_name': benchmark_name,
            'optimizer_name': dc.optimizer,
            'run_id': str(dc.run_id),
            'wallclock_time_used': df.duration.sum().total_seconds(),
            # TODO: Add cost column!
            'total_budget_used': df.duration.sum().total_seconds() + 0,
            'num_tae_used': len(df),
        }
        new_data.append(entry)

    new_data = pd.DataFrame(new_data)

    if stats_file.exists():
        old_data = pd.read_csv(stats_file)
        old_data['run_id'] = old_data['run_id'].apply(str)
        new_data = pd.concat([new_data, old_data], axis=0)
        new_data = new_data.drop_duplicates(subset=['benchmark_name', 'optimizer_name', 'run_id'], keep='first')

    new_data = new_data.sort_values(by=['benchmark_name', 'optimizer_name', 'run_id'])
    new_data.to_csv(stats_file, index=False)


if __name__ == '__main__':

    benchmark_name = 'YAHPO_RBV2_28'
    benchmark_result_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results'
    output_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results/Plots'

    benchmark_result_dir = Path(benchmark_result_dir) / benchmark_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all run histories
    data_containers = load_data_containers_from_directory(experiment_result_dir=benchmark_result_dir)

    run_statistics(
        data_containers=data_containers,
        output_dir=output_dir,
        benchmark_name=benchmark_name,
    )
