from BenchmarkTools.evaluations.trajectories import plot_trajectories_per_objective
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from BenchmarkTools.evaluations.data_container import (
    DataContainerFromSQLite,
    load_data_containers_from_directory,
    combine_multiple_data_container
)
from BenchmarkTools.evaluations.trajectories import plot_trajectories_per_objective
from BenchmarkTools.evaluations.pareto_front import plot_pareto_front_one_experiment
from BenchmarkTools.evaluations.run_statistics import run_statistics


# ################################ ARGUMENTS ######################################################################### #
benchmark_name = 'YAHPO_RBV2_28'
benchmark_result_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results'
output_dir = '/home/pm/Dokumente/Code/BenchmarkTools/Results/Plots/'
# ################################ ARGUMENTS ######################################################################### #

# ################################ INITIALIZE ######################################################################## #
# Set the correct paths and load the corresponding data
benchmark_result_dir = Path(benchmark_result_dir) / benchmark_name
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Load all run histories (=DataContainers)
data_containers: List[DataContainerFromSQLite] = \
    load_data_containers_from_directory(experiment_result_dir=benchmark_result_dir)

# Group them by optimizer -> Dict[OptimizerName: List[DataContainer1, DataContainer2, ..]]
data_containers_by_optimizer: Dict[str, List[DataContainerFromSQLite]] = defaultdict(lambda: [])
for data_container in data_containers:
    data_containers_by_optimizer[data_container.optimizer].append(data_container)

# Aggregate the runhistories of each optimizer: [DC_1, DC_2, .., DC_k] --> DC_combined
data_containers_combined = []
for optimizer in data_containers_by_optimizer.keys():
    dc = combine_multiple_data_container(data_containers_by_optimizer[optimizer], same_optimizer=True)
    data_containers_combined.append(dc)
# ################################ INITIALIZE ######################################################################## #

# || Benchmark | Optimizer | RunID  || Wallclock Time | Total Time Used (+SurogateCosts) | Num TAE ||
# || ------------------------------ || ----------------------------------------------------------- ||
# TODO: Add surrogate costs
run_statistics(data_containers, output_dir, benchmark_name)

# TODO: Extract performance table


# ################################ PLOTTING 1: Trajectories over time ################################################ #
plot_trajectories_per_objective(
    data_containers_by_optimizer=data_containers_by_optimizer,
    output_dir=output_dir / 'TrajectoriesPerObjective',
    benchmark_name=benchmark_name,
)
# ################################ PLOTTING 1: Trajectories over time ################################################ #

# ################################ PLOTTING 2: Pareto Front ########################################################## #
plot_pareto_front_one_experiment(
    data_containers=data_containers_combined,
    output_dir=output_dir / 'ParetoFront',
    benchmark_name=benchmark_name,
)
# ################################ PLOTTING 2: Pareto Front ########################################################## #
