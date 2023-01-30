from enum import Enum


class _BaseEnum(str, Enum):
    """ Helperclass:
    Improves printing Enums. (print(enum_obj) shows now the value instead of EnumObject.name: value
    """
    def __str__(self):
        return str(self.value)

    def __getitem__(self, item):
        return str(item.value)


class BenchmarkToolsTrackMetrics(_BaseEnum):
    """
    Collection of all logging and metric tracking related names
    """
    COST = 'cost'
    WALLCLOCK_CONFIG_START = 'EXP_WALLCLOCK_CONFIG_START'
    WALLCLOCK_CONFIG_END = 'EXP_WALLCLOCK_CONFIG_END'


class BenchmarkToolsConstants(_BaseEnum):
    """
    Collection of names used in the project.
    """
    FINISHED_FLAG = 'RUN_HAS_FINISHED.FLAG'
    DATABASE_NAME = 'run_storage.db'
    OPT_HISTORY_NAME = 'optimization_history.csv'
    MO_EMP_PF_SUMMARY_FILE_NAME = 'summary_empirical_pareto_front.json'


class BenchmarkTypes(_BaseEnum):
    """
    Supported Tasks. Compare BenchmarkTools/benchmarks
    """
    HPOBENCH_CONTAINER = 'HPOBENCH_CONTAINER'
    HPOBENCH_LOCAL = 'HPOBENCH_LOCAL'
    BOTORCH_TOY = 'BOTORCH_TOY'
