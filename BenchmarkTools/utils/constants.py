from enum import Enum


class _BaseEnum(str, Enum):
    def __str__(self):
        return str(self.value)

    def __getitem__(self, item):
        return str(item.value)


class BenchmarkToolsTrackMetrics(_BaseEnum):

    COST = 'cost'
    WALLCLOCK_CONFIG_START = 'EXP_WALLCLOCK_CONFIG_START'
    WALLCLOCK_CONFIG_END = 'EXP_WALLCLOCK_CONFIG_END'


class BenchmarkToolsConstants(_BaseEnum):
    FINISHED_FLAG = 'RUN_HAS_FINISHED.FLAG'
