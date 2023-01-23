class BudgetExhaustedException(Exception):
    """
    This exception is called when an optimization run has reached the run limit.
    """
    pass


class AlreadyFinishedException(Exception):
    """
    This exception is called when a run already is about to being started but already exists and has finished.
    """
    pass
