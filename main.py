from src.costs.base import BaseCost
from src.data import Simulation, CoilConfig
from src.optimizers.optuna_optimizer import OptunaOptimizer


def run(simulation: Simulation,
        cost_function: BaseCost,
        timeout: int = 10) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    optimizer = OptunaOptimizer(
        cost_function=cost_function,
        timeout=timeout
    )
    return optimizer.optimize(simulation)
