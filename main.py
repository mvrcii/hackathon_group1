from src.costs.base import BaseCost
from src.data import Simulation, CoilConfig
from src.optimizers.optuna_optimizer import OptunaOptimizer


def run(simulation: Simulation,
        cost_function: BaseCost,
        timeout: int = 300) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    config = {
        'max_time_seconds': timeout,  # Use the full available timeout
        'time_buffer_seconds': 3,  # Buffer to ensure we return before timeout
        'n_startup_trials': 10,  # Number of random trials before using Bayesian optimization
    }

    optimizer = OptunaOptimizer(cost_function=cost_function, **config)
    return optimizer.optimize(simulation)
