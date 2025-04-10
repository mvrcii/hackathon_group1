import json
import logging
import os
from datetime import datetime

import optuna
from rich.logging import RichHandler

from main import run
from src.costs import B1HomogeneityCost
from src.costs.b1_homogeneity_sar_marcel import B1HomogeneitySARCost
from src.data import Simulation
from src.utils import evaluate_coil_config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # or another level
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()


def save_results(sim_path, result, lambda_weight=None, timeout=300):
    sim_name = os.path.basename(sim_path).split(".")[0]
    target_dir = os.path.join("results", sim_name)
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_cost = result.get("best_coil_config_cost", "unknown")
    best_cost_str = f"{best_cost:.2f}" if isinstance(best_cost, (int, float)) else str(best_cost)
    config_info = f"l={lambda_weight}_t={timeout}" if lambda_weight else f"t={timeout}"
    filename = f"{timestamp}_{best_cost_str}_{config_info}.json"
    file_path = os.path.join(target_dir, filename)

    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    setup_logging()

    timeout = 300
    lambda_weight = None

    sim_path = "data/simulations/children_0_tubes_2_id_19969.h5"
    simulation = Simulation(sim_path)
    #cost_function = B1HomogeneitySARCost(lambda_weight=lambda_weight)
    cost_function = B1HomogeneityCost()

    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function, timeout=timeout)

    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)
    save_results(sim_path, result, lambda_weight, timeout)
