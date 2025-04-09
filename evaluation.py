import json

from main import run
from src.costs.b1_homogeneity_sar_marcel import B1HomogeneitySARCost
from src.data import Simulation
from src.utils import evaluate_coil_config

if __name__ == "__main__":
    simulation = Simulation("data/simulations/children_0_tubes_2_id_19969.h5")
    cost_function = B1HomogeneitySARCost(lambda_weight=0.05)

    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function, timeout=20)

    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
