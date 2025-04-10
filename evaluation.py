import json

from main import run
from src.costs.b1_homogeneity_sar_marcel import B1HomogeneitySARCost, B1HomogeneitySARCost_torch
from src.costs.b1_homogeneity import B1HomogeneityCost
from src.data import Simulation
from src.utils import evaluate_coil_config

if __name__ == "__main__":
    timeout = 300
    lambda_weight = 0.10

    simulation = Simulation("data/simulations/children_0_tubes_2_id_19969.h5")
    #cost_function = B1HomogeneitySARCost(lambda_weight=lambda_weight)
    cost_function = B1HomogeneityCost()

    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function, timeout=timeout)
    # convert back to numpy

    #simulation.simulation_raw_data.coil =simulation.simulation_raw_data.coil.numpy()
    #simulation.simulation_raw_data.field = simulation.simulation_raw_data.field.numpy()



    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
