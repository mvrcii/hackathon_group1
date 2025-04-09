import time

import numpy as np
import optuna
from optuna.trial import TrialState

from .base import BaseOptimizer
from ..costs.base import BaseCost
from ..data.simulation import Simulation, CoilConfig


class OptunaOptimizer(BaseOptimizer):
    """
    Coil configuration optimizer using Optuna for efficient parameter search.
    Respects strict time constraints and provides detailed feedback.
    """

    def __init__(
            self,
            cost_function: BaseCost,
            max_time_seconds: int = 100,
            time_buffer_seconds: int = 3,
            n_startup_trials: int = 50,
            verbose: bool = True
    ):
        super().__init__(cost_function)
        self.max_time_seconds = max_time_seconds
        self.time_buffer_seconds = time_buffer_seconds
        self.n_startup_trials = n_startup_trials
        self.verbose = verbose
        self.direction = "maximize" if cost_function.direction == "maximize" else "minimize"

        # For tracking optimization progress
        self.history = {
            'trial_number': [],
            'timestamp': [],
            'cost': [],
            'phase': [],
            'amplitude': []
        }

        # For storing the best result
        self.best_cost = None
        self.best_config = None

    def optimize(self, simulation: Simulation) -> CoilConfig:
        """
        Optimize coil configuration using Optuna within the time constraint.

        Args:
            simulation: Simulation object to evaluate configurations

        Returns:
            The best CoilConfig found within the time constraint
        """
        start_time = time.time()
        end_time = start_time + self.max_time_seconds - self.time_buffer_seconds

        # Test the default configuration as a baseline
        default_config = CoilConfig()
        default_sim_data = simulation(default_config)
        default_cost = self.cost_function(default_sim_data)

        print(f"Starting Optuna optimization")
        print(f"Time constraint: {self.max_time_seconds}s (with {self.time_buffer_seconds}s buffer)")
        print(f"Baseline default configuration cost: {default_cost:.6f}")
        print(f"Cost function: {self.cost_function.__class__.__name__}")
        print(f"Optimization direction: {self.cost_function.direction}")

        # Initialize best with default values
        self.best_cost = default_cost
        self.best_config = default_config

        sampler = optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials)

        # Create Optuna study
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        # Add a default configuration as a starting point
        default_trial = self._create_fixed_params_trial(default_config)
        study.add_trial(default_trial)

        # Define the objective function for Optuna
        def objective(trial):
            # Check if we're approaching the time limit
            if time.time() + 1.0 > end_time:  # 1 second safety buffer for each trial
                # Raise exception to stop optimization
                raise optuna.exceptions.TrialPruned("Time limit approaching")

            # Suggest values for each phase and amplitude parameter
            phase = np.array([
                trial.suggest_float(f"phase_{i}", 0, 2 * np.pi) for i in range(8)
            ])

            amplitude = np.array([
                trial.suggest_float(f"amplitude_{i}", 0.2, 1) for i in range(8)
            ])

            # Create configuration and run simulation
            config = CoilConfig(phase=phase, amplitude=amplitude)
            sim_data = simulation(config)
            cost_value = self.cost_function(sim_data)

            # Update optimization history
            current_time = time.time()
            trial_number = len(self.history['trial_number'])
            self.history['trial_number'].append(trial_number)
            self.history['timestamp'].append(current_time)
            self.history['cost'].append(cost_value)
            self.history['phase'].append(phase.copy())
            self.history['amplitude'].append(amplitude.copy())

            # Update best result if better
            is_better = ((cost_value > self.best_cost) if self.direction == "maximize"
                         else (cost_value < self.best_cost))

            if is_better:
                self.best_cost = cost_value
                self.best_config = config

                # Print update on significant improvements
                if self.verbose:
                    improvement = abs(cost_value - default_cost) / abs(default_cost) * 100
                    relation = "better" if ((cost_value > default_cost) if self.direction == "maximize"
                                            else (cost_value < default_cost)) else "worse"
                    print(f"New best: {cost_value:.6f} ({relation} than default by {improvement:.2f}%)")

            return cost_value

        # Optimize using Optuna with timeout
        try:
            study.optimize(
                objective,
                timeout=(end_time - time.time()),
                n_jobs=-1,  # Use all available CPU cores
                show_progress_bar=True,
                catch=(optuna.exceptions.TrialPruned,)
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        except optuna.exceptions.TrialPruned:
            print("Optimization stopped due to time constraint.")

        # Extract best trial
        best_trial = study.best_trial

        # Report final results
        total_time = time.time() - start_time
        n_completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])

        print("\nOptimization complete:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Trials completed: {n_completed}")
        print(f"  Trials pruned: {n_pruned}")
        print(f"  Default cost: {default_cost:.6f}")
        print(f"  Best cost found: {self.best_cost:.6f}")

        if self.direction == "maximize":
            relative_change = (self.best_cost - default_cost) / abs(default_cost) * 100
            comparison = "higher" if self.best_cost > default_cost else "lower"
        else:
            relative_change = (default_cost - self.best_cost) / abs(default_cost) * 100
            comparison = "lower" if self.best_cost < default_cost else "higher"

        print(f"  Optimized result is {abs(relative_change):.2f}% {comparison} than default")

        return self.best_config

    def _create_fixed_params_trial(self, config: CoilConfig) -> optuna.trial.FrozenTrial:
        """Create a trial with fixed parameters based on an existing configuration"""
        params = {}
        for i in range(8):
            params[f"phase_{i}"] = config.phase[i]
            params[f"amplitude_{i}"] = config.amplitude[i]

        return optuna.trial.create_trial(
            params=params,
            distributions={
                              f"phase_{i}": optuna.distributions.FloatDistribution(0, 2 * np.pi)
                              for i in range(8)
                          } | {
                              f"amplitude_{i}": optuna.distributions.FloatDistribution(0.2, 1)
                              for i in range(8)
                          },
            value=self.best_cost
        )
