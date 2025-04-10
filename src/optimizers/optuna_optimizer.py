import logging
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
    Employs a two-phase optimization strategy:
      - Phase 1: TPE warmup with a fixed number of startup trials.
      - Phase 2: Refinement with CMA-ES until the timeout.

    Attributes:
        cost_function (BaseCost): The cost function to evaluate configurations.
        max_time_seconds (int): Total allowed optimization time (in seconds).
        time_buffer_seconds (int): A buffer to ensure safe termination before the timeout.
        n_startup_trials (int): Number of trials to run during the TPE warmup phase.
        verbose (bool): If True, logs progress and update messages.
        direction (str): Optimization direction ("maximize" or "minimize").
        history (dict): Records details about each trial.
        best_cost (float): Best cost value found so far.
        best_config (CoilConfig): Configuration yielding the best cost.
    """

    def __init__(
            self,
            cost_function: BaseCost,
            timeout: int = 100,
            time_buffer_seconds: int = 2,
            n_startup_trials: int = 250,
            verbose: bool = True
    ):
        super().__init__(cost_function)
        self.max_time_seconds = timeout
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
        Run the two-phase optimization:
          1. Warm-up phase using TPE.
          2. Refinement phase using CMA-ES, which runs until the timeout.

        Args:
            simulation (Simulation): Simulation object to evaluate configurations.

        Returns:
            CoilConfig: The best configuration found within the time constraint.
        """
        start_time = time.time()
        end_time = start_time + self.max_time_seconds - self.time_buffer_seconds

        # Evaluate default configuration as the baseline.
        default_config = CoilConfig()
        default_sim_data = simulation(default_config)
        default_cost = self.cost_function(default_sim_data)

        logging.info("Starting Optuna optimization")
        logging.info(f"Time constraint: {self.max_time_seconds}s (with {self.time_buffer_seconds}s buffer)")
        logging.info(f"Baseline default configuration cost: {default_cost:.6f}")
        logging.info(f"Cost function: {self.cost_function.__class__.__name__}")
        logging.info(f"Optimization direction: {self.cost_function.direction}")

        # Initialize the best result with the default configuration.
        self.best_cost = default_cost
        self.best_config = default_config

        def objective_fn(trial):
            # Check if we are nearing the optimization time limit.
            if time.time() + 1.0 > end_time:  # 1-second safety margin.
                raise optuna.exceptions.TrialPruned("Time limit approaching")

            # Suggest parameters for phase and amplitude for 8 coils.
            phase = np.array([trial.suggest_float(f"phase_{i}", 0, 2 * np.pi)
                              for i in range(8)])
            amplitude = np.array([trial.suggest_float(f"amplitude_{i}", 0.2, 1)
                                  for i in range(8)])

            # Create configuration, run simulation and compute cost.
            config = CoilConfig(phase=phase, amplitude=amplitude)
            sim_data = simulation(config)
            cost_value = self.cost_function(sim_data)

            # Record trial details in history.
            current_time = time.time()
            trial_number = len(self.history['trial_number'])
            self.history['trial_number'].append(trial_number)
            self.history['timestamp'].append(current_time)
            self.history['cost'].append(cost_value)
            self.history['phase'].append(phase.copy())
            self.history['amplitude'].append(amplitude.copy())

            # Update best result if current cost is better.
            is_better = (cost_value > self.best_cost) if self.direction == "maximize" else (cost_value < self.best_cost)
            if is_better:
                self.best_cost = cost_value
                self.best_config = config
                if self.verbose:
                    improvement = abs(cost_value - default_cost) / abs(default_cost) * 100
                    relation = ("better" if cost_value > default_cost else "worse")
                    timestamp = time.time() - start_time
                    logging.info(f"{timestamp:.2f}s - New best: {cost_value:.6f} ({relation} than default by {improvement:.2f}%)")

            return cost_value

        # --- Phase 1: Warm-up with TPE ---
        logging.info(f"Starting TPE warm-up for {self.n_startup_trials} trials...")
        tpe_sampler = optuna.samplers.TPESampler(seed=42)
        tpe_study = optuna.create_study(direction=self.direction, sampler=tpe_sampler)
        tpe_study.optimize(objective_fn, timeout=5, n_jobs=-1)
        best_tpe_trial = tpe_study.best_trial

        logging.info(f"Finished TPE warm-up. Best value so far: {best_tpe_trial.value:.6f}")
        logging.info(f"Best params from TPE: {best_tpe_trial.params}")

        # --- Phase 2: Refinement with CMA-ES ---
        # Initialize CMA-ES with TPE's best parameters.
        cmaes_sampler = optuna.samplers.CmaEsSampler(x0=best_tpe_trial.params)
        cmaes_study = optuna.create_study(direction=self.direction, sampler=cmaes_sampler)
        logging.info("Starting CMA-ES refinement phase...")

        try:
            remaining_time = end_time - time.time()
            cmaes_study.optimize(
                objective_fn,
                timeout=remaining_time,
                n_jobs=-1,  # Use all available CPU cores.
                show_progress_bar=False,
                catch=(optuna.exceptions.TrialPruned,)
            )
        except KeyboardInterrupt:
            logging.warning("Optimization interrupted by user.")
        except optuna.exceptions.TrialPruned:
            logging.warning("Optimization halted due to time constraints.")

        best_cmaes_trial = cmaes_study.best_trial
        logging.info(f"Finished CMA-ES phase. Best value: {best_cmaes_trial.value:.6f}")
        logging.info(f"Best params from CMA-ES: {best_cmaes_trial.params}")

        # --- Summary ---
        total_time = time.time() - start_time
        completed_trials = [t for t in cmaes_study.trials if t.state == TrialState.COMPLETE]
        pruned_trials = [t for t in cmaes_study.trials if t.state == TrialState.PRUNED]

        logging.info("\nOptimization complete:")
        logging.info(f"  Total time: {total_time:.2f} seconds")
        logging.info(f"  Trials completed: {len(completed_trials)}")
        logging.info(f"  Trials pruned: {len(pruned_trials)}")
        logging.info(f"  Default cost: {default_cost:.6f}")
        logging.info(f"  Best cost found: {self.best_cost:.6f}")

        if self.direction == "maximize":
            relative_change = (self.best_cost - default_cost) / abs(default_cost) * 100
            comparison = "higher"
        else:
            relative_change = (default_cost - self.best_cost) / abs(default_cost) * 100
            comparison = "lower"

        logging.info(f"  Optimized result is {abs(relative_change):.2f}% {comparison} than default")

        return self.best_config

    def _create_fixed_params_trial(self, config: CoilConfig) -> optuna.trial.FrozenTrial:
        """
        Create a frozen trial with fixed parameters based on an existing configuration.

        Args:
            config (CoilConfig): The configuration to convert into a trial.

        Returns:
            optuna.trial.FrozenTrial: A trial with preset parameter values and distributions.
        """
        params = {}
        for i in range(8):
            params[f"phase_{i}"] = config.phase[i]
            params[f"amplitude_{i}"] = config.amplitude[i]

        distributions = {
            **{f"phase_{i}": optuna.distributions.FloatDistribution(0, 2 * np.pi) for i in range(8)},
            **{f"amplitude_{i}": optuna.distributions.FloatDistribution(0.2, 1) for i in range(8)}
        }

        return optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=self.best_cost
        )
