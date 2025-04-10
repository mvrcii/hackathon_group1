from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable
import numpy as np

from tqdm import trange
import torch

from torch.optim import LBFGS
from typing import Optional

class DummyOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        
    def _sample_coil_config(self) -> CoilConfig:
        phase = np.random.uniform(low=0, high=2*np.pi, size=(8,))
        amplitude = np.random.uniform(low=0, high=1, size=(8,))
        return CoilConfig(phase=phase, amplitude=amplitude)
        
    def optimize(self, simulation: Simulation):
        best_coil_config = None
        best_cost = -np.inf if self.direction == "maximize" else np.inf
        
        pbar = trange(self.max_iter)
        for i in pbar:
            coil_config = self._sample_coil_config()
            simulation_data = simulation(coil_config)
            
            cost = self.cost_function(simulation_data)
            if (self.direction == "minimize" and cost < best_cost) or (self.direction == "maximize" and cost > best_cost):
                best_cost = cost
                best_coil_config = coil_config
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")
        
        return best_coil_config


import numpy as np
from scipy.optimize import minimize
from tqdm import trange


class LBFGSOptimizer(BaseOptimizer):
    """
    NumPy implementation of L-BFGS optimizer for coil configuration optimization.
    Maintains the same interface as DummyOptimizer but with proper gradient-based optimization.
    """

    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 10,
                 num_coils: int = 8):
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.num_coils = num_coils

        # 7T-specific constraints
        self.phase_bounds = (0, 2 * np.pi)
        self.amp_bounds = (0.2, 1.0)  # For better B1 homogeneity at 7T

    def _pack_parameters(self, phases, amplitudes):
        """Combine phases and amplitudes into a single parameter vector"""
        return np.concatenate([phases, amplitudes])

    def _unpack_parameters(self, x):
        """Extract phases and amplitudes from parameter vector"""
        phases = x[:self.num_coils]
        amps = x[self.num_coils:]
        return phases, amps

    def _apply_constraints(self, x):
        """Apply physical constraints to parameters"""
        phases, amps = self._unpack_parameters(x)

        # Phase wrapping [0, 2π]
        phases = np.mod(phases, 2 * np.pi)

        # Amplitude normalization [0.2, 1.0]
        amps = np.clip(amps, 0, 1)  # First clip to [0,1]
        amps = 0.2 + 0.8 * (amps / np.sum(amps))  # Then scale to [0.2,1.0]

        return self._pack_parameters(phases, amps)

    def _scipy_cost_wrapper(self, x, simulation):
        """Wrapper function for scipy.optimize"""
        x = self._apply_constraints(x)
        phases, amplitudes = self._unpack_parameters(x)

        coil_config = CoilConfig(
            phase=phases,
            amplitude=amplitudes
        )

        simulation_data = simulation(coil_config)
        cost = self.cost_function(simulation_data)

        return cost if self.direction == "minimize" else -cost

    def optimize(self, simulation: Simulation):
        # Initial random guess within bounds
        x0 = self._pack_parameters(
            np.random.uniform(*self.phase_bounds, size=self.num_coils),
            np.random.uniform(*self.amp_bounds, size=self.num_coils)
        )

        # Callback for progress tracking
        best_cost = np.inf if self.direction == "minimize" else -np.inf
        best_x = x0.copy()

        def callback(x):
            nonlocal best_cost, best_x
            current_cost = self._scipy_cost_wrapper(x, simulation)
            if (self.direction == "minimize" and current_cost < best_cost) or \
                    (self.direction == "maximize" and current_cost > best_cost):
                best_cost = current_cost
                best_x = x.copy()

        # Run L-BFGS optimization
        res = minimize(
            fun=self._scipy_cost_wrapper,
            x0=x0,
            args=(simulation,),
            method='L-BFGS-B',
            jac=None,  # Let scipy approximate gradient
            options={
                'maxiter': self.max_iter,
                'disp': True
            },
            bounds=[self.phase_bounds] * self.num_coils + [self.amp_bounds] * self.num_coils,
            callback=callback
        )

        # Extract best parameters found
        phases, amplitudes = self._unpack_parameters(best_x)
        return CoilConfig(
            phase=phases,
            amplitude=amplitudes
        )
