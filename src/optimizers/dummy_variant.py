from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from ..optimizers.base import BaseOptimizer
import numpy as np
from tqdm import trange
import random
import copy
from threading import Timer

from typing import Callable

class DVOptimizer(BaseOptimizer):
    def __init__(self,
                 cost_function: BaseCost, max_iter:int=100):
        self.cost_function = cost_function
        self.direction = cost_function.direction
        assert self.direction in ["minimize", "maximize"], f"Invalid direction: {self.direction}"
        self.max_iter = max_iter
        self.t=Timer(max_iter, self.timeout)
        self.t.start()

    def _sample_coil_config(self):
        phase = np.random.uniform(low=0, high=2 * np.pi, size=(8,))
        amplitude = np.random.uniform(low=0, high=1, size=(8,))
        return CoilConfig(phase=phase, amplitude=amplitude)

    def _sample_single_change(self, original_config):
        new_config=copy.deepcopy(original_config)
        elem_id=random.randrange(0,16)
        if(elem_id<8):
            new_config.phase[elem_id]=random.uniform(0, 2*np.pi)
        else:
            new_config.amplitude[elem_id-8]=random.uniform(0,1)
        return new_config

    def optimize(self, simulation: Simulation):
        self.best_coil_config = None
        self.best_cost = -np.inf if self.direction == "maximize" else np.inf

        init_coil_config=self._sample_coil_config()

        pbar = trange(int(3*self.max_iter))
        for i in pbar:
            new_coil_config = self._sample_single_change(init_coil_config)
            #calculate solution with current parameters
            simulation_data = simulation(new_coil_config)

            #calculate cost
            cost = self.cost_function(simulation_data)

            if (self.direction == "minimize" and cost < self.best_cost) or (self.direction == "maximize" and cost > self.best_cost):
                self.best_cost = cost
                self.best_coil_config = new_coil_config
                init_coil_config = new_coil_config
                pbar.set_postfix_str(f"Best cost {self.best_cost:.2f}")


        return self.best_coil_config

    def timeout(self):
        return self.best_coil_config
