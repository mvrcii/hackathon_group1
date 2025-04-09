from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator
from ..data.utils import SARCalculator

import numpy as np


class B1HomogeneitySARCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()
        self.sar_calculator = SARCalculator()

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        sar = self.sar_calculator(simulation_data)
        subject = simulation_data.subject

        l=1.0

        b1_field_abs = np.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs[subject]
        max_sar=np.max(sar[subject])
        b1= np.mean(b1_field_subject_voxels) / np.std(b1_field_subject_voxels)
        return (b1-l*max_sar)

