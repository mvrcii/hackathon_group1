import numpy as np
import torch as t
from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator, SARCalculator, B1Calculator_torch, SARCalculator_torch


class B1HomogeneitySARCost(BaseCost):
    def __init__(self, lambda_weight: float = 0.01) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()
        self.sar_calculator = SARCalculator()
        self.lambda_weight = lambda_weight

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        subject = simulation_data.subject
        b1_field_abs = np.abs(b1_field)
        b1_subject = b1_field_abs[subject]
        b1_homogeneity = np.mean(b1_subject) / np.std(b1_subject)

        # Calculate peak SAR
        sar = self.sar_calculator(simulation_data)
        peak_sar = np.max(sar[subject])

        # Combined cost
        return b1_homogeneity - self.lambda_weight * peak_sar

class B1HomogeneitySARCost_torch(BaseCost):
    def __init__(self, lambda_weight: float = 0.01) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator_torch()
        self.sar_calculator = SARCalculator_torch()
        self.lambda_weight = lambda_weight

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        subject = simulation_data.subject
        b1_field_abs = t.abs(b1_field)
        b1_subject = b1_field_abs[subject]
        b1_homogeneity = t.mean(b1_subject) / t.std(b1_subject)

        # Calculate peak SAR
        sar = self.sar_calculator(simulation_data)
        peak_sar = t.max(sar[subject])

        # Combined cost
        return b1_homogeneity - self.lambda_weight * peak_sar
