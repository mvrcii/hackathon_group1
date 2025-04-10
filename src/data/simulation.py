import os
import time
from typing import Tuple

import einops
import h5py
import numpy as np
import numpy.typing as npt

from src.data.dataclasses import SimulationRawData, SimulationData, CoilConfig


class Simulation:
    def __init__(self,
                 path: str,
                 coil_path: str = "data/antenna/antenna.h5"):
        self.path = path
        self.coil_path = coil_path

        self.simulation_raw_data = self._load_raw_simulation_data()
        self.subject_indices = np.where(self.simulation_raw_data.subject)

    def _load_raw_simulation_data(self) -> SimulationRawData:
        # Load raw simulation data from path

        def read_field() -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            with h5py.File(self.path) as f:
                re_efield, im_efield = f["efield"]["re"][:], f["efield"]["im"][:]
                re_hfield, im_hfield = f["hfield"]["re"][:], f["hfield"]["im"][:]
                field = np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)], axis=0)
            return field

        def read_physical_properties() -> npt.NDArray[np.float32]:
            with h5py.File(self.path) as f:
                physical_properties = f["input"][:]
            return physical_properties

        def read_subject_mask() -> npt.NDArray[np.bool_]:
            with h5py.File(self.path) as f:
                subject = f["subject"][:]
            subject = np.max(subject, axis=-1)
            return subject

        def read_coil_mask() -> npt.NDArray[np.float32]:
            with h5py.File(self.coil_path) as f:
                coil = f["masks"][:]
            return coil

        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawData(
            simulation_name=read_simulation_name(),
            properties=read_physical_properties(),
            field=read_field(),
            subject=read_subject_mask(),
            coil=read_coil_mask()
        )

        return simulation_raw_data

    def _shift_field(self,
                     field: npt.NDArray[np.float32],
                     phase: npt.NDArray[np.float32],
                     amplitude: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils.
        """
        # Calculate phase coefficients
        re_phase = np.cos(phase) * amplitude
        im_phase = np.sin(phase) * amplitude

        # Create coefficients arrays
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(field, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift

    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:
        field = self.simulation_raw_data.field
        mask = self.simulation_raw_data.subject

        # 1) Extract only masked voxels: shape e.g. (2,2,3,#mask_vox,8)
        field_in_mask = field[..., mask, :]

        # 2) Shift field in the reduced array
        shifted_in_mask = self._shift_field(
            field_in_mask,
            coil_config.phase,
            coil_config.amplitude
        )  # shape (2,2,3,#mask_vox,8)

        field_shifted = np.zeros(field.shape[:-1], dtype=field.dtype)  # coil dimension removed
        field_shifted[..., mask] = shifted_in_mask  # shape (2,2,3,#mask)

        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data

    def __call__(self, coil_config: CoilConfig) -> SimulationData:
        simulation_data = self.phase_shift(coil_config)

        return simulation_data


if __name__ == '__main__':
    simulation = Simulation("data/simulations/children_0_tubes_2_id_19969.h5")
    default_config = CoilConfig()

    num_runs = 50
    simulation_times = []  # List to store each run's elapsed time

    # Run the simulation multiple times and record the time taken for each run.
    for _ in range(num_runs):
        start_time = time.perf_counter()
        default_sim_data = simulation(default_config)
        iteration_time = time.perf_counter() - start_time
        print(f"Iteration {_ + 1}: {iteration_time:.6f} seconds")
        simulation_times.append(iteration_time)

    # Calculate total and average times.
    total_time = sum(simulation_times)
    average_time = total_time / num_runs

    print(f"Total simulation time for '{simulation.simulation_raw_data.simulation_name}': {total_time:.6f} seconds")
    print(f"Average simulation time per run: {average_time:.6f} seconds")
