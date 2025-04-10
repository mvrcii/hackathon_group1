import os
import time
from typing import Tuple

import einops
import h5py
import numpy as np
import numpy.typing as npt
from numba import njit, prange, set_num_threads

set_num_threads(8)
from src.data.dataclasses import SimulationRawData, SimulationData, CoilConfig


@njit(parallel=True)
def contract_field_coeffs(field: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Manual equivalent of:
        np.einsum('h r ... c, h R r c -> h R ...', field, coeffs)
    but JIT-compiled with Numba.
    """
    # Make sure 'field' is C-contiguous before reshape:
    field_c = np.ascontiguousarray(field)

    h = field_c.shape[0]
    r = field_c.shape[1]
    c = field_c.shape[-1]
    # Flatten the middle dimensions
    rest_shape = field_c.shape[2:-1]
    P = 1
    for dim in rest_shape:
        P *= dim

    # Now this reshape is allowed in nopython mode:
    field_2d = field_c.reshape((h, r, P, c))

    # shape of coeffs: (h, R, r, c)
    # prepare output
    out_2d = np.zeros((h, coeffs.shape[1], P), dtype=field_c.dtype)

    # Loop manually
    for i in prange(h):  # hf
        for j in prange(coeffs.shape[1]):  # reimout
            for k in range(r):  # reim
                for l in range(c):  # coils
                    out_2d[i, j, :] += field_2d[i, k, :, l] * coeffs[i, j, k, l]

    # Reshape back to (h, R, ...)
    out = out_2d.reshape((h, coeffs.shape[1]) + rest_shape)
    return out


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
                field = np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)],
                                 axis=0)
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
        field_shift = einops.einsum(field, coeffs,
                                    'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift

    def _shift_field_numpy(
            self,
            field: npt.NDArray[np.float32],
            phase: npt.NDArray[np.float32],
            amplitude: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Shift the field calculating:
            field_shifted = field * amplitude * exp(1j * phase),
        then summing over all coils.

        Shapes (conceptually):
          - field: (hf, reim, ..., coils)
          - phase, amplitude: (coils,)
          - output: (hf, reimout, ...)
        """

        # Compute real and imaginary components of (phase * amplitude)
        re_phase = np.cos(phase) * amplitude  # shape: (coils,)
        im_phase = np.sin(phase) * amplitude  # shape: (coils,)

        # Combine them into the needed arrangement
        #   coeffs_real = [ re_phase, -im_phase ] -> shape (2, coils)
        #   coeffs_im   = [ im_phase,  re_phase ] -> shape (2, coils)
        # -> stacked shape (2, 2, coils) = (reimout, reim, coils)
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)  # (2, coils)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)  # (2, coils)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)  # (2, 2, coils)

        # Repeat along an 'hf' dimension of size 2 -> final shape (2, 2, 2, coils)
        coeffs = np.expand_dims(coeffs, axis=0)  # (1, 2, 2, coils)
        coeffs = np.tile(coeffs, (2, 1, 1, 1))  # (2, 2, 2, coils)

        # Perform the summation over reim and coils using opt_einsum
        # field:  (hf, reim, fieldxyz..., coils)
        # coeffs: (hf, reimout, reim, coils)
        # --> out: (hf, reimout, fieldxyz...)
        field_shift = np.einsum(
            'h r ... c, h R r c -> h R ...',
            field,
            coeffs
        )

        return field_shift

    def _shift_field_numba(
            self,
            field: npt.NDArray[np.float32],
            phase: npt.NDArray[np.float32],
            amplitude: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Shift the field calculating:
            field_shifted = field * amplitude * exp(1j * phase),
        then summing over coils.
        CPU-based manual summation JIT-compiled by Numba for potential speedup.

        Shapes (conceptually):
          - field: (hf, reim, ..., coils)
          - phase, amplitude: (coils,)
          - output: (hf, reimout, ...)
        """

        # 1) Compute real and imaginary components of phase * amplitude
        re_phase = np.cos(phase) * amplitude  # shape: (coils,)
        im_phase = np.sin(phase) * amplitude  # shape: (coils,)

        # 2) Build the small (2,2,coils) array: (reimout, reim, coils)
        #    [ [ re_phase,    -im_phase ],
        #      [ im_phase,     re_phase ] ]
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)  # (2, coils)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)  # (2, coils)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)  # shape (2, 2, coils)

        # 3) Expand/tile along 'hf' dimension => (hf=2, reimout=2, reim=2, coils)
        coeffs = np.expand_dims(coeffs, axis=0)  # now (1, 2, 2, coils)
        coeffs = np.tile(coeffs, (2, 1, 1, 1))  # now (2, 2, 2, coils)

        # 4) Manually contract via JIT-ed function (instead of np.einsum)
        field_shift = contract_field_coeffs(field, coeffs)
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

        field_shifted = np.zeros(field.shape[:-1])  # coil dimension removed
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

    # einops
    # Total simulation time for 'children_0_tubes_2_id_19969': 3.012579 seconds
    # Average simulation time per run: 0.060252 seconds

    # numpy
    # Total simulation time for 'children_0_tubes_2_id_19969': 2.936453 seconds
    # Average simulation time per run: 0.058729 seconds

    # numba
    # Total simulation time for 'children_0_tubes_2_id_19969': 2.568560 seconds
    # Average simulation time per run: 0.051371 seconds
