
import numpy.typing as npt
import numpy as np
import h5py
import os
import einops
import torch as t
from typing import Tuple
from .dataclasses import SimulationRawData, SimulationData, CoilConfig



class Simulation:
    def __init__(self, 
                 path: str,
                 coil_path: str = "data/antenna/antenna.h5"):
        self.path = path
        self.coil_path = coil_path
        
        self.simulation_raw_data = self._load_raw_simulation_data()
        
    def _load_raw_simulation_data(self) -> SimulationRawData:
        # Load raw simulation data from path
        
        def read_field() -> Tuple[t.float32, t.float32]:
            with h5py.File(self.path) as f:
                re_efield, im_efield = t.from_numpy(f["efield"]["re"]), t.from_numpy(f["efield"]["im"])
                re_hfield, im_hfield = t.from_numpy(f["hfield"]["re"]), t.from_numpy(f["hfield"]["im"]                                                                       )
                field = t.stack([t.stack([re_efield, im_efield], axis=0), t.stack([re_hfield, im_hfield], axis=0)], axis=0)
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
                coil = t.from_numpy(f["masks"][:])
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
                     field: t.float32,
                     phase: t.float32,
                     amplitude: t.float32) -> t.float32:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils.
        """
        re_phase = t.cos(phase) * amplitude
        im_phase = t.sin(phase) * amplitude
        coeffs_real = t.stack((re_phase, -im_phase), axis=0)
        coeffs_im = t.stack((im_phase, re_phase), axis=0)
        coeffs = t.stack((coeffs_real, coeffs_im), axis=0).double()
        field = field.double()
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(field, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift

    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:
        # TODO only shift the subject (using mask??)
        # TODO: get the mask (subject) from the data
        # TODO: Then only use the relevant data for field shift
        # TODO: Process the masked part in parallel with multiprocessing ?
        coil_config_phase = t.from_numpy(coil_config.phase)
        coil_config_amplitude = t.from_numpy(coil_config.amplitude)
        if type(self.simulation_raw_data.field) == np.float32:
            raw_data_field = t.from_numpy(self.simulation_raw_data.field)
        else:
            raw_data_field = self.simulation_raw_data.field
        field_shifted = self._shift_field(raw_data_field, coil_config_phase, coil_config_amplitude)
        
        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data
    
    def __call__(self, coil_config: CoilConfig) -> SimulationData:

        return self.phase_shift(coil_config)
