"""
Metastasis-Tracker-AI: Dynamic Configuration & Matrix Loader
Filename: config_loader.py
Author: Archival Software Component

This script dynamically ingests, validates, and initializes calibration 
matrices and pulse sequence profiles from Carestream and GE Medical devices
at software initialization.
"""

import os
import json
import numpy as np

class ConfigurationLoader:
    def __init__(self, json_path: str = "config_matrices.json"):
        """
        Initializes the dynamic matrix ingestion interface.
        :param json_path: The file path to the target JSON configuration dataset.
        """
        self.json_path = json_path
        self.config_data = {}
        
    def load_and_validate_matrices(self) -> bool:
        """
        Loads the calibration configuration profile from disk and parses 
        the nested transformation layers. Returns True if structurally valid.
        """
        if not os.path.exists(self.json_path):
            print(f"[CRITICAL ERROR] Configuration file missing at: {self.json_path}")
            return False
            
        try:
            with open(self.json_path, 'r') as file:
                self.config_data = json.load(file)
            print(f"[SUCCESS] Config matrices loaded from {self.json_path} cleanly.")
            return self.verify_structural_integrity()
        except json.JSONDecodeError as err:
            print(f"[CRITICAL ERROR] Invalid JSON compilation syntax detected: {err}")
            return False

    def verify_structural_integrity(self) -> bool:
        """
        Performs explicit schema check loops to ensure necessary affine metrics exist.
        """
        required_keys = ["scanner_profiles", "pipeline_global_constraints"]
        for key in required_keys:
            if key not in self.config_data:
                print(f"[ERROR] Required tracking section key missing: '{key}'")
                return False
        return True

    def extract_carestream_affine_vectors(self) -> tuple:
        """
        Parses and returns Carestream configuration parameters converted to 
        NumPy-ready numeric vector components.
        """
        try:
            profile = self.config_data["scanner_profiles"]["carestream_xray_default"]
            translation = np.array(profile["calibration_matrices"]["affine_translation"], dtype=np.float32)
            scaling = np.array(profile["calibration_matrices"]["spatial_scaling"], dtype=np.float32)
            hu_offset = float(profile["calibration_matrices"]["hounsfield_offset"])
            return translation, scaling, hu_offset
        except KeyError as err:
            print(f"[ERROR] Carestream dictionary extraction anomaly at key: {err}")
            return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), 0.0

    def extract_ge_mri_profiles(self) -> tuple:
        """
        Parses and extracts GE 3T magnetic configuration arrays, Dixon phase 
        offsets, and ADC normalization coefficients.
        """
        try:
            profile = self.config_data["scanner_profiles"]["ge_mri_3t_default"]
            phase_offset = np.array(profile["calibration_matrices"]["t2_dixon_phase_offset"], dtype=np.float32)
            adc_multiplier = float(profile["calibration_matrices"]["adc_normalization_multiplier"])
            spair_settings = profile["pulse_sequence_profiles"]["spair_fat_sat"]
            return phase_offset, adc_multiplier, spair_settings
        except KeyError as err:
            print(f"[ERROR] GE Medical extraction anomaly at missing profile key: {err}")
            return np.zeros(3, dtype=np.float32), 1e-06, {}

    def get_global_pipeline_constraints(self) -> dict:
        """
        Returns the unified tracking bounds for processing restricted diffusion matrices.
        """
        return self.config_data.get("pipeline_global_constraints", {})

# Standalone block verification
if __name__ == "__main__":
    loader = ConfigurationLoader()
    if loader.load_and_validate_matrices():
        trans, scale, hu = loader.extract_carestream_affine_vectors()
        phase, adc, pulse = loader.extract_ge_mri_profiles()
        print(f"[TEST CHECK] Carestream Alignment Translation Layer : {trans}")
        print(f"[TEST CHECK] GE Medical 3T Normalization Factor     : {adc}")
        print(f"[TEST CHECK] SPAIR Pulse Sequence Repetition Time  : {pulse.get('repetition_time_ms')} ms")
