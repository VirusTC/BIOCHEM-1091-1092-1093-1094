"""
Metastasis-Tracker-AI: Programmatic DICOM Ingestion & Normalization Engine
Filename: dicom_processing_layer.py
Author: Archival Software Component

This script extends the spatial tracking architecture by directly ingesting raw
.dcm files, extracting geometric slice spacing matrices, and normalizing array
pixel intensities using embedded hardware calibration scaling tags.
"""

import os
import numpy as np
import pydicom

class DICOMProcessingLayer:
    def __init__(self, file_path: str):
        """
        Initializes the raw DICOM ingestion matrix block.
        :param file_path: Path to the target .dcm file structure.
        """
        self.file_path = file_path
        self.dataset = None
        self.pixel_array = None
        
    def ingest_and_parse_metadata(self) -> bool:
        """
        Loads the target DICOM file from disk using the pydicom library.
        Parses structural header elements into memory.
        """
        if not os.path.exists(self.file_path):
            print(f"[CRITICAL ERROR] DICOM file missing at path: {self.file_path}")
            return False
            
        try:
            # Read the dataset structure including meta information header tags
            self.dataset = pydicom.dcmread(self.file_path)
            print(f"[SUCCESS] Parsed DICOM file: {os.path.basename(self.file_path)}")
            print(f"  [HARDWARE INFO] Vendor: {self.dataset.get('Manufacturer', 'UNKNOWN')}")
            print(f"  [MODALITY TYPE] Mode  : {self.dataset.get('Modality', 'UNKNOWN')}")
            return True
        except Exception as err:
            print(f"[CRITICAL ERROR] Failed parsing DICOM structural metadata layers: {err}")
            return False

    def normalize_pixel_intensities(self) -> np.ndarray:
        """
        Extracts raw image metrics and transforms array voxel weights into calibrated
        units using Rescale Slope and Intercept attributes.
        """
        if self.dataset is None:
            raise ValueError("[ERROR] Dataset not loaded. Call ingest_and_parse_metadata() first.")
            
        # Extract the raw integer pixel matrix from file arrays
        raw_pixels = self.dataset.pixel_array.astype(np.float32)
        
        # Pull embedded rescale tags (standard defaults applied if absent)
        # Final Value = (Raw Pixel * Slope) + Intercept
        rescale_slope = float(self.dataset.get('RescaleSlope', 1.0))
        rescale_intercept = float(self.dataset.get('RescaleIntercept', 0.0))
        
        print(f"[INFO] Applying hardware calibration tags - Slope: {rescale_slope}, Intercept: {rescale_intercept}")
        self.pixel_array = (raw_pixels * rescale_slope) + rescale_intercept
        return self.pixel_array

    def extract_spatial_resolution_metrics(self) -> dict:
        """
        Parses coordinate delta variables (PixelSpacing and SliceThickness) 
        required for true 3D spatial registration inside tracker modules.
        """
        if self.dataset is None:
            return {"dx": 1.0, "dy": 1.0, "dz": 1.0, "matrix_dims": (0, 0)}
            
        # Pixel Spacing tags provide millimeter offsets across the [X, Y] plan grid
        pixel_spacing = self.dataset.get('PixelSpacing', [1.0, 1.0])
        # Slice Thickness maps physical gap thickness along the Z dimensional coordinate
        slice_thickness = self.dataset.get('SliceThickness', 1.0)
        
        spatial_metrics = {
            "dx": float(pixel_spacing[0]),
            "dy": float(pixel_spacing[1]),
            "dz": float(slice_thickness),
            "matrix_dims": self.dataset.pixel_array.shape
        }
        
        print(f"[SUCCESS] Ingested Spatial Spacing Vectors (mm): [{spatial_metrics['dx']:.4f}, "
              f"{spatial_metrics['dy']:.4f}, {spatial_metrics['dz']:.4f}]")
        return spatial_metrics
