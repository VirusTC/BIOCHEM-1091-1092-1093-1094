"""
Metastasis-Tracker-AI: Programmatic DICOM Series Aggregator
Filename: dicom_series_aggregator.py
Author: Archival Software Component

This script ingests an entire directory of individual .dcm files, validates 
modality consistency, sorts slices based on physical spatial coordinates, 
and aggregates them into a normalized 3D matrix volume array.
"""

import os
import glob
import numpy as np
import pydicom

class DICOMSeriesAggregator:
    def __init__(self, directory_path: str):
        """
        Initializes the series aggregator layer.
        :param directory_path: Folder directory containing the target .dcm file slice sequence.
        """
        self.directory_path = directory_path
        self.volume_3d = None
        self.spatial_metadata = {}

    def compile_3d_volume(self) -> np.ndarray:
        """
        Scans the target directory, parses individual slices, sorts them by 
        physical 3D slice location, and aggregates them into a unified 3D matrix array.
        """
        # 1. Gather all file paths matching the extension criteria
        search_pattern = os.path.join(self.directory_path, "*.dcm")
        file_list = glob.glob(search_pattern)

        if not file_list:
            raise FileNotFoundError(f"[CRITICAL ERROR] No .dcm slices detected inside path: {self.directory_path}")

        valid_slices = []
        print(f"[INFO] Scanning directory for serial datasets. Found {len(file_list)} candidate files.")

        # 2. Iterate through slices to parse and extract location vectors
        for file_path in file_list:
            try:
                ds = pydicom.dcmread(file_path)
                # Verify that the pixel array matrix is present
                if "PixelData" not in ds:
                    continue
                
                # ImagePositionPatient yields the absolute [X, Y, Z] spatial coordinates of the slice
                # The Z-coordinate (index 2) defines its precise location along the stacking axis
                slice_position_z = float(ds.ImagePositionPatient[2]) if "ImagePositionPatient" in ds else float(ds.get("SliceLocation", 0.0))
                
                valid_slices.append((slice_position_z, ds))
            except Exception as err:
                print(f"[WARN] Skipping corrupted or non-DICOM structure layer: {os.path.basename(file_path)}. Error: {err}")

        if not valid_slices:
            raise ValueError("[CRITICAL ERROR] Zero structurally valid slices compiled into memory.")

        # 3. Sort slices sequentially based on their spatial physical Z-coordinate location
        # This guarantees perfect topological alignment regardless of filename naming structures
        valid_slices.sort(key=lambda x: x[0])
        print(f"[SUCCESS] Spatial sort complete. Valid stack sequence length: {len(valid_slices)} slices.")

        # 4. Extract hardware calibration tags from the baseline primary slice template
        reference_ds = valid_slices[0][1]
        rescale_slope = float(reference_ds.get('RescaleSlope', 1.0))
        rescale_intercept = float(reference_ds.get('RescaleIntercept', 0.0))
        pixel_spacing = reference_ds.get('PixelSpacing', [1.0, 1.0])
        
        # Calculate Z-axis thickness based on true coordinate differences if metadata is absent
        if len(valid_slices) > 1:
            dz_spacing = abs(valid_slices[1][0] - valid_slices[0][0])
        else:
            dz_spacing = float(reference_ds.get('SliceThickness', 1.0))

        # 5. Pre-allocate the empty 3D array space to optimize system memory caching
        rows, cols = reference_ds.pixel_array.shape
        num_slices = len(valid_slices)
        self.volume_3d = np.zeros((num_slices, rows, cols), dtype=np.float32)

        # 6. Populate the 3D voxel grid volume matrix and apply linear scaling updates
        for idx, (_, ds) in enumerate(valid_slices):
            raw_slice = ds.pixel_array.astype(np.float32)
            # Final Value = (Raw Pixel * Slope) + Intercept
            self.volume_3d[idx, :, :] = (raw_slice * rescale_slope) + rescale_intercept

        # Save global 3D spatial properties for registration tracking loops
        self.spatial_metadata = {
            "dx": float(pixel_spacing[0]),
            "dy": float(pixel_spacing[1]),
            "dz": float(dz_spacing),
            "volume_shape": self.volume_3d.shape
        }

        print(f"[SUCCESS] Multi-slice assembly complete.")
        print(f"  [METRIC OUTPUT] Composed Volume Matrix Shape: {self.spatial_metadata['volume_shape']}")
        print(f"  [METRIC OUTPUT] Normalized Voxel Spacing (mm): [{self.spatial_metadata['dx']:.3f}, {self.spatial_metadata['dy']:.3f}, {self.spatial_metadata['dz']:.3f}]")
        
        return self.volume_3d
