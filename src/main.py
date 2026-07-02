"""
Metastasis-Tracker-AI: Main Driver & Execution Pipeline
Filename: main.py
Author: Archival Software Component

This script coordinates the ingestion of multi-modal datasets, sets up spatial
affine transformation matrices, allocates GPU memory pools, and outputs geometric 
density metrics for downstream tracking verification.
"""

import os
import sys
import numpy as np

# Import the core modules built in the previous steps
# Ensure these files are located in the same directory or within your PYTHONPATH
try:
    from core_registration_engine import MultiModalRegistrationEngine
    from cuda_driver import CUDAVoxelAllocator, pycuda_available
except ImportError:
    # Fallback to structural simulation placeholders if executed standalone
    print("[WARN] External tracking engine components not detected in path. Initializing inline code mocks.")
    pycuda_available = False

def generate_mock_datasets(dimensions: tuple) -> tuple:
    """
    Synthesizes mock multi-modal datasets to test structural spatial transformations.
    
    :param dimensions: Tuple representing the target (X, Y, Z) array bounds.
    """
    print(f"[INFO] Synthesizing simulated 3D volumetric medical datasets: {dimensions}")
    # 1. Simulate 2D projection array (X-ray frame)
    xray_projection = np.random.rand(dimensions[0], dimensions[1]).astype(np.float32) * 100.0
    
    # 2. Simulate 3D matrix volume (MRI array) with an encapsulated high-density central core
    mri_volume = np.random.rand(*dimensions).astype(np.float32) * 50.0
    
    # Inject a hyperintense spherical shell mimicking the lipid-chitin vesicle matrix signature
    cx, cy, cz = dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2
    radius = min(dimensions) // 4
    
    z, y, x = np.ogrid[:dimensions[0], :dimensions[1], :dimensions[2]]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    
    # Saturate the signal values inside the simulated mask zone
    vesicle_mask = (dist_from_center >= radius - 2) & (dist_from_center <= radius + 2)
    mri_volume[vesicle_mask] = 400.0  # Hyperintense boundary mapping marker
    
    return xray_projection, mri_volume, vesicle_mask


# Core Component Module Ingestions
from config_loader import ConfigurationLoader
from core_registration_engine import MultiModalRegistrationEngine
from cuda_driver import CUDAVoxelAllocator, pycuda_available
from dicom_series_aggregator import DICOMSeriesAggregator

def setup_runtime_directories() -> str:
    """
    Ensures baseline project directories exist and creates mock dataset slices
    if no raw scanning folder path is provided.
    """
    mock_dir = "dicom_input_series"
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir, exist_ok=True)
        print(f"[INFO] Creating mock input staging folder structure at: ./{mock_dir}")
        
        # Synthesize multiple small placeholder DICOM slice objects for standalone verification
        # Imported inline to prevent adding dependencies if raw data is already present
        from test_series_aggregator import write_mock_slice
        write_mock_slice(mock_dir, "slice_z30.dcm", z_position=3.0, pixel_value=150)
        write_mock_slice(mock_dir, "slice_z00.dcm", z_position=0.0, pixel_value=120)
        write_mock_slice(mock_dir, "slice_z15.dcm", z_position=1.5, pixel_value=135)
        print(f"[SUCCESS] Synthesized baseline verification slice layers inside: ./{mock_dir}")
        
    return mock_dir

# Insert this directly into the main() block of your main.py file:

from config_loader import ConfigurationLoader

def main():
    print("==================================================================")
    print("      METASTASIS-TRACKER-AI: MASTER 3D PIPELINE INITIALIZATION    ")
    print("==================================================================")

    # Add this code block inside your main() execution pool within main.py:

from anisotropic_filter import AnisotropicFilterEngine

# ... [Previous Ingestion and Warping Processes Complete] ...

print("[INFO] Executing edge-preserving anisotropic spatial filtering...")
filter_engine = AnisotropicFilterEngine(volume_shape)
filtered_volume = filter_engine.execute_filter(
    warped_volume, 
    iterations=3, 
    lambda_val=0.15, 
    k_val=25.0
)
print("[SUCCESS] Spatial matrix noise scrubbed cleanly from 3D array grids.")

# Allocate GPU pools using filtered arrays
print("[INFO] Setting up PyCUDA parallel acceleration context blocks...")
allocator = CUDAVoxelAllocator(volume_shape)
device_ready = allocator.allocate_device_buffers()
if device_ready:
    allocator.transfer_to_device(filtered_volume)

    # 1. Load and parse external vendor scanner alignment configurations
    config = ConfigurationLoader("config_matrices.json")
    if not config.load_and_validate_matrices():
        print("[CRITICAL ERROR] Hardware spatial tuning assets missing. Halting execution.")
        return
        
    # Extract live configuration coefficients
    xray_trans, xray_scale, hu_offset = config.extract_carestream_affine_vectors()
    mri_phase, adc_scale, spair_settings = config.extract_ge_mri_profiles()
    global_constraints = config.get_global_pipeline_constraints()

    # 2. Initialize target input data directories
    input_directory = setup_runtime_directories()

    # 3. Aggregate serial cross-sections into a unified 3D matrix volume array
    print(f"[INFO] Initializing serial file compilation from folder: ./{input_directory}")
    aggregator = DICOMSeriesAggregator(input_directory)
    
    try:
        compiled_volume = aggregator.compile_3d_volume()
    except Exception as err:
        print(f"[CRITICAL ERROR] Data matrix tracking compilation breakdown: {err}")
        return

    # Extract parsed dimension arrays to configure downstream grids
    volume_shape = aggregator.spatial_metadata["volume_shape"]

    # 4. Instantiate the registration engine using real aggregated coordinates
    # For testing, we mock a 2D frame matching the spatial width of the 3D grid
    mock_2d_projection = np.zeros((volume_shape[1], volume_shape[2]), dtype=np.float32)
    engine = MultiModalRegistrationEngine(mock_2d_projection, compiled_volume)
    
    # Compute the 4x4 matrix mapping configuration from hardware profiles
    print("[INFO] Computing 4x4 coordinate warp matrix values...")
    affine_matrix = engine.configure_affine_parameters(
        scale=tuple(xray_scale),
        rotation=(0.0, 0.0, float(mri_phase[2])),  # Rotate using config parameters
        translation=tuple(xray_trans)
    )
    
    print("[INFO] Processing spatial grid transformations over aggregated series...")
    warped_volume = engine.execute_volume_warp()

    # 5. Lock data array structures into parallel GPU device memory layout
    print("[INFO] Setting up PyCUDA parallel acceleration context blocks...")
    # Seed allocator using the true Z-axis size calculated by the aggregator
    allocator = CUDAVoxelAllocator(volume_shape[0])
    
    device_ready = allocator.allocate_device_buffers()
    if device_ready:
        print(f"[INFO] Copying aggregated 3D data block ({volume_shape}) to device...")
        allocator.transfer_to_device(warped_volume)
        print("[SUCCESS] Live structural spatial arrays locked inside hardware thread blocks.")
    else:
        print("[WARN] GPU core initialization skipped. Vectorized calculations fallback to CPU space.")

    # 6. Evaluate baseline feature data parameters
    # Construct a validation target mask covering the whole input tracking sequence
    validation_mask = warped_volume > float(global_constraints.get("chitin_attenuation_bounds", [140.0])[0])
    metrics = engine.calculate_attenuation_vectors(validation_mask)

    print("==================================================================")
    print("                FINAL PIPELINE TRACKING VALIDATION                ")
    print("==================================================================")
    print(f"Total Voxel Clusters Active : {metrics['total_voxels']} voxels")
    print(f"Mean Aggregated Core Value   : {metrics['mean_density']:.4f} normalized units")
    print(f"Peak Matrix Intensity Vector : {metrics['peak_density']:.4f} Tesla/Hounsfield")
    print("==================================================================")
    print("[SUCCESS] Data aggregation cycle completed cleanly. Code structures ready for Git tracking.")

if __name__ == "__main__":
    main()
