"""
Metastasis-Tracker-AI: Series Aggregator Verification Suite
Filename: test_series_aggregator.py
"""

import os
import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from dicom_series_aggregator import DICOMSeriesAggregator

def write_mock_slice(directory, filename, z_position, pixel_value):
    """Utility generator to save a customized structural DICOM slice blueprint onto disk."""
    file_path = os.path.join(directory, filename)
    
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
    
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.5
    ds.ImagePositionPatient = [0.0, 0.0, float(z_position)]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    
    # 2x2 test matrix track filled with uniform density tokens
    mock_matrix = np.full((2, 2), pixel_value, dtype=np.uint16)
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = mock_matrix.tobytes()
    
    ds.save_as(file_path, write_like_original=False)

@pytest.fixture
def scrambled_dicom_directory(tmp_path):
    """
    Constructs a temporary workspace containing 3 custom DICOM slices generated 
    completely out of spatial sequence order to test the pipeline sorting matrices.
    """
    dir_path = tmp_path / "dicom_series_stack"
    os.makedirs(dir_path, exist_ok=True)
    
    # Write files with names that do NOT match their physical order in 3D coordinate space
    # Slice 3: Located at Z=3.0, Token value=30
    write_mock_slice(str(dir_path), "file_c.dcm", z_position=3.0, pixel_value=30)
    # Slice 1: Located at Z=0.0, Token value=10
    write_mock_slice(str(dir_path), "file_a.dcm", z_position=0.0, pixel_value=10)
    # Slice 2: Located at Z=1.5, Token value=20
    write_mock_slice(str(dir_path), "file_b.dcm", z_position=1.5, pixel_value=20)
    
    return str(dir_path)

def test_series_sorting_and_matrix_aggregation(scrambled_dicom_directory):
    """
    Verifies the pipeline sorts the files into physical sequential order (10 -> 20 -> 30)
    instead of relying on alphabetical names, and aggregates them into a valid 3D array block.
    """
    aggregator = DICOMSeriesAggregator(scrambled_dicom_directory)
    volume_matrix = aggregator.compile_3d_volume()
    
    # Assert dimension matrix tracks: 3 stacked slices of 2x2 grids -> (3, 2, 2)
    assert volume_matrix.shape == (3, 2, 2)
    
    # Verify accurate structural sorting outputs via density check vectors
    # Slice index 0 must evaluate to the true baseline spatial floor (Z=0.0 -> token 10)
    assert np.all(volume_matrix[0, :, :] == 10.0)
    # Slice index 1 must evaluate to the middle slice layer (Z=1.5 -> token 20)
    assert np.all(volume_matrix[1, :, :] == 20.0)
    # Slice index 2 must evaluate to the upper terminal ceiling layer (Z=3.0 -> token 30)
    assert np.all(volume_matrix[2, :, :] == 30.0)
    
    # Verify computed inter-slice tracking distances match (1.5 - 0.0) -> 1.5mm
    assert aggregator.spatial_metadata["dz"] == pytest.approx(1.5)
