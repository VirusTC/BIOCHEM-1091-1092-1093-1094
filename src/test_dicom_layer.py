"""
Metastasis-Tracker-AI: DICOM Processing Verification Module
Filename: test_dicom_layer.py
"""

import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from dicom_processing_layer import DICOMProcessingLayer

@pytest.fixture
def synthesized_dicom_file(tmp_path):
    """
    Synthesizes a true mock .dcm file structure on disk with custom 
    rescale factors and spatial metrics for isolated pipeline validation.
    """
    file_path = tmp_path / "mock_scan.dcm"
    
    # 1. Establish the File Meta Information header
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 0
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage class
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2' # Implicit VR Little Endian
    file_meta.ImplementationClassUID = '1.2.3.4'
    
    # 2. Build the primary dataset properties
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    ds.Manufacturer = "Carestream Medical Simulation"
    ds.Modality = "CT"
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    
    # Configure known scaling transformation modifiers
    ds.RescaleSlope = 2.0
    ds.RescaleIntercept = -1000.0
    
    # Create simple 4x4 matrix data
    mock_matrix = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    ds.Rows = mock_matrix.shape[0]
    ds.Columns = mock_matrix.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = mock_matrix.tobytes()
    
    ds.save_as(str(file_path), write_like_original=False)
    return str(file_path)

def test_dicom_metadata_ingestion_and_normalization(synthesized_dicom_file):
    """
    Verifies the processing layer parses the file, checks the hardware metrics, 
    and applies pixel linear transformations mathematically.
    """
    processor = DICOMProcessingLayer(synthesized_dicom_file)
    
    # Validate successful file read phase
    assert processor.ingest_and_parse_metadata() is True
    
    # Process array transformations
    normalized_array = processor.normalize_pixel_intensities()
    
    # Check linear calculation mapping output accuracy: (Raw * 2.0) - 1000.0
    # Expected result array values: [(10*2)-1000, (20*2)-1000] -> [-980.0, -960.0]
    assert normalized_array[0, 0] == -980.0
    assert normalized_array[0, 1] == -960.0
    
    # Validate spatial metrics parsing blocks
    spatial = processor.extract_spatial_resolution_metrics()
    assert spatial["dx"] == 0.5
    assert spatial["dz"] == 1.0
