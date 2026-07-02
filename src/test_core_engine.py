"""
Metastasis-Tracker-AI: Automated Coordinate & Metric Verification Suite
Filename: test_core_engine.py
Author: Archival Test Component

This testing module leverages pytest to systematically evaluate the precision
of 4x4 affine transform matrices, verify voxel grid dimensions, and ensure 
attenuation vectors match the target thresholds.
"""

import os
import pytest
import numpy as np

# Core software imports to satisfy the execution framework validation
from core_registration_engine import MultiModalRegistrationEngine
from config_loader import ConfigurationLoader

# 1. Verification Fixtures
@pytest.fixture
def mock_volume_dimensions():
    """Returns a standardized spatial coordinate grid dimension tuple."""
    return (32, 32, 32)

@pytest.fixture
def initialized_registration_engine(mock_volume_dimensions):
    """Provides a baseline engine instance loaded with mock scanning arrays."""
    dims = mock_volume_dimensions
    xray_mock = np.zeros((dims, dims), dtype=np.float32)
    mri_mock = np.zeros(dims, dtype=np.float32)
    
    # Inject a known hyperintense verification locus at the geometric center
    mid = dims // 2
    mri_mock[mid, mid, mid] = 500.0
    
    return MultiModalRegistrationEngine(xray_mock, mri_mock)

@pytest.fixture
def local_config_loader():
    """Provides a functional calibration profile ingestion instance."""
    # Instantiates the loader using the standardized project JSON config path
    return ConfigurationLoader("config_matrices.json")


# 2. Functional Code Test Blocks
def test_identity_matrix_preserves_spatial_coordinates(initialized_registration_engine):
    """
    Verifies that configuring a strict 1:1 identity transform preserves the 
    baseline spatial coordinates without creating skew or distortion arrays.
    """
    engine = initialized_registration_engine
    
    # Configure an exact identity setup: scaling = 1, rotation/translation = 0
    affine_transform = engine.configure_affine_parameters(
        scale=(1.0, 1.0, 1.0),
        rotation=(0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 0.0)
    )
    
    # Assert matrix structural footprint equates to a standard 4x4 identity map
    expected_identity = np.eye(4, dtype=np.float32)
    np.testing.assert_array_almost_equal(affine_transform, expected_identity, decimal=5)


def test_translation_components_map_accurately(initialized_registration_engine):
    """
    Ensures translation offset variables are correctly mapped into the 
    final index column ([0:3], 3) of the composed affine transform matrix.
    """
    engine = initialized_registration_engine
    target_tx, target_ty, target_tz = 5.5, -3.2, 0.0
    
    affine_transform = engine.configure_affine_parameters(
        scale=(1.0, 1.0, 1.0),
        rotation=(0.0, 0.0, 0.0),
        translation=(target_tx, target_ty, target_tz)
    )
    
    # Assert explicit index extraction matches the target offsets
    assert affine_transform[0, 3] == pytest.approx(target_tx)
    assert affine_transform[1, 3] == pytest.approx(target_ty)
    assert affine_transform[2, 3] == pytest.approx(target_tz)


def test_volumetric_attenuation_metrics_parsing(initialized_registration_engine, mock_volume_dimensions):
    """
    Validates that the feature extraction loop parses Hounsfield/Tesla values
    accurately inside a designated tracking array mask.
    """
    engine = initialized_registration_engine
    dims = mock_volume_dimensions
    
    # Construct a verification mask covering only the exact center voxel coordinates
    mask = np.zeros(dims, dtype=bool)
    mid = dims // 2
    mask[mid, mid, mid] = True
    
    metrics = engine.calculate_attenuation_vectors(mask)
    
    # Assert parsing outputs match the injected central peak metrics
    assert metrics["total_voxels"] == 1
    assert metrics["peak_density"] == pytest.approx(500.0)
    assert metrics["mean_density"] == pytest.approx(500.0)


def test_empty_mask_handling_yields_safe_fallbacks(initialized_registration_engine, mock_volume_dimensions):
    """
    Ensures that passing an empty boolean matrix array doesn't crash the loop,
    returning zero-value metrics gracefully.
    """
    engine = initialized_registration_engine
    empty_mask = np.zeros(mock_volume_dimensions, dtype=bool)
    
    metrics = engine.calculate_attenuation_vectors(empty_mask)
    
    assert metrics["total_voxels"] == 0
    assert metrics["mean_density"] == 0.0
    assert metrics["peak_density"] == 0.0


def test_config_loader_syntax_and_file_presence(local_config_loader):
    """
    Verifies the configuration file loader checks file presence and pulls key sections.
    Note: Requires config_matrices.json to be generated in the test path context.
    """
    loader = local_config_loader
    
    # Check execution profile behavior across file validation checks
    if os.path.exists(loader.json_path):
        success = loader.load_and_validate_matrices()
        assert success is True
        
        # Verify global constraint structures contain values
        constraints = loader.get_global_pipeline_constraints()
        assert isinstance(constraints, dict)
        assert "target_voxel_resolution_mm" in constraints
    else:
        # If running in an empty workspace environment without the config file, verify safe fallback exit values
        success = loader.load_and_validate_matrices()
        assert success is False
