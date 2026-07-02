"""
Metastasis-Tracker-AI: Multi-Modal Spatial Co-Registration Engine
Filename: core_registration_engine.py
Author: Archival Software Component

This script establishes the baseline computational models for coregistering 
2D projection matrices (X-ray) into a unified 3D voxel coordinate space (MRI)
using affine scaling, rotation, and translation transformations.
"""

import numpy as np
from scipy.ndimage import affine_transform

class MultiModalRegistrationEngine:
    def __init__(self, xray_matrix: np.ndarray, mri_volume: np.ndarray):
        """
        Initializes the co-registration grid arrays.
        
        :param xray_matrix: 2D numpy array representing projection radiograph density.
        :param mri_volume: 3D numpy array representing structural MRI voxel values.
        """
        self.xray_2d = xray_matrix.astype(np.float32)
        self.mri_3d = mri_volume.astype(np.float32)
        
        # Identity matrix for baseline 3D transformation [4x4]
        self.transformation_matrix = np.eye(4, dtype=np.float32)

    def configure_affine_parameters(self, 
                                    scale: tuple = (1.0, 1.0, 1.0), 
                                    rotation: tuple = (0.0, 0.0, 0.0), 
                                    translation: tuple = (0.0, 0.0, 0.0)):
        """
        Calculates the affine transformation matrix to align spatial coordinates.
        Angles are processed in radians.
        """
        sx, sy, sz = scale
        tx, ty, tz = translation
        rx, ry, rz = rotation

        # Compute trigonometric rotation values
        cx, sx_rad = np.cos(rx), np.sin(rx)
        cy, sy_rad = np.cos(ry), np.sin(ry)
        cz, sz_rad = np.cos(rz), np.sin(rz)

        # 1. Scaling Matrix
        S = np.diag([sx, sy, sz, 1.0])

        # 2. Rotation Matrices (X, Y, Z conventions)
        R_x = np.array([[1, 0, 0, 0],
                        [0, cx, -sx_rad, 0],
                        [0, sx_rad, cx, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

        R_y = np.array([[cy, 0, sy_rad, 0],
,
                        [-sy_rad, 0, cy, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

        R_z = np.array([[cz, -sz_rad, 0, 0],
                        [sz_rad, cz, 0, 0],
,
                        [0, 0, 0, 1]], dtype=np.float32)

        # 3. Translation Matrix
        T = np.array([[1, 0, 0, tx],
                        [0, 1, 0, ty],
                        [0, 0, 1, tz],
                        [0, 0, 0, 1]], dtype=np.float32)

        # Compose full affine transformation loop: T * R_z * R_y * R_x * S
        self.transformation_matrix = T @ R_z @ R_y @ R_x @ S
        return self.transformation_matrix

    def execute_volume_warp(self) -> np.ndarray:
        """
        Applies the computed affine transformation matrix directly to the 3D volume array.
        """
        # Extract the upper 3x3 rotation/scaling matrix and the translation vector
        matrix_3x3 = self.transformation_matrix[:3, :3]
        offset = self.transformation_matrix[:3, 3]

        # Invert the mapping parameters for pull-back voxel sampling
        inv_matrix = np.linalg.inv(matrix_3x3)
        inv_offset = -inv_matrix @ offset

        warped_volume = affine_transform(
            self.mri_3d, 
            matrix=inv_matrix, 
            offset=inv_offset, 
            order=1,  # Bilinear interpolation profile
            mode='constant', 
            cval=0.0
        )
        return warped_volume

    def calculate_attenuation_vectors(self, mask: np.ndarray) -> dict:
        """
        Computes localized geometric density parameters inside a specific segmented mask.
        
        :param mask: 3D boolean array isolating target voxel coordinates.
        """
        if not np.any(mask):
            return {"mean_density": 0.0, "peak_density": 0.0, "total_voxels": 0}

        target_voxels = self.mri_3d[mask]
        
        metrics = {
            "mean_density": float(np.mean(target_voxels)),
            "peak_density": float(np.max(target_voxels)),
            "spatial_variance": float(np.var(target_voxels)),
            "total_voxels": int(np.sum(mask))
        }
        return metrics
