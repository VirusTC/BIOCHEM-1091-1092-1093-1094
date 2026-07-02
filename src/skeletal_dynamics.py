"""
Metastasis-Tracker-AI: Section 10 - Multi-Planar Flow Equations
Module: src/skeletal_dynamics.py
"""

import numpy as np

class MultiPlanarSkeletalDynamics:
    def __init__(self, pelvic_voxel_grid: np.ndarray, voxel_spacing_mm: tuple = (0.5, 0.5, 1.0)):
        """Initializes the multi-planar bone fluid transit parameters."""
        self.grid = pelvic_voxel_grid.astype(np.float32)
        self.dx, self.dy, self.dz = voxel_spacing_mm
        self.depth, self.height, self.width = self.grid.shape

    def compute_multi_planar_gradients(self) -> tuple:
        """Calculates independent spatial gradient tensors across orthogonal planes."""
        grad_z = np.gradient(self.grid, axis=0) / self.dz
        grad_y = np.gradient(self.grid, axis=1) / self.dy
        grad_x = np.gradient(self.grid, axis=2) / self.dx
        return grad_z, grad_y, grad_x

    def evaluate_realtime_density_shifts(self, baseline_grid: np.ndarray, diffusion_coefficient: float = 0.12) -> dict:
        """Computes time-dependent density variations (dH/dt) and 3D Laplacian flux values."""
        if baseline_grid.shape != self.grid.shape:
            raise ValueError("Dimensional mismatch across compared longitudinal 3D voxel arrays.")

        temporal_shift_matrix = self.grid - baseline_grid.astype(np.float32)
        gz, gy, gx = self.compute_multi_planar_gradients()
        
        laplacian_z = np.gradient(gz, axis=0) / self.dz
        laplacian_y = np.gradient(gy, axis=1) / self.dy
        laplacian_x = np.gradient(gx, axis=2) / self.dx
        total_laplacian = laplacian_z + laplacian_y + laplacian_x
        
        calcium_depletion_mask = (temporal_shift_matrix < -25.0) & (self.grid < 200.0)
        
        return {
            "mean_global_shift": float(np.mean(temporal_shift_matrix)),
            "peak_resorption_velocity": float(np.min(temporal_shift_matrix)),
            "marrow_depletion_voxels": int(np.sum(calcium_depletion_mask)),
            "directional_vectors": {
                "sagittal_x_leach": float(np.sum(np.abs(gx)[calcium_depletion_mask])),
                "coronal_y_leach": float(np.sum(np.abs(gy)[calcium_depletion_mask])),
                "axial_z_leach": float(np.sum(np.abs(gz)[calcium_depletion_mask]))
            },
            "net_skeletal_flux": float(np.sum(diffusion_coefficient * total_laplacian))
        }
