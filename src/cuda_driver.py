"""
Metastasis-Tracker-AI: CUDA Voxel Allocation Driver
Filename: cuda_driver.py
"""

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    pycuda_available = True
except ImportError:
    pycuda_available = False

class CUDAVoxelAllocator:
    def __init__(self, volume_dimensions: tuple):
        """
        Allocates continuous GPU memory for 3D coordinate arrays.
        :param volume_dimensions: Tuple representing (X, Y, Z) voxel array shape.
        """
        self.dims = volume_dimensions
        self.total_voxels = volume_dimensions[0] * volume_dimensions[1] * volume_dimensions[2]
        self.bytes_allocated = self.total_voxels * 4  # 32-bit float profile
        
        self.d_voxel_grid = None
        self.d_output_image = None

    def allocate_device_buffers(self):
        """
        Executes explicit raw pointers mapping on the device.
        """
        if not pycuda_available:
            print("[ERROR] PyCUDA library not detected. Running simulation block.")
            return False
            
        self.d_voxel_grid = cuda.mem_alloc(self.bytes_allocated)
        # Allocate flat output array for standard screen projections
        self.d_output_image = cuda.mem_alloc(self.dims[0] * self.dims[1] * 4)
        return True

    def transfer_to_device(self, host_volume: np.ndarray):
        """
        Copies raw host tissue arrays into GPU memory.
        """
        if pycuda_available and self.d_voxel_grid:
            flat_data = host_volume.astype(np.float32).flatten()
            cuda.memcpy_htod(self.d_voxel_grid, flat_data)
            return True
        return False
