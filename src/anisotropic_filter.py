"""
Metastasis-Tracker-AI: Parallelized Anisotropic Spatial Filtering Engine
Filename: src/anisotropic_filter.py
"""

import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    pycuda_available = True
except ImportError:
    pycuda_available = False

cuda_kernel_source = """
__global__ void anisotropic_diffusion_3d(
    const float* __restrict__ input_grid, 
    float* __restrict__ output_grid, 
    const int width, const int height, const int depth, 
    const float lambda_val, const float k_val) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    const int slice_stride = width * height;
    const int idx = z * slice_stride + y * width + x;
    const float val = input_grid[idx];
    
    const float n = (y > 0)          ? input_grid[idx - width]        : val;
    const float s = (y < height - 1) ? input_grid[idx + width]        : val;
    const float e = (x < width - 1)  ? input_grid[idx + 1]            : val;
    const float w = (x > 0)          ? input_grid[idx - 1]            : val;
    const float u = (z < depth - 1)  ? input_grid[idx + slice_stride] : val;
    const float d = (z > 0)          ? input_grid[idx - slice_stride] : val;
    
    const float grad_n = n - val;
    const float grad_s = s - val;
    const float grad_e = e - val;
    const float grad_w = w - val;
    const float grad_u = u - val;
    const float grad_d = d - val;
    
    const float k_sq = k_val * k_val;
    const float c_n = __expf(-(grad_n * grad_n) / k_sq);
    const float c_s = __expf(-(grad_s * grad_s) / k_sq);
    const float c_e = __expf(-(grad_e * grad_e) / k_sq);
    const float c_w = __expf(-(grad_w * grad_w) / k_sq);
    const float c_u = __expf(-(grad_u * grad_u) / k_sq);
    const float c_d = __expf(-(grad_d * grad_d) / k_sq);
    
    output_grid[idx] = val + lambda_val * (
        c_n * grad_n + c_s * grad_s + 
        c_e * grad_e + c_w * grad_w + 
        c_u * grad_u + c_d * grad_d
    );
}
"""

class AnisotropicFilterEngine:
    def __init__(self, volume_shape: tuple):
        """Initializes the parallelized 3D mathematical filtering engine."""
        self.shape = volume_shape
        self.depth, self.height, self.width = volume_shape
        
        if pycuda_available:
            self.mod = SourceModule(cuda_kernel_source)
            self.cuda_kernel = self.mod.get_function("anisotropic_diffusion_3d")
        else:
            self.cuda_kernel = None

    def execute_filter(self, input_volume: np.ndarray, iterations: int = 3, lambda_val: float = 0.15, k_val: float = 25.0) -> np.ndarray:
        """Runs the edge-preserving smoothing pass over the input 3D matrix volume."""
        if self.cuda_kernel:
            float_input = input_volume.astype(np.float32)
            h_output = np.zeros_like(float_input)
            d_input = cuda.mem_alloc(float_input.nbytes)
            d_output = cuda.mem_alloc(float_input.nbytes)
            
            cuda.memcpy_htod(d_input, float_input)
            block_dims = (8, 8, 4)
            grid_dims = (
                int(np.ceil(self.width / block_dims[0])),
                int(np.ceil(self.height / block_dims[1])),
                int(np.ceil(self.depth / block_dims[2]))
            )
            
            for _ in range(iterations):
                self.cuda_kernel(
                    d_input, d_output,
                    np.int32(self.width), np.int32(self.height), np.int32(self.depth),
                    np.float32(lambda_val), np.float32(k_val),
                    block=block_dims, grid=grid_dims
                )
                cuda.memcpy_dtod(d_input, d_output, float_input.nbytes)
                
            cuda.memcpy_dtoh(h_output, d_output)
            d_input.free()
            d_output.free()
            return h_output
            
        return self._execute_cpu_vectorized(input_volume, iterations, lambda_val, k_val)

    def _execute_cpu_vectorized(self, input_volume: np.ndarray, iterations: int, lambda_val: float, k_val: float) -> np.ndarray:
        """Vectorized execution fallback utilizing high-speed NumPy slice metrics."""
        current_volume = input_volume.astype(np.float32)
        k_sq = k_val * k_val
        
        for _ in range(iterations):
            grad_n = np.zeros_like(current_volume)
            grad_s = np.zeros_like(current_volume)
            grad_e = np.zeros_like(current_volume)
            grad_w = np.zeros_like(current_volume)
            grad_u = np.zeros_like(current_volume)
            grad_d = np.zeros_like(current_volume)
            
            grad_n[:, 1:, :] = current_volume[:, :-1, :] - current_volume[:, 1:, :]
            grad_s[:, :-1, :] = current_volume[:, 1:, :] - current_volume[:, :-1, :]
            grad_e[:, :, :-1] = current_volume[:, :, 1:] - current_volume[:, :, :-1]
            grad_w[:, :, 1:] = current_volume[:, :, :-1] - current_volume[:, :, 1:]
            grad_u[:-1, :, :] = current_volume[1:, :, :] - current_volume[:-1, :, :]
            grad_d[1:, :, :] = current_volume[:-1, :, :] - current_volume[1:, :, :]
            
            c_n = np.exp(-(grad_n * grad_n) / k_sq)
            c_s = np.exp(-(grad_s * grad_s) / k_sq)
            c_e = np.exp(-(grad_e * grad_e) / k_sq)
            c_w = np.exp(-(grad_w * grad_w) / k_sq)
            c_u = np.exp(-(grad_u * grad_u) / k_sq)
            c_d = np.exp(-(grad_d * grad_d) / k_sq)
            
            current_volume += lambda_val * (
                c_n * grad_n + c_s * grad_s + 
                c_e * grad_e + c_w * grad_w + 
                c_u * grad_u + c_d * grad_d
            )
        return current_volume
