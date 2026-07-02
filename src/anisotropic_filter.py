import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    pycuda_available = True
except ImportError:
    pycuda_available = False

cuda_code = """
__global__ void anisotropic_filter_3d(float* input, float* output, int width, int height, int depth, float lambda_val, float k_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int idx = z * (width * height) + y * width + x;
    float val = input[idx];
    
    float n = (y > 0) ? input[idx - width] : val;
    float s = (y < height - 1) ? input[idx + width] : val;
    float e = (x < width - 1) ? input[idx + 1] : val;
    float w = (x > 0) ? input[idx - 1] : val;
    float u = (z < depth - 1) ? input[idx + (width * height)] : val;
    float d = (z > 0) ? input[idx - (width * height)] : val;
    
    float dn = n - val;
    float ds = s - val;
    float de = e - val;
    float dw = w - val;
    float du = u - val;
    float dd = d - val;
    
    float cn = expf(-(dn * dn) / (k_val * k_val));
    float cs = expf(-(ds * ds) / (k_val * k_val));
    float ce = expf(-(de * de) / (k_val * k_val));
    float cw = expf(-(dw * dw) / (k_val * k_val));
    float cu = expf(-(du * du) / (k_val * k_val));
    float cd = expf(-(dd * dd) / (k_val * k_val));
    
    output[idx] = val + lambda_val * (cn * dn + cs * ds + ce * de + cw * dw + cu * du + cd * dd);
}
"""

class AnisotropicFilterEngine:
    def __init__(self, shape: tuple):
        self.shape = shape
        self.program = SourceModule(cuda_code) if pycuda_available else None
        self.kernel = self.program.get_function("anisotropic_filter_3d") if pycuda_available else None

    def execute_filter(self, h_input: np.ndarray, iterations: int = 3, lambda_val: float = 0.15, k_val: float = 20.0) -> np.ndarray:
        if not pycuda_available or self.kernel is None:
            return h_input
            
        depth, height, width = self.shape
        h_input = h_input.astype(np.float32)
        h_output = np.zeros_like(h_input)
        
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        block_dims = (8, 8, 4)
        grid_dims = (
            int(np.ceil(width / block_dims[0])),
            int(np.ceil(height / block_dims[1])),
            int(np.ceil(depth / block_dims[2]))
        )
        
        cuda.memcpy_htod(d_input, h_input)
        
        for _ in range(iterations):
            self.kernel(
                d_input, d_output,
                np.int32(width), np.int32(height), np.int32(depth),
                np.float32(lambda_val), np.float32(k_val),
                block=block_dims, grid=grid_dims
            )
            cuda.memcpy_dtod(d_input, d_output, h_input.nbytes)
            
        cuda.memcpy_dtoh(h_output, d_output)
        return h_output
