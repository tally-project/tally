import pathlib

current_file_path = pathlib.Path(__file__).parent
nccl_header_path = str(current_file_path / "../../../third_party/nccl/build/include/nccl.h")

CUDA_API_HEADER_FILES = [
    "/usr/local/cuda/include/cuda.h",
    "/usr/local/cuda/include/cuda_runtime.h",
    "/usr/local/cuda/include/cudnn.h",
    "/usr/local/cuda/include/cublas_v2.h",
    "/usr/local/cuda/include/cuda_profiler_api.h",
    "/usr/local/cuda/include/cudaProfiler.h",
    "/usr/local/cuda/include/nvrtc.h",
    "/usr/local/cuda/include/cublasLt.h",
    "/usr/local/cuda/include/curand.h",
    nccl_header_path
]

FUNC_SIG_MUST_CONTAIN = [("cu", "nvrtc", "nccl"), "(", ")"]
FUNC_SIG_MUST_NOT_CONTAIN = ["{", "}",]
FUNC_SIG_MUST_NOT_CONTAIN_KEYWORDS = ["noexcept", "return", "for", "throw", "sizeof"]

IGNORE_KEYWORDS = [
    "\"C\"", "CUDARTAPI", "extern", "__host__", "__cudart_builtin__",
    "__attribute__((deprecated))"
]

EXTERN_C_BEGIN = """

extern "C" { 

"""

EXTERN_C_END = """

}

"""