#ifndef TALLY_CUDA_UTIL_H
#define TALLY_CUDA_UTIL_H

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cassert>
#include <map>
#include <elf.h>

#include <tally/util.h>
#include <tally/msg_struct.h>
#include <tally/cuda_launch.h>
#include <tally/generated/cuda_api.h>
#include <tally/cutlass/cutlass_struct.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fatbinary_section.h>

struct DeviceMemoryKey {
    void *addr;
    size_t size;
};

// Implicitly initialize CUDA context
inline void implicit_init_cuda_ctx()
{
    float *arr;
    cudaMalloc(&arr, sizeof(float));
    cudaFree(arr);
}

enum CUDA_MODULE_TYPE {
    PTX_STRING,
    FATBIN,
    ELF
};

inline CUDA_MODULE_TYPE get_cuda_module_type(const void * image)
{
    // Test if it is fatbin
    auto fbh = (fatBinaryHeader *) image;
    if (fbh->magic == FATBIN_MAGIC_NUMBER) {
        return CUDA_MODULE_TYPE::FATBIN;
    }

    // Test if it is in-memory elf format
    auto hdr = (Elf64_Ehdr *) image;
    if (hdr->e_ident[EI_MAG0] == ELFMAG0 && hdr->e_ident[EI_MAG1] == ELFMAG1 ||
        hdr->e_ident[EI_MAG2] == ELFMAG2 && hdr->e_ident[EI_MAG3] == ELFMAG3) {
        return CUDA_MODULE_TYPE::ELF;
    }

    // Test if it is ptx string
    std::string image_str((char *)image);
    if (containsSubstring(image_str, ".target")) {
        return CUDA_MODULE_TYPE::PTX_STRING;
    }

    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Cannot identify cuda module.");
}

inline cutlassOperation_t cublas_op_to_cutlass_op(cublasOperation_t op)
{
    switch(op) {
        case CUBLAS_OP_N:
            return cutlassOperation_t::CUTLASS_OP_N;
        case CUBLAS_OP_T:
            return cutlassOperation_t::CUTLASS_OP_T;
        case CUBLAS_OP_C:
            return cutlassOperation_t::CUTLASS_OP_C;
        default:
            throw std::runtime_error("unknown type");
    }
}

inline CUfunction_attribute convert_func_attribute(cudaFuncAttribute attr) {
    switch(attr) {
        case cudaFuncAttributeMaxDynamicSharedMemorySize:
            return CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        case cudaFuncAttributePreferredSharedMemoryCarveout:
            return CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        case cudaFuncAttributeClusterDimMustBeSet:
            return CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET;
        case cudaFuncAttributeRequiredClusterWidth:
            return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH;
        case cudaFuncAttributeRequiredClusterHeight:
            return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT;
        case cudaFuncAttributeRequiredClusterDepth: 
            return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH;
        case cudaFuncAttributeNonPortableClusterSizeAllowed:
            return CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED;
        case cudaFuncAttributeClusterSchedulingPolicyPreference:
            return CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        default:
            throw std::runtime_error("unknown type");
    }
}

inline std::string get_func_attr_str(CUfunction_attribute attr) {
    switch(attr) {
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
            return "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES";
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
            return "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT";
        case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET:
            return "CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET";
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH:
            return "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH";
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
            return "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT";
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: 
            return "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH";
        case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED:
            return "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED";
        case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
            return "CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE";
        default:
            throw std::runtime_error("unknown type");
    }
}

inline bool is_dev_addr(const std::vector<DeviceMemoryKey> &dev_addr_map, const void *addr)
{
    for (auto &dev_addr_key : dev_addr_map) {
        if ((uint64_t) addr >= (uint64_t) dev_addr_key.addr && (uint64_t) addr < ((uint64_t) dev_addr_key.addr + dev_addr_key.size)) {
            return true;
        }
    }

    return false;
}

inline void convert_stack_void_ptr_to_value(void *arrayOfElements, size_t elementCount, std::vector<DeviceMemoryKey> &dev_addr_map)
{
    auto pointer_arr = (void **) (arrayOfElements);

    for (int i = 0; i < elementCount; i++) {
        auto pointer = pointer_arr[i];

        if (pointer == nullptr) {
            continue;
        }

        auto found = is_dev_addr(dev_addr_map, pointer);

        // pointer points to CPU memory
        if (!found) {

            // Get the value from the CPU pointers
            uint64_t val = *((uint64_t *) pointer);

            // Store the value instead of addr
            pointer_arr[i] = (void *) val;
        }
    }
}

inline void free_dev_addr(std::vector<DeviceMemoryKey> &dev_addr_map, void *addr)
{   
    for (auto it = dev_addr_map.begin(); it != dev_addr_map.end(); it++) {
        auto key = *it;
        if (key.addr == addr) {
            dev_addr_map.erase(it);
            return;
        }
    }

    assert(false);
}


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err)
    {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        if constexpr (std::is_same<T, cudaError_t>::value) {
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        } else if constexpr (std::is_same<T, CUresult>::value) {
            const char *err_msg;
            cuGetErrorString(err, &err_msg);
            std::cerr << err_msg << " " << func << std::endl;
        }
    }
}

inline size_t get_cudnn_attribute_size(cudnnBackendAttributeType_t type)
{
    size_t attr_size;
    switch(type) {
        case CUDNN_TYPE_HANDLE:
            attr_size = sizeof(cudnnHandle_t);
            break;
        case CUDNN_TYPE_DATA_TYPE:
            attr_size = sizeof(cudnnDataType_t);
            break;
        case CUDNN_TYPE_BOOLEAN:
            attr_size = sizeof(bool);
            break;
        case CUDNN_TYPE_INT64:
            attr_size = sizeof(int64_t);
            break;
        case CUDNN_TYPE_FLOAT:
            attr_size = sizeof(float);
            break;
        case CUDNN_TYPE_DOUBLE:
            attr_size = sizeof(double);
            break;
        case CUDNN_TYPE_VOID_PTR:
            attr_size = sizeof(void *);
            break;
        case CUDNN_TYPE_CONVOLUTION_MODE:
            attr_size = sizeof(cudnnConvolutionMode_t);
            break;
        case CUDNN_TYPE_HEUR_MODE:
            attr_size = sizeof(cudnnBackendHeurMode_t);
            break;
        case CUDNN_TYPE_KNOB_TYPE:
            attr_size = sizeof(cudnnBackendKnobType_t);
            break;
        case CUDNN_TYPE_NAN_PROPOGATION:
            attr_size = sizeof(cudnnNanPropagation_t);
            break;
        case CUDNN_TYPE_NUMERICAL_NOTE:
            attr_size = sizeof(cudnnBackendNumericalNote_t);
            break;
        case CUDNN_TYPE_LAYOUT_TYPE:
            attr_size = sizeof(cudnnBackendLayoutType_t);
            break;
        case CUDNN_TYPE_ATTRIB_NAME:
            attr_size = sizeof(cudnnBackendAttributeName_t);
            break;
        case CUDNN_TYPE_POINTWISE_MODE:
            attr_size = sizeof(cudnnPointwiseMode_t);
            break;
        case CUDNN_TYPE_BACKEND_DESCRIPTOR:
            attr_size = sizeof(cudnnBackendDescriptor_t);
            break;
        case CUDNN_TYPE_GENSTATS_MODE:
            attr_size = sizeof(cudnnGenStatsMode_t);
            break;
        case CUDNN_TYPE_BN_FINALIZE_STATS_MODE:
            attr_size = sizeof(cudnnBnFinalizeStatsMode_t);
            break;
        case CUDNN_TYPE_REDUCTION_OPERATOR_TYPE:
            attr_size = sizeof(cudnnReduceTensorOp_t);
            break;
        case CUDNN_TYPE_BEHAVIOR_NOTE:
            attr_size = sizeof(cudnnBackendBehaviorNote_t);
            break;
        case CUDNN_TYPE_TENSOR_REORDERING_MODE:
            attr_size = sizeof(cudnnBackendTensorReordering_t);
            break;
        case CUDNN_TYPE_RESAMPLE_MODE:
            attr_size = sizeof(cudnnResampleMode_t);
            break;
        case CUDNN_TYPE_PADDING_MODE:
            attr_size = sizeof(cudnnPaddingMode_t);
            break;
        case CUDNN_TYPE_INT32:
            attr_size = sizeof(int32_t);
            break;
        case CUDNN_TYPE_CHAR:
            attr_size = sizeof(char);
            break;
        case CUDNN_TYPE_SIGNAL_MODE:
            attr_size = sizeof(cudnnSignalMode_t);
            break;
        case CUDNN_TYPE_FRACTION:
            attr_size = sizeof(cudnnFraction_t);
            break;
        case CUDNN_TYPE_NORM_MODE:
            attr_size = sizeof(cudnnBackendNormMode_t);
            break;
        case CUDNN_TYPE_NORM_FWD_PHASE:
            attr_size = sizeof(cudnnBackendNormFwdPhase_t);
            break;
        case CUDNN_TYPE_RNG_DISTRIBUTION:
            attr_size = sizeof(cudnnRngDistribution_t);
            break;
        default:
            throw std::runtime_error("unknown type");
    }

    return attr_size;
}

inline size_t get_CUjit_option_size(CUjit_option option)
{
    size_t option_size;
    switch(option) {
        case CU_JIT_MAX_REGISTERS:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_THREADS_PER_BLOCK:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_WALL_TIME:
            option_size = sizeof(float);
            break;
        case CU_JIT_INFO_LOG_BUFFER:
            throw std::runtime_error("Unsupported CUjit_option");
            break;
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_ERROR_LOG_BUFFER:
            throw std::runtime_error("Unsupported CUjit_option");
            break;
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_OPTIMIZATION_LEVEL:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_TARGET_FROM_CUCONTEXT:
            option_size = 0;
            break;
        case CU_JIT_TARGET:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_FALLBACK_STRATEGY:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_GENERATE_DEBUG_INFO:
            option_size = sizeof(int);
            break;
        case CU_JIT_LOG_VERBOSE:
            option_size = sizeof(int);
            break;
        case CU_JIT_GENERATE_LINE_INFO:
            option_size = sizeof(int);
            break;
        case CU_JIT_CACHE_MODE:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_FAST_COMPILE:
            throw std::runtime_error("Internal CUjit_option");
            break;
        case CU_JIT_GLOBAL_SYMBOL_NAMES:
            throw std::runtime_error("Unsupported CUjit_option");
            break;
        case CU_JIT_GLOBAL_SYMBOL_ADDRESSES:
            throw std::runtime_error("Unsupported CUjit_option");
            break;
        case CU_JIT_GLOBAL_SYMBOL_COUNT:
            option_size = sizeof(unsigned int);
            break;
        case CU_JIT_POSITION_INDEPENDENT_CODE:
            option_size = sizeof(int);
            break;
        default:
            throw std::runtime_error("Unknown or deprecated CUjit_option");
    }

    return option_size;
}

// typedef enum CUpointer_attribute_enum {
//     CU_POINTER_ATTRIBUTE_CONTEXT = 1,                     /**< The ::CUcontext on which a pointer was allocated or registered */
//     CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,                 /**< The ::CUmemorytype describing the physical location of a pointer */
//     CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,              /**< The address at which a pointer's memory may be accessed on the device */
//     CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,                /**< The address at which a pointer's memory may be accessed on the host */
//     CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,                  /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
//     CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,                 /**< Synchronize every synchronous memory operation initiated on this region */
//     CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,                   /**< A process-wide unique ID for an allocated memory region*/
//     CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,                  /**< Indicates if the pointer points to managed memory */
//     CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,              /**< A device ordinal of a device on which a pointer was allocated or registered */
//     CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10, /**< 1 if this pointer maps to an allocation that is suitable for ::cudaIpcGetMemHandle, 0 otherwise **/
//     CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,           /**< Starting address for this requested pointer */
//     CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,                 /**< Size of the address range for this requested pointer */
//     CU_POINTER_ATTRIBUTE_MAPPED = 13,                     /**< 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise **/
//     CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,       /**< Bitmask of allowed ::CUmemAllocationHandleType for this allocation **/
//     CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15, /**< 1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API **/
//     CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,               /**< Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given */
//     CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17              /**< Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL. **/
//     ,
//     CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,               /**< Size of the actual underlying mapping that the pointer belongs to **/
//     CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,          /**< The start address of the mapping that the pointer belongs to **/
//     CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20             /**< A process-wide unique id corresponding to the physical allocation the pointer belongs to **/
// } CUpointer_attribute;

inline size_t get_cupointer_attribute_size(CUpointer_attribute attribute)
{
    size_t attr_size;
    switch(attribute) {
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
            attr_size = sizeof(CUdeviceptr);
            break;
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
            attr_size = sizeof(int);
            break;
        default:
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + "unknown type: " + std::to_string(attribute));
    }

    return attr_size;
}

inline void print_arrayOfElements(cudnnBackendAttributeType_t  attributeType, int64_t  elementCount, const void * arrayOfElements)
{
    for (int i = 0; i < elementCount; i++) {

        std::cout << "arrayOfElements[" << i << "]: ";

        if (attributeType == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
            cudnnBackendDescriptor_t *arr = (cudnnBackendDescriptor_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_DATA_TYPE) {
            cudnnDataType_t *arr = (cudnnDataType_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_CONVOLUTION_MODE) {
            cudnnConvolutionMode_t *arr = (cudnnConvolutionMode_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_INT64) {
            int64_t *arr = (int64_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_FLOAT) {
            float *arr = (float *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_HANDLE) {
            cudnnHandle_t *arr = (cudnnHandle_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_VOID_PTR) {
            void **arr = (void **) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_BOOLEAN) {
            bool *arr = (bool *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_BN_FINALIZE_STATS_MODE) {
            cudnnBnFinalizeStatsMode_t *arr = (cudnnBnFinalizeStatsMode_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else if (attributeType == CUDNN_TYPE_HEUR_MODE) {
            cudnnBackendHeurMode_t *arr = (cudnnBackendHeurMode_t *) arrayOfElements;
            std::cout << arr[i] ;
        } else { 
            throw std::runtime_error("unsupported data type");
        }
        
        std::cout << std::endl;
    }
}

void write_cubin_to_file(const char *cubin_data, uint32_t cubin_size);

std::string get_fatbin_str_from_ptx_str(std::string &ptx_str);

std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);

std::vector<std::pair<std::string, uint32_t>> get_kernel_names_and_nparams_from_ptx(std::string &ptx_str);

template <class KERNEL_NAME_MAP_TYPE, class KERNEL_MAP_TYPE>
void register_kernels_from_ptx_fatbin(
    CUmodule cudaModule,
    std::string &ptx_str,
    std::string &fatbin_str,
    KERNEL_NAME_MAP_TYPE &kernel_name_map,
    KERNEL_MAP_TYPE &original_kernel_map,
    KERNEL_MAP_TYPE &ptb_kernel_map,
    KERNEL_MAP_TYPE &dynamic_ptb_kernel_map,
    KERNEL_MAP_TYPE &preemptive_ptb_kernel_map
)
{
    std::vector<KERNEL_MAP_TYPE *> kernel_map_ptrs {
        &original_kernel_map,
        &ptb_kernel_map,
        &dynamic_ptb_kernel_map,
        &preemptive_ptb_kernel_map
    };

    auto kernel_names_and_nparams = get_kernel_names_and_nparams_from_ptx(ptx_str);
    
    for (auto &name_and_nparams : kernel_names_and_nparams) {

        auto &kernel_name = name_and_nparams.first;

        if (kernel_name_map.find(kernel_name) == kernel_name_map.end()) {
            continue;
        }

        auto host_func = kernel_name_map[kernel_name];

        if (original_kernel_map.find(host_func) == original_kernel_map.end()) {

            std::vector<std::string> transform_kernel_names {
                kernel_name,
                kernel_name + "_tally_ptb",
                kernel_name + "_tally_dynamic_ptb",
                kernel_name + "_tally_preemptive_ptb"
            };
   
            for (int i = 0; i < transform_kernel_names.size(); i++) {

                uint32_t num_params = name_and_nparams.second;

                auto transform_kernel_name = transform_kernel_names[i];
                auto kernel_map_ptr = kernel_map_ptrs[i];

                // Ignore kernels without parameters
                if (i > 0 && num_params == 0) {
                    continue;
                }

                if (transform_kernel_name == kernel_name + "_tally_ptb") {
                    num_params += 1;
                } else if (transform_kernel_name == kernel_name + "_tally_dynamic_ptb") {
                    num_params += 3;
                } else if (transform_kernel_name == kernel_name + "_tally_preemptive_ptb") {
                    num_params += 4;
                }

                CUfunction function;
                CHECK_CUDA_ERROR(cuModuleGetFunction(&function, cudaModule, transform_kernel_name.c_str()));

                WrappedCUfunction wrapped_cu_func;

                wrapped_cu_func.func = function;
                wrapped_cu_func.num_args = num_params;
                CHECK_CUDA_ERROR(cuFuncGetAttribute (&(wrapped_cu_func.meta_data.max_threads_per_block), CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
                CHECK_CUDA_ERROR(cuFuncGetAttribute (&(wrapped_cu_func.meta_data.static_shmem_size_bytes), CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
                CHECK_CUDA_ERROR(cuFuncGetAttribute (&(wrapped_cu_func.meta_data.num_regs), CU_FUNC_ATTRIBUTE_NUM_REGS, function));
                CHECK_CUDA_ERROR(cuFuncGetAttribute (&(wrapped_cu_func.meta_data.max_dynamic_shmem_size_bytes), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function));

                if constexpr (std::is_same<KERNEL_MAP_TYPE, std::unordered_map<const void*, WrappedCUfunction>>::value) {
                    (*kernel_map_ptr)[host_func] = wrapped_cu_func;
                } else {
                    kernel_map_ptr->insert(host_func, wrapped_cu_func);
                }
            }
        }
    }
};

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf(std::string elf_path);

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf_str(std::string elf_str);

#endif // TALLY_CUDA_UTIL_H