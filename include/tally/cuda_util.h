#ifndef TALLY_CUDA_UTIL_H
#define TALLY_CUDA_UTIL_H

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cassert>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

struct DeviceMemoryKey {
    void *addr;
    size_t size;
};

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
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
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

std::string get_fatbin_str_from_ptx_str(std::string ptx_str);

std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);

std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> register_kernels_from_ptx_fatbin(
    std::vector<std::pair<std::string, std::string>> &ptx_fatbin_strs,
    std::map<std::string, const void *> &kernel_name_map
);

std::vector<std::pair<std::string, uint32_t>> get_kernel_names_and_nparams_from_ptx(std::string &ptx_str);

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf(std::string elf_path);

#endif // TALLY_CUDA_UTIL_H