#ifndef TALLY_CUDA_UTIL_H
#define TALLY_CUDA_UTIL_H

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

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

    std::cout << "attr_size: " << attr_size << std::endl;

    return attr_size;
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