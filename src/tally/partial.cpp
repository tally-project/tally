#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>

#include "spdlog/spdlog.h"

#include <tally/log.h>
#include <tally/cuda_util.h>
#include <tally/cuda_launch.h>
#include <tally/msg_struct.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>

#define MAXIMUM_ARG_COUNT 50

#define PARTIAL_ARGUMENTS \
    CudaLaunchConfig config, \
    uint32_t *global_idx, \
    bool *retreat, \
    bool repeat, \
    float dur_seconds, \
    float *time_ms, \
    float *iters, \
    int32_t total_iters

template
std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)>
TallyServer::cudaLaunchKernel_Partial<const void *>(const void *, dim3, dim3, size_t, cudaStream_t, char *);

template
std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)>
TallyServer::cudaLaunchKernel_Partial<CUfunction>(CUfunction, dim3, dim3, size_t, cudaStream_t, char *);

template <typename T>
std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)>
TallyServer::cudaLaunchKernel_Partial(T func, dim3  gridDim, dim3  blockDim, size_t  sharedMem, cudaStream_t  stream, char *params)
{

    assert(func);

    std::vector<uint32_t> arg_sizes;

    if constexpr (std::is_same<T, const void *>::value) {
        assert(_kernel_addr_to_args.find(func) != _kernel_addr_to_args.end());
        arg_sizes = _kernel_addr_to_args[func];
    } else if constexpr (std::is_same<T, CUfunction>::value) {
        assert(_jit_kernel_addr_to_args.find(func) != _jit_kernel_addr_to_args.end());
        arg_sizes = _jit_kernel_addr_to_args[func];
    } else {
        throw std::runtime_error("Unsupported typename");
    }

    auto argc = arg_sizes.size();
    auto args_bytes = std::reduce(arg_sizes.begin(), arg_sizes.end());

    auto params_local = (char *) malloc(args_bytes);
    memcpy(params_local, params, args_bytes);

    void *__args_arr[MAXIMUM_ARG_COUNT];
    int __args_idx = 0;
    int offset = 0;

    for (size_t i = 0; i < argc; i++) {
        __args_arr[__args_idx] = (void *) (params_local + offset);
        ++__args_idx;
        offset += arg_sizes[i];
    }

    return [func, gridDim, blockDim, __args_arr, sharedMem, stream, params_local] (PARTIAL_ARGUMENTS) {

        CUresult err;

        if (repeat) {
            err = config.repeat_launch(func, gridDim, blockDim, (void **) __args_arr, sharedMem, stream, dur_seconds, global_idx, retreat, time_ms, iters, total_iters);
        } else {
            err = config.launch(func, gridDim, blockDim, (void **) __args_arr, sharedMem, stream, global_idx, retreat);
        }

        // free(params_local);

        if (err) {
            char *str;
            cuGetErrorString(err, (const char **)&str);
            std::cout << str << std::endl;
        }

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        return err;
    };
}

std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)>
TallyServer::cublasSgemm_v2_Partial(cublasSgemm_v2Arg *__args)
{
    auto args = (cublasSgemm_v2Arg *) malloc(sizeof(cublasSgemm_v2Arg));
    memcpy(args, __args, sizeof(cublasSgemm_v2Arg));

    return [args] (PARTIAL_ARGUMENTS) {

        auto err = cublasSgemm_v2(
            args->handle,
            args->transa,
            args->transb,
            args->m,
            args->n,
            args->k,
            &args->alpha,
            args->A,
            args->lda,
            args->B,
            args->ldb,
            &args->beta,
            args->C,
            args->ldc
        );

        // free(args);

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };
}
