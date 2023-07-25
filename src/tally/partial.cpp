#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>

#include "spdlog/spdlog.h"

#include <tally/log.h>
#include <tally/cuda_util.h>
#include <tally/msg_struct.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>

#define MAXIMUM_ARG_COUNT 50

std::function<void()> TallyServer::cudaLaunchKernel_Partial(const void * func, dim3  gridDim, dim3  blockDim, size_t  sharedMem, cudaStream_t  stream, char *params)
{
    auto &arg_sizes = _kernel_addr_to_args[(void *)func];
    auto argc = arg_sizes.size();

    void *__args_arr[MAXIMUM_ARG_COUNT];
    int __args_idx = 0;
    int offset = 0;

    for (size_t i = 0; i < argc; i++) {
        __args_arr[__args_idx] = (void *) (params + offset);
        ++__args_idx;
        offset += arg_sizes[i];
    }

    return [&, func, gridDim, blockDim, __args_arr, sharedMem, stream]() {
        auto err = cudaLaunchKernel((const void *) func, gridDim, blockDim, (void **) __args_arr, sharedMem, stream);
        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}