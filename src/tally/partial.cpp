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
    PTBKernelArgs *ptb_args, \
    uint32_t *curr_idx_arr, \
    SlicedKernelArgs *slice_args, \
    bool repeat, \
    float dur_seconds, \
    float *time_ms, \
    float *iters, \
    int32_t total_iters, \
    bool exit_if_fail \
    
std::pair<partial_t, void *> TallyServer::cudaLaunchKernel_Partial(const void *func, dim3  gridDim, dim3  blockDim, size_t  sharedMem, cudaStream_t  stream, char *params)
{
    assert(func);
    assert(_kernel_addr_to_args.find(func) != _kernel_addr_to_args.end());

    std::vector<uint32_t> arg_sizes = _kernel_addr_to_args[func];

    auto argc = arg_sizes.size();
    auto args_len = std::reduce(arg_sizes.begin(), arg_sizes.end());

    auto args = (char *) malloc(args_len);
    memcpy(args, params, args_len);

    void *__args_arr[MAXIMUM_ARG_COUNT];
    int __args_idx = 0;
    int offset = 0;

    for (size_t i = 0; i < argc; i++) {
        __args_arr[__args_idx] = (void *) (args + offset);
        ++__args_idx;
        offset += arg_sizes[i];
    }

    auto partial = [this, func, gridDim, blockDim, __args_arr, sharedMem, stream] (PARTIAL_ARGUMENTS) {

        CUresult err;

        if (repeat) {
            err = config.repeat_launch(func, gridDim, blockDim, (void **) __args_arr, sharedMem, stream, dur_seconds, ptb_args, curr_idx_arr, slice_args, time_ms, iters, total_iters);
        } else {
            err = config.launch(func, gridDim, blockDim, (void **) __args_arr, sharedMem, stream, ptb_args, curr_idx_arr, slice_args);
        }

        if (exit_if_fail && err) {

            auto kernel_name = host_func_to_demangled_kernel_name_map[func];

            char *str;
            cuGetErrorString(err, (const char **)&str);
            std::cout << kernel_name << std::endl;
            std::cout << str << std::endl;
            std::cout << config.str() << std::endl;

            CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
        }

        return err;
    };

    return std::make_pair(partial, (void *) args);
}

std::pair<partial_t, void *> TallyServer::cublasSgemm_v2_Partial(cublasSgemm_v2Arg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

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

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnRNNBackwardWeights_Partial(cudnnRNNBackwardWeightsArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnRNNBackwardWeights(
            args->handle,
            args->rnnDesc,
            args->seqLength,
            args->xDesc_yDesc,
            args->x,
            args->hxDesc,
            args->hx,
            args->xDesc_yDesc + args->seqLength,
            args->y,
            args->workSpace,
            args->workSpaceSizeInBytes,
            args->dwDesc,
            args->dw,
            args->reserveSpace,
            args->reserveSpaceSizeInBytes
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnRNNBackwardData_Partial(cudnnRNNBackwardDataArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnRNNBackwardData(
            args->handle,
            args->rnnDesc,
            args->seqLength,
            args->yDesc_dyDesc_dxDesc,
            args->y,
            args->yDesc_dyDesc_dxDesc + args->seqLength,
            args->dy,
            args->dhyDesc,
            args->dhy, 
            args->dcyDesc, 
            args->dcy, 
            args->wDesc, 
            args->w, 
            args->hxDesc, 
            args->hx, 
            args->cxDesc, 
            args->cx, 
            args->yDesc_dyDesc_dxDesc + args->seqLength * 2, 
            args->dx, 
            args->dhxDesc, 
            args->dhx, 
            args->dcxDesc, 
            args->dcx, 
            args->workSpace, 
            args->workSpaceSizeInBytes, 
            args->reserveSpace, 
            args->reserveSpaceSizeInBytes
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnRNNForwardTraining_Partial(cudnnRNNForwardTrainingArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnRNNForwardTraining(
            args->handle,
            args->rnnDesc,
            args->seqLength,
            args->xDesc_yDesc,
            args->x,
            args->hxDesc,
            args->hx,
            args->cxDesc,
            args->cx,
            args->wDesc,
            args->w,
            args->xDesc_yDesc + args->seqLength,
            args->y,
            args->hyDesc,
            args->hy,
            args->cyDesc,
            args->cy,
            args->workSpace,
            args->workSpaceSizeInBytes,
            args->reserveSpace,
            args->reserveSpaceSizeInBytes
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnMultiHeadAttnBackwardData_Partial(cudnnMultiHeadAttnBackwardDataArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnMultiHeadAttnBackwardData(
            args->handle,
            args->attnDesc,
            args->loWinIdx_hiWinIdx,
            args->loWinIdx_hiWinIdx + args->winIdxLen,
            args->devSeqLengthsDQDO,
            args->devSeqLengthsDKDV,
            args->doDesc,
            args->dout,
            args->dqDesc,
            args->dqueries,
            args->queries,
            args->dkDesc,
            args->dkeys,
            args->keys,
            args->dvDesc,
            args->dvalues,
            args->values,
            args->weightSizeInBytes,
            args->weights,
            args->workSpaceSizeInBytes,
            args->workSpace,
            args->reserveSpaceSizeInBytes,
            args->reserveSpace
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnMultiHeadAttnForward_Partial(cudnnMultiHeadAttnForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnMultiHeadAttnForward(
            args->handle,
            args->attnDesc,
            args->currIdx,
            args->loWinIdx_hiWinIdx,
            args->loWinIdx_hiWinIdx + args->winIdxLen,
            args->devSeqLengthsQO,
            args->devSeqLengthsKV,
            args->qDesc,
            args->queries,
            args->residuals,
            args->kDesc,
            args->keys,
            args->vDesc,
            args->values,
            args->oDesc,
            args->out,
            args->weightSizeInBytes,
            args->weights,
            args->workSpaceSizeInBytes,
            args->workSpace,
            args->reserveSpaceSizeInBytes,
            args->reserveSpace
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cublasSgemmEx_Partial(cublasSgemmExArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cublasSgemmEx(
            args->handle,
            args->transa,
            args->transb,
            args->m,
            args->n,
            args->k,
            &(args->alpha),
            args->A,
            args->Atype,
            args->lda,
            args->B,
            args->Btype,
            args->ldb,
            &(args->beta),
            args->C,
            args->Ctype,
            args->ldc
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnTransformTensor_Partial(cudnnTransformTensorArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnTransformTensor(
            args->handle,
            &(args->alpha),
            args->xDesc,
            args->x,
            &(args->beta),
            args->yDesc,
            args->y
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cublasSgemv_v2_Partial(cublasSgemv_v2Arg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cublasSgemv_v2(
            args->handle,
            args->trans,
            args->m,
            args->n,
            &args->alpha,
            args->A,
            args->lda,
            args->x,
            args->incx,
            &args->beta,
            args->y,
            args->incy
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnLRNCrossChannelForward_Partial(cudnnLRNCrossChannelForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnLRNCrossChannelForward(
            args->handle,
            args->normDesc,
            args->lrnMode,
            &(args->alpha),
            args->xDesc,
            args->x,
            &(args->beta),
            args->yDesc,
            args->y
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnSoftmaxForward_Partial(cudnnSoftmaxForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnSoftmaxForward(
            args->handle,
            args->algo,
            args->mode,
            &(args->alpha),
            args->xDesc,
            args->x,
            &(args->beta),
            args->yDesc,
            args->y
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnAddTensor_Partial(cudnnAddTensorArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnAddTensor(
            args->handle,
            (void *) &(args->alpha),
            args->aDesc,
            args->A,
            (void *) &(args->beta),
            args->cDesc,
            args->C
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cublasLtMatmul_Partial(cublasLtMatmulArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cublasLtMatmul(
            args->lightHandle,
            args->computeDesc,
            (void *) &(args->alpha),
            args->A,
            args->Adesc,
            args->B,
            args->Bdesc,
            (void *) &(args->beta),
            args->C,
            args->Cdesc,
            args->D,
            args->Ddesc,
            args->algo_is_null ? NULL : &(args->algo),
            args->workspace,
            args->workspaceSizeInBytes,
            args->stream
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnActivationForward_Partial(cudnnActivationForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnActivationForward(
            args->handle,
            args->activationDesc,
            (void *) &(args->alpha),
            args->xDesc,
            args->x,
            (void *) &(args->beta),
            args->yDesc,
            args->y
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnConvolutionForward_Partial(cudnnConvolutionForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnConvolutionForward(
            args->handle,
            (void *) &(args->alpha),
            args->xDesc,
            args->x,
            args->wDesc,
            args->w,
            args->convDesc,
            args->algo,
            args->workSpace,
            args->workSpaceSizeInBytes,
            (void *) &(args->beta),
            args->yDesc,
            args->y
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnPoolingForward_Partial(cudnnPoolingForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnPoolingForward(
            args->handle,
            args->poolingDesc,
            (void *) &(args->alpha),
            args->xDesc,
            args->x,
            (void *) &(args->beta),
            args->yDesc,
            args->y
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnMultiHeadAttnBackwardWeights_Partial(cudnnMultiHeadAttnBackwardWeightsArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnMultiHeadAttnBackwardWeights(
            args->handle,
            args->attnDesc,
            args->addGrad,
            args->qDesc,
            args->queries,
            args->kDesc,
            args->keys,
            args->vDesc,
            args->values,
            args->doDesc,
            args->dout,
            args->weightSizeInBytes,
            args->weights,
            args->dweights,
            args->workSpaceSizeInBytes,
            args->workSpace,
            args->reserveSpaceSizeInBytes,
            args->reserveSpace
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnReorderFilterAndBias_Partial(cudnnReorderFilterAndBiasArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnReorderFilterAndBias(
            args->handle,
            args->filterDesc,
            args->reorderType,
            args->filterData,
            args->reorderedFilterData,
            args->reorderBias,
            args->biasData,
            args->reorderedBiasData
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnBatchNormalizationForwardTrainingEx_Partial(cudnnBatchNormalizationForwardTrainingExArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnBatchNormalizationForwardTrainingEx(
            args->handle,
            args->mode,
            args->bnOps,
            (void *) &(args->alpha),
            (void *) &(args->beta),
            args->xDesc,
            args->xData,
            args->zDesc,
            args->zData,
            args->yDesc,
            args->yData,
            args->bnScaleBiasMeanVarDesc,
            args->bnScale,
            args->bnBias,
            args->exponentialAverageFactor,
            args->resultRunningMean,
            args->resultRunningVariance,
            args->epsilon,
            args->resultSaveMean,
            args->resultSaveInvVariance,
            args->activationDesc,
            args->workspace,
            args->workSpaceSizeInBytes,
            args->reserveSpace,
            args->reserveSpaceSizeInBytes
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnBatchNormalizationBackwardEx_Partial(cudnnBatchNormalizationBackwardExArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnBatchNormalizationBackwardEx(
            args->handle,
            args->mode,
            args->bnOps,
            (void *) &(args->alphaDataDiff),
            (void *) &(args->betaDataDiff),
            (void *) &(args->alphaParamDiff),
            (void *) &(args->betaParamDiff),
            args->xDesc,
            args->xData,
            args->yDesc,
            args->yData,
            args->dyDesc,
            args->dyData,
            args->dzDesc,
            args->dzData,
            args->dxDesc,
            args->dxData,
            args->dBnScaleBiasDesc,
            args->bnScaleData,
            args->bnBiasData,
            args->dBnScaleData,
            args->dBnBiasData,
            args->epsilon,
            args->savedMean,
            args->savedInvVariance,
            args->activationDesc,
            args->workSpace,
            args->workSpaceSizeInBytes,
            args->reserveSpace,
            args->reserveSpaceSizeInBytes
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnRNNBackwardWeights_v8_Partial(cudnnRNNBackwardWeights_v8Arg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnRNNBackwardWeights_v8(
            args->handle,
            args->rnnDesc,
            args->addGrad,
            args->devSeqLengths,
            args->xDesc,
            args->x,
            args->hDesc,
            args->hx,
            args->yDesc,
            args->y,
            args->weightSpaceSize,
            args->dweightSpace,
            args->workSpaceSize,
            args->workSpace,
            args->reserveSpaceSize,
            args->reserveSpace
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnRNNBackwardData_v8_Partial(cudnnRNNBackwardData_v8Arg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnRNNBackwardData_v8(
            args->handle,
            args->rnnDesc,
            args->devSeqLengths,
            args->yDesc,
            args->y,
            args->dy,
            args->xDesc,
            args->dx,
            args->hDesc,
            args->hx,
            args->dhy,
            args->dhx,
            args->cDesc,
            args->cx,
            args->dcy,
            args->dcx,
            args->weightSpaceSize,
            args->weightSpace,
            args->workSpaceSize,
            args->workSpace,
            args->reserveSpaceSize,
            args->reserveSpace
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnRNNForward_Partial(cudnnRNNForwardArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cudnnRNNForward(
            args->handle,
            args->rnnDesc,
            args->fwdMode,
            args->devSeqLengths,
            args->xDesc,
            args->x,
            args->yDesc,
            args->y,
            args->hDesc,
            args->hx,
            args->hy,
            args->cDesc,
            args->cx,
            args->cy,
            args->weightSpaceSize,
            args->weightSpace,
            args->workSpaceSize,
            args->workSpace,
            args->reserveSpaceSize,
            args->reserveSpace
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnBackendExecute_Partial(cudnnBackendExecuteArg *args, cudnnStatus_t *err)
{
    auto partial = [args, err] (PARTIAL_ARGUMENTS) {

        *err = cudnnBackendExecute(
            args->handle,
            args->executionPlan,
            args->variantPack
        );

        // CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!(*err)) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cublasGemmEx_Partial(cublasGemmExArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cublasGemmEx(
            args->handle,
            args->transa,
            args->transb,
            args->m,
            args->n,
            args->k,
            &(args->alpha),
            args->A,
            args->Atype,
            args->lda,
            args->B,
            args->Btype,
            args->ldb,
            &(args->beta),
            args->C,
            args->Ctype,
            args->ldc,
            args->computeType,
            args->algo
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cublasGemmStridedBatchedEx_Partial(cublasGemmStridedBatchedExArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cublasGemmStridedBatchedEx(
            args->handle,
            args->transa,
            args->transb,
            args->m,
            args->n,
            args->k,
            &(args->alpha),
            args->A,
            args->Atype,
            args->lda,
            args->strideA,
            args->B,
            args->Btype,
            args->ldb,
            args->strideB,
            &(args->beta),
            args->C,
            args->Ctype,
            args->ldc,
            args->strideC,
            args->batchCount,
            args->computeType,
            args->algo
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cublasSgemmStridedBatched_Partial(cublasSgemmStridedBatchedArg *args)
{
    auto partial = [this, args] (PARTIAL_ARGUMENTS) {

        auto err = cublasSgemmStridedBatched(
            args->handle,
            args->transa,
            args->transb,
            args->m,
            args->n,
            args->k,
            &(args->alpha),
            args->A,
            args->lda,
            args->strideA,
            args->B,
            args->ldb,
            args->strideB,
            &(args->beta),
            args->C,
            args->ldc,
            args->strideC,
            args->batchCount
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}

std::pair<partial_t, void *> TallyServer::cudnnReduceTensor_Partial(cudnnReduceTensorArg *args, void *indices)
{
    auto partial = [this, args, indices] (PARTIAL_ARGUMENTS) {

        auto err = cudnnReduceTensor(
            args->handle,
            args->reduceTensorDesc,
            indices,
            args->indicesSizeInBytes,
            args->workspace,
            args->workspaceSizeInBytes,
            &args->alpha,
            args->aDesc,
            args->A,
            &args->beta,
            args->cDesc,
            args->C
        );

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");

        if (!err) {
            return CUDA_SUCCESS;
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    };

    return std::make_pair(partial, nullptr);
}