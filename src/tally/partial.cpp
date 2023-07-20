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

std::function<void()> TallyServer::cudaLaunchKernel_Partial(const void * client_func, dim3  gridDim, dim3  blockDim, size_t  sharedMem, cudaStream_t  stream, char *params)
{
    assert(_kernel_client_addr_mapping.find((void *) client_func) != _kernel_client_addr_mapping.end());
    void *kernel_server_addr = _kernel_client_addr_mapping[(void *) client_func];
    auto &arg_sizes = _kernel_addr_to_args[kernel_server_addr];
    auto argc = arg_sizes.size();

    void *__args_arr[MAXIMUM_ARG_COUNT];
    int __args_idx = 0;
    int offset = 0;

    for (size_t i = 0; i < argc; i++) {
        __args_arr[__args_idx] = (void *) (params + offset);
        ++__args_idx;
        offset += arg_sizes[i];
    }

    return [kernel_server_addr, gridDim, blockDim, __args_arr, sharedMem, stream]() {
        auto err = cudaLaunchKernel((const void *) kernel_server_addr, gridDim, blockDim, (void **) __args_arr, sharedMem, stream);
        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cublasSgemm_v2_Partial(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    return [handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc]() {
        cublasStatus_t err = cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cudnnRNNBackwardWeights_Partial(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  seqLength, cudnnTensorDescriptor_t * xDesc, void * x, cudnnTensorDescriptor_t  hxDesc, void * hx, cudnnTensorDescriptor_t * yDesc, void * y, void * workSpace, size_t  workSpaceSizeInBytes, cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
    auto xDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);
    auto yDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);

    memcpy(xDesc_data, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(yDesc_data, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

    return [handle, rnnDesc, seqLength, xDesc_data, x, hxDesc, hx, yDesc_data, y, workSpace,
            workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes]() {
        
        auto err = cudnnRNNBackwardWeights(
            handle,
            rnnDesc,
            seqLength,
            xDesc_data,
            x,
            hxDesc,
            hx,
            yDesc_data,
            y,
            workSpace,
            workSpaceSizeInBytes,
            dwDesc,
            dw,
            reserveSpace,
            reserveSpaceSizeInBytes
        );

        free(xDesc_data);
        free(yDesc_data);

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cudnnRNNBackwardData_Partial(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
    auto yDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);
    auto dyDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);
    auto dxDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);

    memcpy(yDesc_data, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(dyDesc_data, dyDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(dxDesc_data, dxDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

    return [handle, rnnDesc, seqLength, yDesc_data, y, dyDesc_data, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc,
            hx, cxDesc, cx, dxDesc_data, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace,
            reserveSpaceSizeInBytes]() {
        
        auto err = cudnnRNNBackwardData(
            handle,
            rnnDesc,
            seqLength,
            yDesc_data,
            y,
            dyDesc_data,
            dy,
            dhyDesc,
            dhy, 
            dcyDesc, 
            dcy, 
            wDesc, 
            w, 
            hxDesc, 
            hx, 
            cxDesc, 
            cx, 
            dxDesc_data, 
            dx, 
            dhxDesc, 
            dhx, 
            dcxDesc, 
            dcx, 
            workSpace, 
            workSpaceSizeInBytes, 
            reserveSpace, 
            reserveSpaceSizeInBytes
        );

        free(yDesc_data);
        free(dyDesc_data);
        free(dxDesc_data);

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cudnnRNNForwardTraining_Partial(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
    auto xDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);
    auto yDesc_data = (cudnnTensorDescriptor_t *) std::malloc(sizeof(cudnnTensorDescriptor_t) * seqLength);

    memcpy(xDesc_data, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(yDesc_data, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

    return [handle, rnnDesc, seqLength, xDesc_data, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc_data, y, hyDesc, hy, cyDesc, cy,
            workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes]() {
        
        auto err = cudnnRNNForwardTraining(
            handle,
            rnnDesc,
            seqLength,
            xDesc_data,
            x,
            hxDesc,
            hx,
            cxDesc,
            cx,
            wDesc,
            w,
            yDesc_data,
            y,
            hyDesc,
            hy,
            cyDesc,
            cy,
            workSpace,
            workSpaceSizeInBytes,
            reserveSpace,
            reserveSpaceSizeInBytes
        );

        free(xDesc_data);
        free(yDesc_data);

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cudnnMultiHeadAttnBackwardData_Partial (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace, int winIdxLen)
{
    auto loWinIdx_data = (int *) malloc(sizeof(int) * winIdxLen);
    auto hiWinIdx_data = (int *) malloc(sizeof(int) * winIdxLen);

    memcpy(loWinIdx_data, loWinIdx, sizeof(int) * winIdxLen);
    memcpy(hiWinIdx_data, hiWinIdx, sizeof(int) * winIdxLen);

    return [handle, attnDesc, loWinIdx_data, hiWinIdx_data, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc,
            dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes,
            workSpace, reserveSpaceSizeInBytes, reserveSpace]() {
        
        auto err = cudnnMultiHeadAttnBackwardData(
                handle,
                attnDesc,
                loWinIdx_data,
                hiWinIdx_data,
                devSeqLengthsDQDO,
                devSeqLengthsDKDV,
                doDesc,
                dout,
                dqDesc,
                dqueries,
                queries,
                dkDesc,
                dkeys,
                keys,
                dvDesc,
                dvalues,
                values,
                weightSizeInBytes,
                weights,
                workSpaceSizeInBytes,
                workSpace,
                reserveSpaceSizeInBytes,
                reserveSpace
            );

        free(loWinIdx_data);
        free(hiWinIdx_data);

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cudnnMultiHeadAttnForward_Partial (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace, int winIdxLen)
{
    auto loWinIdx_data = (int *) malloc(sizeof(int) * winIdxLen);
    auto hiWinIdx_data = (int *) malloc(sizeof(int) * winIdxLen);

    memcpy(loWinIdx_data, loWinIdx, sizeof(int) * winIdxLen);
    memcpy(hiWinIdx_data, hiWinIdx, sizeof(int) * winIdxLen);

    return [handle, attnDesc, currIdx, loWinIdx_data, hiWinIdx_data, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries,
            residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes,
            workSpace, reserveSpaceSizeInBytes, reserveSpace]() {
        
        auto err = cudnnMultiHeadAttnForward(
            handle,
            attnDesc,
            currIdx,
            loWinIdx_data,
            hiWinIdx_data,
            devSeqLengthsQO,
            devSeqLengthsKV,
            qDesc,
            queries,
            residuals,
            kDesc,
            keys,
            vDesc,
            values,
            oDesc,
            out,
            weightSizeInBytes,
            weights,
            workSpaceSizeInBytes,
            workSpace,
            reserveSpaceSizeInBytes,
            reserveSpace
        );
      
        free(loWinIdx_data);
        free(hiWinIdx_data);

        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cublasSgemmEx_Partial (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
     return [handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc]() {
        cublasStatus_t err = cublasSgemmEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &alpha,
            A,
            Atype,
            lda,
            B,
            Btype,
            ldb,
            &beta,
            C,
            Ctype,
            ldc
        );
        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}

std::function<void()> TallyServer::cudnnTransformTensor_Partial (cudnnHandle_t  handle, uint64_t alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, uint64_t beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
    return [handle, alpha, xDesc, x, beta, yDesc, y]() {
        auto err = cudnnTransformTensor(
            handle,
            &alpha,
            xDesc,
            x,
            &beta,
            yDesc,
            y
        );
        CHECK_ERR_LOG_AND_EXIT(err, "Fail to launch kernel.");
    };
}
