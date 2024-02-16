#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Input tensor dimensions (for example, a 3D tensor with dimensions 5x4x3)
    int inputTensorDims[3] = {5, 4, 3};
    int inputTensorStrides[3] = {4*3, 3, 1}; // assuming contiguous storage in memory

    // Output tensor dimensions (reducing along the first dimension)
    int outputTensorDims[3] = {1, 4, 3};
    int outputTensorStrides[3] = {4*3, 3, 1};

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    
    cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 3, inputTensorDims, inputTensorStrides);
    cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 3, outputTensorDims, outputTensorStrides);

    // Allocate memory for input and output tensors
    float *inputData, *outputData;
    cudaMalloc(&inputData, sizeof(float) * 5 * 4 * 3);
    cudaMalloc(&outputData, sizeof(float) * 1 * 4 * 3);

    // Assuming inputData is filled with values here

    // Create and set up the reduction tensor descriptor
    cudnnReduceTensorDescriptor_t reduceTensorDesc;
    cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);
    cudnnSetReduceTensorDescriptor(reduceTensorDesc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);

    // Workspace for cudnnReduceTensor (query workspace size and allocate)
    size_t workspaceSize = 0;
    cudnnGetReductionWorkspaceSize(cudnn, reduceTensorDesc, inputDesc, outputDesc, &workspaceSize);
    void *workspace = nullptr;
    cudaMalloc(&workspace, workspaceSize);

    // Perform reduction
    const float alpha = 1.0f, beta = 0.0f;
    cudnnReduceTensor(cudnn, reduceTensorDesc, nullptr, 0, workspace, workspaceSize, &alpha, inputDesc, inputData, &beta, outputDesc, outputData);

    // Assuming outputData is used here

    // Cleanup
    cudaFree(workspace);
    cudaFree(outputData);
    cudaFree(inputData);
    cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroy(cudnn);

    return 0;
}
