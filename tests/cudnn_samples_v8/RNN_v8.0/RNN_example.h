/**
* Copyright 2016 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cudnn.h>
#include <cuda.h>
#include <stdio.h>
#include "fp16_emu.h"

#define COUNTOF(arr) int(sizeof(arr) / sizeof(arr[0]))

static size_t
getDeviceMemory(void) {
    struct cudaDeviceProp properties;
    int device;
    cudaError_t error;

    error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        fprintf(stderr, "failed to get device cudaError=%d\n", error);
        return 0;
    }

    error = cudaGetDeviceProperties(&properties, device);
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        fprintf(stderr, "failed to get properties cudaError=%d\n", error);
        return 0;
    }
    return properties.totalGlobalMem;
}

template <typename T_ELEM>
void printWeightAsMatrix(const T_ELEM *wDev, const int nRows, const int nCols)
{
    T_ELEM *wHost = (T_ELEM *) malloc(nRows*nCols*sizeof(T_ELEM));

    cudaMemcpy(wHost, wDev, nRows * nCols * sizeof(T_ELEM), cudaMemcpyDeviceToHost);

    printf("[DEBUG] Printing the weight matrix %dx%d:\n",nRows,nCols); fflush(0);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            printf("%1.6f ",(float)wHost[i*nCols + j]);
        }
        printf("\n");fflush(0);
    }

    free(wHost);
}

// Templated functions to get cudnnDataType_t from a templated type
template <typename T_ELEM> __inline__ cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<double>() { return CUDNN_DATA_DOUBLE; }
template <> __inline__ cudnnDataType_t getDataType<float>()  { return CUDNN_DATA_FLOAT;  }
template <> __inline__ cudnnDataType_t getDataType<half1>()  { return CUDNN_DATA_HALF;   }

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
        exit(-1);
    }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
        exit(-1);
    }
}

// Kernel and launcher to initialize GPU data to some constant value
template <typename T_ELEM>
__global__
void initGPUData_ker(T_ELEM *data, int numElements, T_ELEM value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        data[tid] = value;
    }
}

template <typename T_ELEM>
void initGPUData(T_ELEM *data, int numElements, T_ELEM value) {
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 1024;
    gridDim.x  = (numElements + blockDim.x - 1) / blockDim.x;

    initGPUData_ker<<<gridDim, blockDim>>>(data, numElements, value);
}

struct RNNSampleOptions {
    int dataType;
    int seqLength;     // Specify sequence length
    int numLayers;     // Specify number of layers
    int inputSize;     // Specify input vector size
    int hiddenSize;    // Specify hidden size
    int projSize;      // Specify LSTM cell output size after the recurrent projection
    int miniBatch;     // Specify max miniBatch size
    int inputMode;     // Specify how the input to the RNN model is processed by the first layer (skip or linear input)
    int dirMode;       // Specify the recurrence pattern (bidirectional and unidirectional)
    int cellMode;      // Specify cell type (RELU, TANH, LSTM, GRU)
    int biasMode;      // Specify bias type (no bias, single inp bias, single rec bias, double bias)
    int algorithm;     // Specify recurrence algorithm (standard, persist dynamic, persist static)
    int mathPrecision; // Specify math precision (half, float of double)
    int mathType;      // Specify math type (default, tensor op math or tensor op math with conversion)
    float dropout;
    int printWeights;

    RNNSampleOptions() { memset(this, 0, sizeof(*this)); };
};

template <typename T_ELEM>
class RNNSample {
   public:
    cudnnHandle_t cudnnHandle;

    cudnnRNNDataDescriptor_t xDesc;
    cudnnRNNDataDescriptor_t yDesc;

    cudnnTensorDescriptor_t hDesc;
    cudnnTensorDescriptor_t cDesc;

    cudnnRNNDescriptor_t rnnDesc;

    cudnnDropoutDescriptor_t dropoutDesc;

    void *x;
    void *hx;
    void *cx;

    void *dx;
    void *dhx;
    void *dcx;

    void *y;
    void *hy;
    void *cy;

    void *dy;
    void *dhy;
    void *dcy;

    int *seqLengthArray;
    int *devSeqLengthArray;

    void *weightSpace;
    void *dweightSpace;
    void *workSpace;
    void *reserveSpace;

    size_t weightSpaceSize;
    size_t workSpaceSize;
    size_t reserveSpaceSize;

    cudnnRNNAlgo_t       algorithm;
    cudnnRNNMode_t       cellMode;
    cudnnRNNBiasMode_t   biasMode;
    cudnnDirectionMode_t dirMode;
    cudnnRNNInputMode_t  inputMode;
    cudnnDataType_t      dataType;
    cudnnDataType_t      mathPrecision;
    cudnnMathType_t      mathType;

    int inputSize;
    int hiddenSize;
    int projSize;
    int numLayers;
    int seqLength;
    int miniBatch;

    // Local parameters
    int bidirectionalScale;
    int inputTensorSize;
    int outputTensorSize;
    int hiddenTensorSize;
    int numLinearLayers;

    double paddingFill;

    // Dimensions for hidden state tensors
    int dimHidden[3];
    int strideHidden[3];

    // Dropout descriptor parameters
    unsigned long long seed;
    size_t stateSize;
    void   *states;
    float dropout;

    // Profiling parameters
    int printWeights;
    cudaEvent_t start;
    cudaEvent_t stop;
    float timeForward;
    float timeBackwardData;
    float timeBackwardWeights;
    long long int flopCount;
    long long deviceMemoryAvailable;
    long long totalMemoryConsumption;

    RNNSample<T_ELEM>() :
        seqLengthArray   (NULL),
        devSeqLengthArray(NULL),
        x  (NULL),
        hx (NULL),
        cx (NULL),
        dx (NULL),
        dhx(NULL),
        dcx(NULL),
        y  (NULL),
        hy (NULL),
        cy (NULL),
        dy (NULL),
        dhy(NULL),
        dcy(NULL),
        states(NULL),
        weightSpace(NULL),
        dweightSpace(NULL),
        workSpace(NULL),
        reserveSpace(NULL)
        {};

    void setup(RNNSampleOptions &options);

    void run();

    void testgen();
};

static char * baseFile(char *fname)
{
    char *base;
    for (base = fname; *fname != '\0'; fname++) {
        if (*fname == '/' || *fname == '\\') {
            base = fname + 1;
        }
    }
    return base;
}

static void parseRNNSampleParameters(int argc, char **argv, RNNSampleOptions *options) {
    struct cmdParams {
        const char  *name;
        const char  *format;
        size_t      offset;
        const char  *description;
    } param[] = {
        { "dataType",     "%d",  offsetof(RNNSampleOptions, dataType),      "selects data format (0-FP16, 1-FP32, 2-FP64)"   },
        { "seqLength",    "%d",  offsetof(RNNSampleOptions, seqLength),     "sequence length"  },
        { "numLayers",    "%d",  offsetof(RNNSampleOptions, numLayers),     "number of layers" },
        { "inputSize",    "%d",  offsetof(RNNSampleOptions, inputSize),     "input vector size" },
        { "hiddenSize",   "%d",  offsetof(RNNSampleOptions, hiddenSize),    "hidden size" },
        { "projSize",     "%d",  offsetof(RNNSampleOptions, projSize),      "LSTM cell output size" },
        { "miniBatch",    "%d",  offsetof(RNNSampleOptions, miniBatch),     "miniBatch size" },
        { "inputMode",    "%d",  offsetof(RNNSampleOptions, inputMode),     "input to the RNN model (0-skip input, 1-linear input)" },
        { "dirMode",      "%d",  offsetof(RNNSampleOptions, dirMode),       "recurrence pattern (0-unidirectional, 1-bidirectional)" },
        { "cellMode",     "%d",  offsetof(RNNSampleOptions, cellMode),      "cell type (0-RELU, 1-TANH, 2-LSTM, 3-GRU)" },
        { "biasMode",     "%d",  offsetof(RNNSampleOptions, biasMode),      "bias type (0-no bias, 1-inp bias, 2-rec bias, 3-double bias" },
        { "algorithm",    "%d",  offsetof(RNNSampleOptions, algorithm),     "recurrence algorithm (0-standard, 1-persist static, 2-persist dynamic" },
        { "mathPrecision","%d",  offsetof(RNNSampleOptions, mathPrecision), "math precision (0-FP16, 1-FP32, 2-FP64)" },
        { "mathType",     "%d",  offsetof(RNNSampleOptions, mathType),      "math type (0-default, 1-tensor op math, 2-tensor op math with conversion)" },
        { "dropout",      "%g",  offsetof(RNNSampleOptions, dropout),       "dropout rate" },
        { "printWeights", "%d",  offsetof(RNNSampleOptions, printWeights),  "Print weights" }
    };

    if (argc == 1) {
        printf("This is the cuDNN RNN API sample.\n\n");
        printf("Usage: ./%s [OPTIONS]\n\nProgram options:\n\n", baseFile(*argv));

        for (int i = 0; i < COUNTOF(param); i++) {
            char buf[64];
            sprintf(buf, "-%s<%s>", param[i].name, param[i].format);
            printf("%-20s - %s\n", buf, param[i].description);
        }
        printf("[INFO] Default RNN sample parameters will be used!\n");

        // Default RNN options
        options->dataType      = 1;  // CUDNN_DATA_FLOAT
        options->seqLength     = 20;
        options->numLayers     = 2;
        options->inputSize     = 512;
        options->hiddenSize    = 512;
        options->projSize      = 512;
        options->miniBatch     = 64;
        options->inputMode     = 1;  // CUDNN_LINEAR_INPUT
        options->dirMode       = 0;  // CUDNN_UNIDIRECTIONAL
        options->cellMode      = 0;  // CUDNN_RNN_RELU
        options->biasMode      = 3;  // CUDNN_RNN_DOUBLE_BIAS
        options->algorithm     = 0;  // CUDNN_RNN_ALGO_STANDARD
        options->mathPrecision = 1;  // CUDNN_DATA_FLOAT
        options->mathType      = 0;  // CUDNN_DEFAULT_MATH
        options->dropout       = 0.;
        options->printWeights  = 0;
    }

    while (argc > 1) {
        argc--;
        argv++;

        for (int i = 0; i < COUNTOF(param); i++) {
            const char *pname = param[i].name;
            size_t plen = strlen(pname);
            if (strncmp(*argv + 1, pname, plen) == 0) {
                int count = sscanf(*argv + plen + 1, param[i].format, (char*)options + param[i].offset);
                if (count != 1) {
                    fprintf(stderr, "ERROR: missing numerical argument in option '%s'\n\n", *argv);
                    exit(-1);
                }
                break;
            }
        }
    }
}
