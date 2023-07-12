/**
* Copyright 2020 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "RNN_example.h"

template <typename T_ELEM>
void RNNSample<T_ELEM>::setup(RNNSampleOptions &options) {
    char projSizeUsage[48];
    char inputModeEnumValue[48];
    char dirModeEnumValue[48];
    char cellModeEnumValue[48];
    char biasModeEnumValue[48];
    char algorithmEnumValue[48];
    char mathPrecisionEnumValue[48];
    char mathTypeEnumValue[48];
    char dataTypeEnumValue[48];

    // Convert options to the sample parameters
    switch (options.inputMode) {
      case 0: inputMode = CUDNN_SKIP_INPUT;
              snprintf(inputModeEnumValue,sizeof(inputModeEnumValue),"CUDNN_SKIP_INPUT"); break;
      case 1: inputMode = CUDNN_LINEAR_INPUT;
              snprintf(inputModeEnumValue,sizeof(inputModeEnumValue),"CUDNN_LINEAR_INPUT"); break;
      default:
        printf("[ERROR] Wrong parameter for the inputMode!\n"); fflush(0);
        exit(-1);
    }

    switch (options.dirMode) {
      case 0: dirMode = CUDNN_UNIDIRECTIONAL;
              snprintf(dirModeEnumValue,sizeof(dirModeEnumValue),"CUDNN_UNIDIRECTIONAL"); break;
      case 1: dirMode = CUDNN_BIDIRECTIONAL;
              snprintf(dirModeEnumValue,sizeof(dirModeEnumValue),"CUDNN_BIDIRECTIONAL"); break;
      default:
        printf("[ERROR] Wrong parameter for the dirMode!\n"); fflush(0);
        exit(-1);
    }

    switch (options.cellMode) {
      case 0: cellMode = CUDNN_RNN_RELU;
              snprintf(cellModeEnumValue,sizeof(cellModeEnumValue),"CUDNN_RNN_RELU"); break;
      case 1: cellMode = CUDNN_RNN_TANH;
              snprintf(cellModeEnumValue,sizeof(cellModeEnumValue),"CUDNN_RNN_TANH"); break;
      case 2: cellMode = CUDNN_LSTM;
              snprintf(cellModeEnumValue,sizeof(cellModeEnumValue),"CUDNN_LSTM"); break;
      case 3: cellMode = CUDNN_GRU;
              snprintf(cellModeEnumValue,sizeof(cellModeEnumValue),"CUDNN_GRU"); break;
      default:
        printf("[ERROR] Wrong parameter for the cellMode!\n"); fflush(0);
        exit(-1);
    }

    switch (options.biasMode) {
      case 0: biasMode = CUDNN_RNN_NO_BIAS;
              snprintf(biasModeEnumValue,sizeof(biasModeEnumValue),"CUDNN_RNN_NO_BIAS"); break;
      case 1: biasMode = CUDNN_RNN_SINGLE_INP_BIAS;
              snprintf(biasModeEnumValue,sizeof(biasModeEnumValue),"CUDNN_RNN_SINGLE_INP_BIAS"); break;
      case 2: biasMode = CUDNN_RNN_SINGLE_REC_BIAS;
              snprintf(biasModeEnumValue,sizeof(biasModeEnumValue),"CUDNN_RNN_SINGLE_REC_BIAS"); break;
      case 3: biasMode = CUDNN_RNN_DOUBLE_BIAS;
              snprintf(biasModeEnumValue,sizeof(biasModeEnumValue),"CUDNN_RNN_DOUBLE_BIAS"); break;
      default:
        printf("[ERROR] Wrong parameter for the biasMode!\n"); fflush(0);
        exit(-1);
    }

    switch (options.algorithm) {
      case 0: algorithm = CUDNN_RNN_ALGO_STANDARD;
              snprintf(algorithmEnumValue,sizeof(algorithmEnumValue),"CUDNN_RNN_ALGO_STANDARD"); break;
      case 1: algorithm = CUDNN_RNN_ALGO_PERSIST_STATIC;
              snprintf(algorithmEnumValue,sizeof(algorithmEnumValue),"CUDNN_RNN_ALGO_PERSIST_STATIC"); break;
      case 2: algorithm = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
              snprintf(algorithmEnumValue,sizeof(algorithmEnumValue),"CUDNN_RNN_ALGO_PERSIST_DYNAMIC"); break;
      default:
        printf("[ERROR] Wrong parameter for the algorithm!\n"); fflush(0);
        exit(-1);
    }

    switch (options.mathPrecision) {
      case 0: mathPrecision = CUDNN_DATA_HALF;
              snprintf(mathPrecisionEnumValue,sizeof(mathPrecisionEnumValue),"CUDNN_DATA_HALF"); break;
      case 1: mathPrecision = CUDNN_DATA_FLOAT;
              snprintf(mathPrecisionEnumValue,sizeof(mathPrecisionEnumValue),"CUDNN_DATA_FLOAT"); break;
      case 2: mathPrecision = CUDNN_DATA_DOUBLE;
              snprintf(mathPrecisionEnumValue,sizeof(mathPrecisionEnumValue),"CUDNN_DATA_DOUBLE"); break;
      default:
        printf("[ERROR] Wrong parameter for the mathPrecision!\n"); fflush(0);
        exit(-1);
    }

    switch (options.mathType) {
      case 0: mathType = CUDNN_DEFAULT_MATH;
              snprintf(mathTypeEnumValue,sizeof(mathTypeEnumValue),"CUDNN_DEFAULT_MATH"); break;
      case 1: mathType = CUDNN_TENSOR_OP_MATH;
              snprintf(mathTypeEnumValue,sizeof(mathTypeEnumValue),"CUDNN_TENSOR_OP_MATH"); break;
      case 2: mathType = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
              snprintf(mathTypeEnumValue,sizeof(mathTypeEnumValue),"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION"); break;
      default:
        printf("[ERROR] Wrong parameter for the mathType!\n"); fflush(0);
        exit(-1);
    }

    switch (options.dataType) {
      case 0: dataType = CUDNN_DATA_HALF;
              snprintf(dataTypeEnumValue,sizeof(dataTypeEnumValue),"CUDNN_DATA_HALF"); break;
      case 1: dataType = CUDNN_DATA_FLOAT;
              snprintf(dataTypeEnumValue,sizeof(dataTypeEnumValue),"CUDNN_DATA_FLOAT"); break;
      case 2: dataType = CUDNN_DATA_DOUBLE;
              snprintf(dataTypeEnumValue,sizeof(dataTypeEnumValue),"CUDNN_DATA_DOUBLE"); break;
      default:
        printf("[ERROR] Wrong parameter for the dataType!\n"); fflush(0);
        exit(-1);
    }

    snprintf(projSizeUsage, sizeof(projSizeUsage), (cellMode == CUDNN_LSTM) ? "enabled" : "disabled");

    // Sizes
    inputSize    = options.inputSize;
    hiddenSize   = options.hiddenSize;
    projSize     = options.projSize;
    numLayers    = options.numLayers;
    seqLength    = options.seqLength;
    miniBatch    = options.miniBatch;
    dropout      = options.dropout;
    printWeights = options.printWeights;

    // Compute local parameters
    bidirectionalScale = (dirMode == CUDNN_BIDIRECTIONAL ? 2 : 1);

    // Calculating total elements per each tensor
    inputTensorSize  = seqLength * miniBatch * inputSize;
    outputTensorSize = seqLength * miniBatch * hiddenSize * bidirectionalScale;
    hiddenTensorSize = numLayers * miniBatch * hiddenSize * bidirectionalScale;

    // Dimensions for hidden state tensors
    dimHidden[0] = numLayers * bidirectionalScale;
    dimHidden[1] = miniBatch;
    dimHidden[2] = hiddenSize;

    strideHidden[0] = dimHidden[1] * dimHidden[2];
    strideHidden[1] = dimHidden[2];
    strideHidden[2] = 1;

    // Compute number of linear layers
    numLinearLayers = 0;
    if (cellMode == CUDNN_RNN_RELU || cellMode == CUDNN_RNN_TANH) {
        numLinearLayers = 2;
    } else if (cellMode == CUDNN_LSTM) {
        numLinearLayers = 8;
    } else if (cellMode == CUDNN_GRU) {
        numLinearLayers = 6;
    }

    // Pick a seed. (required by dropout descriptor)
    seed = 1337ull;

    paddingFill = 0.0;

    flopCount = numLinearLayers * 2ull * bidirectionalScale * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers;

    deviceMemoryAvailable  = getDeviceMemory();
    totalMemoryConsumption = (2 * inputTensorSize  +
                              2 * outputTensorSize +
                              8 * hiddenTensorSize) * sizeof(T_ELEM);

    // Check consistency of parameters
    if ((dataType == CUDNN_DATA_HALF   && (mathPrecision != CUDNN_DATA_HALF && mathPrecision != CUDNN_DATA_FLOAT)) ||
        (dataType == CUDNN_DATA_FLOAT  && (mathPrecision != CUDNN_DATA_FLOAT)) ||
        (dataType == CUDNN_DATA_DOUBLE && (mathPrecision != CUDNN_DATA_DOUBLE))) {
        printf("[ERROR] Inconsistent parameter: dataType does not match mathPrecision!\n"); fflush(0);
        exit(-1);
    }

    if ((dataType == CUDNN_DATA_FLOAT  && (mathType != CUDNN_DEFAULT_MATH && mathType != CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)) ||
        (dataType == CUDNN_DATA_DOUBLE && (mathType != CUDNN_DEFAULT_MATH))) {
        printf("[ERROR] Inconsistent parameter: dataType does not match mathType!\n"); fflush(0);
        exit(-1);
    }

    if (inputMode == CUDNN_SKIP_INPUT && inputSize != hiddenSize) {
        printf("[ERROR] Inconsistent parameter: inputSize does not match hiddenSize!\n"); fflush(0);
        exit(-1);
    }

    if (projSize > hiddenSize) {
        printf("[ERROR] Inconsistent parameter: projSize is larger than hiddenSize!\n"); fflush(0);
        exit(-1);
    }

    printf("[INFO] RNN sample parameters:\n");
    printf("[INFO] RNN seqLength        = %5d\n",seqLength);
    printf("[INFO] RNN numLayers        = %5d\n",numLayers);
    printf("[INFO] RNN inputSize        = %5d\n",inputSize);
    printf("[INFO] RNN hiddenSize       = %5d\n",hiddenSize);
    printf("[INFO] RNN projSize         = %5d (%s)\n", projSize, projSizeUsage);
    printf("[INFO] RNN miniBatch        = %5d\n",miniBatch);
    printf("[INFO] RNN inputMode        = %5d (%s)\n",inputMode,inputModeEnumValue);
    printf("[INFO] RNN dirMode          = %5d (%s)\n",dirMode,dirModeEnumValue);
    printf("[INFO] RNN cellMode         = %5d (%s)\n",cellMode,cellModeEnumValue);
    printf("[INFO] RNN biasMode         = %5d (%s)\n",biasMode,biasModeEnumValue);
    printf("[INFO] RNN algorithm        = %5d (%s)\n",algorithm,algorithmEnumValue);
    printf("[INFO] RNN mathPrecision    = %5d (%s)\n",mathPrecision,mathPrecisionEnumValue);
    printf("[INFO] RNN mathType         = %5d (%s)\n",mathType,mathTypeEnumValue);
    printf("[INFO] RNN dataType         = %5d (%s)\n",dataType,dataTypeEnumValue);
    printf("[INFO] RNN dropout          = %5g\n",dropout);
}

template <typename T_ELEM>
void RNNSample<T_ELEM>::testgen() {
    // Initialise weights and inputs
    // We initialise to something simple.
    // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.

    // Initialize inputs
    initGPUData<T_ELEM>((T_ELEM *)x, inputTensorSize, 1.0);
    if (hx != NULL) initGPUData<T_ELEM>((T_ELEM *)hx, hiddenTensorSize, 1.0);
    if (cx != NULL) initGPUData<T_ELEM>((T_ELEM *)cx, hiddenTensorSize, 1.0);

    initGPUData<T_ELEM>((T_ELEM *)dy, outputTensorSize, 1.0);
    if (dhy != NULL) initGPUData<T_ELEM>((T_ELEM *)dhy, hiddenTensorSize, 1.0);
    if (dcy != NULL) initGPUData<T_ELEM>((T_ELEM *)dcy, hiddenTensorSize, 1.0);

    // Initialize Weights
    cudnnTensorDescriptor_t wDesc;
    cudnnTensorDescriptor_t bDesc;

    cudnnErrCheck(cudnnCreateTensorDescriptor(&wDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&bDesc));

    for (int layer = 0; layer < numLayers * bidirectionalScale; layer++) {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
            cudnnDataType_t dataTypeTemp;
            int nbDims = 0;
            int dim[3], stride[3];
            T_ELEM *linLayerMat  = NULL;
            T_ELEM *linLayerBias = NULL;

            cudnnErrCheck(cudnnGetRNNWeightParams(cudnnHandle,
                                                  rnnDesc,
                                                  layer,
                                                  weightSpaceSize,
                                                  weightSpace,
                                                  linLayerID,
                                                  wDesc,
                                                  (void **)&linLayerMat,
                                                  bDesc,
                                                  (void **)&linLayerBias));

            if (linLayerMat) {
                cudnnErrCheck(cudnnGetTensorNdDescriptor(wDesc, 3, &dataTypeTemp, &nbDims, dim, stride));
                initGPUData<T_ELEM>(linLayerMat , dim[0] * dim[1] * dim[2], 1.0 / (dim[0] * dim[1] * dim[2]));
                if(printWeights) {
                    printWeightAsMatrix<T_ELEM>(linLayerMat,dim[1],dim[2]);
                }
            }

            if (linLayerBias) {
                cudnnErrCheck(cudnnGetTensorNdDescriptor(bDesc, 3, &dataTypeTemp, &nbDims, dim, stride));
                initGPUData<T_ELEM>(linLayerBias, dim[0] * dim[1] * dim[2], 1.0);
            }
        }
    }

    cudnnDestroyTensorDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(bDesc);
}

template <typename T_ELEM>
void RNNSample<T_ELEM>::run() {
    FILE *fp = NULL;
    fp = fopen("result.txt", "w");

    if (fp == NULL) {
        printf("[ERROR] Cannot open output file!\n");
        exit(-1);
    }

    // Create cudnn context
    cudnnErrCheck(cudnnCreate(&cudnnHandle));

    // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
    cudaErrCheck(cudaMalloc((void **)&x,  inputTensorSize  * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&y,  outputTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dx, inputTensorSize  * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dy, outputTensorSize * sizeof(T_ELEM)));

    cudaErrCheck(cudaMalloc((void **)&hx,  hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&cx,  hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&hy,  hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&cy,  hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dhx, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dcx, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dhy, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dcy, hiddenTensorSize * sizeof(T_ELEM)));

    // Memory allocation for seqLengthArray on the host and device
    seqLengthArray = (int *)malloc(miniBatch * sizeof(int));

    cudaErrCheck(cudaMalloc((void **)&devSeqLengthArray, miniBatch * sizeof(int)));
    totalMemoryConsumption += miniBatch * sizeof(int);

    for (int i = 0; i < miniBatch; i++) {
        seqLengthArray[i] = seqLength;
    }
    cudaErrCheck(cudaMemcpy(devSeqLengthArray, seqLengthArray, miniBatch * sizeof(int), cudaMemcpyHostToDevice));

    // Create RNN Data descriptors
    cudnnErrCheck(cudnnCreateRNNDataDescriptor(&xDesc));
    cudnnErrCheck(cudnnCreateRNNDataDescriptor(&yDesc));

    // Set RNN Data descriptors
    cudnnErrCheck(cudnnSetRNNDataDescriptor(xDesc,
                                            dataType,
                                            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                            seqLength,
                                            miniBatch,
                                            inputSize,
                                            seqLengthArray,
                                            &paddingFill));

    cudnnErrCheck(cudnnSetRNNDataDescriptor(yDesc,
                                            dataType,
                                            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                            seqLength,
                                            miniBatch,
                                            hiddenSize * bidirectionalScale,
                                            seqLengthArray,
                                            &paddingFill));


    cudnnErrCheck(cudnnCreateTensorDescriptor(&hDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cDesc));

    cudnnErrCheck(cudnnSetTensorNdDescriptor(hDesc, dataType, 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cDesc, dataType, 3, dimHidden, strideHidden));

    // Set up the dropout descriptor (needed for the RNN descriptor)
    cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));

    // How much memory does dropout need for states?
    // These states are used to generate random numbers internally
    // and should not be freed until the RNN descriptor is no longer used
    cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

    cudaErrCheck(cudaMalloc(&states, stateSize));
    totalMemoryConsumption += stateSize;

    cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc,
                                            cudnnHandle,
                                            dropout,
                                            states,
                                            stateSize,
                                            seed));

    // Set up the RNN descriptor
    cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));

    cudnnErrCheck(cudnnSetRNNDescriptor_v8(rnnDesc,
                                           algorithm,
                                           cellMode,
                                           biasMode,
                                           dirMode,
                                           inputMode,
                                           dataType,
                                           mathPrecision,
                                           mathType,
                                           inputSize,
                                           hiddenSize,
                                           projSize,
                                           numLayers,
                                           dropoutDesc,
                                           0));

    // Set up weights and bias parameters
    cudnnErrCheck(cudnnGetRNNWeightSpaceSize(cudnnHandle,rnnDesc,&weightSpaceSize));

    cudaErrCheck(cudaMalloc((void **)&weightSpace, weightSpaceSize));
    cudaErrCheck(cudaMalloc((void **)&dweightSpace, weightSpaceSize));
    totalMemoryConsumption += (2 * weightSpaceSize);

    // Set up work space and reserved memory
    cudnnErrCheck(cudnnGetRNNTempSpaceSizes(cudnnHandle,
                                            rnnDesc,
                                            CUDNN_FWD_MODE_TRAINING,
                                            xDesc,
                                            &workSpaceSize,
                                            &reserveSpaceSize));

    cudaErrCheck(cudaMalloc((void **)&workSpace, workSpaceSize));
    cudaErrCheck(cudaMalloc((void **)&reserveSpace, reserveSpaceSize));
    totalMemoryConsumption += (workSpaceSize + reserveSpaceSize);

    printf("[INFO] weightSpaceSize : %g MiB\n",weightSpaceSize/1024.0/1024.0);
    printf("[INFO] workSpaceSize   : %g MiB\n",workSpaceSize/1024.0/1024.0);
    printf("[INFO] reserveSpaceSize: %g MiB\n",reserveSpaceSize/1024.0/1024.0);
    printf("\n");
    printf("[INFO] Total required memory        : %g MiB\n",totalMemoryConsumption/1024.0/1024.0);
    printf("[INFO] Total available device memory: %g MiB\n",deviceMemoryAvailable/1024.0/1024.0);
    fflush(0);

    // Initialize all the data
    testgen();

    // Dynamic persistent RNN plan
    if (algorithm == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
        //       minibatch or datatype don't change.
        cudnnErrCheck(cudnnBuildRNNDynamic(cudnnHandle, rnnDesc, miniBatch));
    }

    // *********************************************************************************************************
    // At this point all of the setup is done. We now need to pass through the RNN.
    // *********************************************************************************************************

    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));

    cudnnErrCheck(cudnnRNNForward(cudnnHandle,
                                  rnnDesc,
                                  CUDNN_FWD_MODE_TRAINING,
                                  devSeqLengthArray,
                                  xDesc,
                                  x,
                                  yDesc,
                                  y,
                                  hDesc,
                                  hx,
                                  hy,
                                  cDesc,
                                  cx,
                                  cy,
                                  weightSpaceSize,
                                  weightSpace,
                                  workSpaceSize,
                                  workSpace,
                                  reserveSpaceSize,
                                  reserveSpace));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeForward, start, stop));

    cudaErrCheck(cudaEventRecord(start));

    cudnnErrCheck(cudnnRNNBackwardData_v8(cudnnHandle,
                                          rnnDesc,
                                          devSeqLengthArray,
                                          yDesc,
                                          y,
                                          dy,
                                          xDesc,
                                          dx,
                                          hDesc,
                                          hx,
                                          dhy,
                                          dhx,
                                          cDesc,
                                          cx,
                                          dcy,
                                          dcx,
                                          weightSpaceSize,
                                          weightSpace,
                                          workSpaceSize,
                                          workSpace,
                                          reserveSpaceSize,
                                          reserveSpace));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeBackwardData, start, stop));

    // cudnnRNNBackwardWeights adds to the data in dw.
    cudaErrCheck(cudaEventRecord(start));

    cudaErrCheck(cudaMemset(dweightSpace, 0, weightSpaceSize));

    cudnnErrCheck(cudnnRNNBackwardWeights_v8(cudnnHandle,
                                             rnnDesc,
                                             CUDNN_WGRAD_MODE_ADD,
                                             devSeqLengthArray,
                                             xDesc,
                                             x,
                                             hDesc,
                                             hx,
                                             yDesc,
                                             y,
                                             weightSpaceSize,
                                             dweightSpace,
                                             workSpaceSize,
                                             workSpace,
                                             reserveSpaceSize,
                                             reserveSpace));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeBackwardWeights, start, stop));

    // Report the performanc
    printf("[INFO] timeForward        : %2.5f ms\n",timeForward);
    printf("[INFO] timeBackwardData   : %2.5f ms\n",timeBackwardData);
    printf("[INFO] timeBackwardWeights: %2.5f ms\n",timeBackwardWeights);

    printf("[INFO] Forward: %3.0f GFLOPS\n", flopCount / (1e6 * timeForward));
    printf("[INFO] Backward: %3.0f GFLOPS, ", 2ull * flopCount / (1e6 * (timeBackwardData + timeBackwardWeights)));
    printf("(%3.0f GFLOPS), ", flopCount / (1e6 * timeBackwardData));
    printf("(%3.0f GFLOPS)\n", flopCount / (1e6 * timeBackwardWeights));
    fflush(0);

    // Save FLOPS to file
    fprintf(fp, "Forward: %3.0f GFLOPS\n", flopCount / (1e6 * timeForward));
    fprintf(fp, "Backward: %3.0f GFLOPS, ", 2ull * flopCount / (1e6 * (timeBackwardData + timeBackwardWeights)));
    fprintf(fp, "(%3.0f GFLOPS), ", flopCount / (1e6 * timeBackwardData));
    fprintf(fp, "(%3.0f GFLOPS)\n", flopCount / (1e6 * timeBackwardWeights));

    cudaDeviceSynchronize();

    // *********************************************************************************************************
    // Print checksums.
    // *********************************************************************************************************
    {
        T_ELEM *testOutputy;
        T_ELEM *testOutputhy;
        T_ELEM *testOutputcy;

        testOutputy = (T_ELEM *)malloc(outputTensorSize * sizeof(T_ELEM));
        testOutputhy = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));
        testOutputcy = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));

        cudaErrCheck(cudaMemcpy(testOutputy, y, outputTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        if (hy != NULL) {
            cudaErrCheck(cudaMemcpy(testOutputhy, hy, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }
        if (cy != NULL && cellMode == CUDNN_LSTM) {
            cudaErrCheck(cudaMemcpy(testOutputcy, cy, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }

        double checksumy = 0.f;
        double checksumhy = 0.f;
        double checksumcy = 0.f;

        for (int m = 0; m < miniBatch; m++) {
            double localSumi = 0;
            double localSumh = 0;
            double localSumc = 0;

            for (int j = 0; j < seqLength; j++) {
                for (int i = 0; i < hiddenSize * bidirectionalScale; i++) {
                    localSumi += (double) testOutputy[j * miniBatch * hiddenSize * bidirectionalScale + m * hiddenSize * bidirectionalScale + i];
                }
            }
            for (int j = 0; j < numLayers * bidirectionalScale; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    if (hy != NULL) {
                        localSumh += (double) testOutputhy[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                    if ((cy != NULL) && (cellMode == CUDNN_LSTM)) {
                        localSumc += (double) testOutputcy[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                }
            }

            checksumy += localSumi;
            checksumhy += localSumh;
            checksumcy += localSumc;
        }

        printf("y checksum %E     ", checksumy);
        fprintf(fp, "y checksum %E     ", checksumy);
        if (cellMode == CUDNN_LSTM) {
            printf("cy checksum %E     ", checksumcy);
            fprintf(fp, "cy checksum %E     ", checksumcy);
        }
        printf("hy checksum %E\n", checksumhy);
        fprintf(fp, "hy checksum %E\n", checksumhy);

        free(testOutputy);
        free(testOutputcy);
        free(testOutputhy);
    }

    {
        T_ELEM *testOutputdx;
        T_ELEM *testOutputdhx;
        T_ELEM *testOutputdcx;

        testOutputdx = (T_ELEM *)malloc(inputTensorSize * sizeof(T_ELEM));
        testOutputdhx = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));
        testOutputdcx = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));

        cudaErrCheck(cudaMemcpy(testOutputdx, dx, inputTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        if (dhx != NULL) {
            cudaErrCheck(cudaMemcpy(testOutputdhx, dhx, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }
        if ((dcx != NULL) && (cellMode == CUDNN_LSTM)) {
            cudaErrCheck(cudaMemcpy(testOutputdcx, dcx, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }

        double checksumdx = 0.f;
        double checksumdhx = 0.f;
        double checksumdcx = 0.f;

        for (int m = 0; m < miniBatch; m++) {
            double localSumdx = 0;
            double localSumdhx = 0;
            double localSumdcx = 0;

            for (int j = 0; j < seqLength; j++) {
                for (int i = 0; i < inputSize; i++) {
                    localSumdx += (double) testOutputdx[j * miniBatch * inputSize + m * inputSize + i];
                }
            }

            for (int j = 0; j < numLayers * bidirectionalScale; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    localSumdhx += (double) testOutputdhx[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    if (cellMode == CUDNN_LSTM) {
                        localSumdcx += (double) testOutputdcx[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                }
            }

            checksumdx += localSumdx;
            checksumdhx += localSumdhx;
            checksumdcx += localSumdcx;
        }

        printf("dx checksum %E    ", checksumdx);
        fprintf(fp, "dx checksum %E    ", checksumdx);
        if (cellMode == CUDNN_LSTM) {
            printf("dcx checksum %E    ", checksumdcx);
            fprintf(fp, "dcx checksum %E    ", checksumdcx);
        }
        printf("dhx checksum %E\n", checksumdhx);
        fprintf(fp, "dhx checksum %E\n", checksumdhx);

        free(testOutputdx);
        free(testOutputdhx);
        free(testOutputdcx);
    }

    {
        T_ELEM *testOutputdw;
        testOutputdw = (T_ELEM *)malloc(weightSpaceSize);

        cudaErrCheck(cudaMemcpy(testOutputdw, dweightSpace, weightSpaceSize, cudaMemcpyDeviceToHost));

        double checksumdw = 0.;

        for (int i = 0; i < weightSpaceSize / sizeof(T_ELEM); i++) {
            checksumdw += (double) testOutputdw[i];
        }

        printf("dw checksum %E\n", checksumdw);
        fprintf(fp, "dw checksum %E\n", checksumdw);

        free(testOutputdw);
    }

    //Free all previously allocated memory, destroy all created cudnn descriptors
    free(seqLengthArray);

    cudaFree(x);
    cudaFree(hx);
    cudaFree(cx);
    cudaFree(y);
    cudaFree(hy);
    cudaFree(cy);
    cudaFree(dx);
    cudaFree(dhx);
    cudaFree(dcx);
    cudaFree(dy);
    cudaFree(dhy);
    cudaFree(dcy);
    cudaFree(workSpace);
    cudaFree(reserveSpace);
    cudaFree(weightSpace);
    cudaFree(dweightSpace);
    cudaFree(states);
    cudaFree(devSeqLengthArray);

    cudnnDestroyRNNDataDescriptor(xDesc);
    cudnnDestroyRNNDataDescriptor(yDesc);

    cudnnDestroyTensorDescriptor(hDesc);
    cudnnDestroyTensorDescriptor(cDesc);

    cudnnDestroyDropoutDescriptor(dropoutDesc);
    cudnnDestroyRNNDescriptor(rnnDesc);

    cudnnDestroy(cudnnHandle);

    printf("Output saved to result.txt\n");
    fclose(fp);
}

template <typename T_ELEM>
void runRNNSample(RNNSampleOptions &options) {
    RNNSample<T_ELEM> sample;
    sample.setup(options);
    sample.run();
}

int main(int argc, char *argv[]) {
    RNNSampleOptions options;

    printf("Executing: %s", baseFile(argv[0]));
    for (int i = 1; i < argc; i++) {
        printf(" %s", argv[i]);
    }
    printf("\n");

    parseRNNSampleParameters(argc, argv, &options);

    switch (options.dataType) {
      case 0: runRNNSample<half1> (options); break;
      case 1: runRNNSample<float> (options); break;
      case 2: runRNNSample<double>(options); break;
    }

    return 0;
}
