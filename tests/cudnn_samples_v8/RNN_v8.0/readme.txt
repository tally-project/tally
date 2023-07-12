This sample demonstrates how to use cuDNN library to implement RNN(Recurrent Neural Network)
forward and backward pass.

The sample is based on RNN and its two improved methods: LTSM and GRU. For RNN, it's a
single-gate recurrent neural network with 1 ReLU or tanh activation function. For LTSM, it's
a four-gate Long Short-Term Memory networkwith no peephole connections. For GRU, it's a
three-gate network consisting of Gated Recurrent Unit.

Supported platforms: identical to cuDNN

================================================================================
USAGE:
  > ./RNN <flags>
Command line flags:
  -dataType{0,1,2}      : selects data format (0-FP16, 1-FP32, 2-FP64)
  -seqLength<int>       : sequence length
  -numLayers<int>       : number of layers
  -inputSize<int>       : input vector size
  -hiddenSize<int>      : hidden size
  -projSize<int>        : LSTM cell output size
  -miniBatch<int>       : miniBatch size
  -inputMode{0,1}       : input to the RNN model (0-skip input, 1-linear input)
  -dirMode{0,1}         : recurrence pattern (0-unidirectional, 1-bidirectional)
  -cellMode{0,1,2,3}    : cell type (0-RELU, 1-TANH, 2-LSTM, 3-GRU)
  -biasMode{0,1,2,3}    : bias type (0-no bias, 1-inp bias, 2-rec bias, 3-double bias
  -algorithm{0,1,2}     : recurrence algorithm (0-standard, 1-persist static, 2-persist dynamic
  -mathPrecision{0,1,2} : math precision (0-FP16, 1-FP32, 2-FP64)
  -mathType{0,1,2}      : math type (0-default, 1-tensor op math, 2-tensor op math with conversion)
  -dropout<float>       : dropout rate
  -printWeights{0,1}    : Print weights
  -H                    : Display this help message

================================================================================
RESULTS FORMAT:
The sample will run cudnnRNNForwardTraining, cudnnRNNBackwardData, and cudnnRNNBackwardWeights. It will print
the performance for each function, as well as checksums for the outputs of each function in the following format.
// <Forward Training GFLOPS>
// <Total Backward Pass GFLOPS>, <Backward Data GFLOPS>, <Backward Weights GFLOPS>
// <Forward Training Checksums>
// <Backward Data Checksums>
// <Backward Weights Checksums>

================================================================================
COMPARING RESULTS:
By default, the sample provides a compare.py script with four checksum reference files: golden_1.txt, golden_2.txt
golden_3.txt and golden_4.txt. Run ./RNN with the corresponding flags listed below, then use this command to compare results:
// > python compare.py result.txt golden_1.txt

The command line arguments corresponding to each of the reference files are as follows:
golden_1.txt (default case if you just run ./RNN)
// > ./RNN -dataType1 -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -projSize512 -miniBatch64 -inputMode1 -dirMode0 -cellMode0 -biasMode3 -algorithm0 -mathPrecision1 -mathType0 -dropout0.0 -printWeights0
// Forward: 1250 GFLOPS
// Backward: 1896 GFLOPS, (1299 GFLOPS), (3511 GFLOPS)
// y checksum 1.315793E+06     hy checksum 1.315212E+05
// dx checksum 6.676003E+01    dhx checksum 6.425050E+01
// dw checksum 1.453750E+09

golden_2.txt
// > ./RNN -dataType1 -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -projSize512 -miniBatch64 -inputMode1 -dirMode0 -cellMode1 -biasMode3 -algorithm0 -mathPrecision1 -mathType0 -dropout0.0 -printWeights0
// Forward: 1225 GFLOPS
// Backward: 1910 GFLOPS, (1299 GFLOPS), (3601 GFLOPS)
// y checksum 6.319591E+05     hy checksum 6.319605E+04
// dx checksum 4.501830E+00    dhx checksum 4.489543E+00
// dw checksum 5.012598E+07

golden_3.txt
// > ./RNN -dataType1 -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -projSize512 -miniBatch64 -inputMode1 -dirMode0 -cellMode2 -biasMode3 -algorithm0 -mathPrecision1 -mathType0 -dropout0.0 -printWeights0
// Forward: 2569 GFLOPS
// Backward: 2654 GFLOPS, (2071 GFLOPS), (3694 GFLOPS)
// y checksum 5.749536E+05     cy checksum 4.365091E+05     hy checksum 5.774818E+04
// dx checksum 3.842206E+02    dcx checksum 9.323785E+03    dhx checksum 1.182562E+01
// dw checksum 4.313461E+08

golden_4.txt
// > ./RNN -dataType1 -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -projSize512 -miniBatch64 -inputMode1 -dirMode0 -cellMode3 -biasMode3 -algorithm0 -mathPrecision1 -mathType0 -dropout0.0 -printWeights0
// Forward: 2310 GFLOPS
// Backward: 2536 GFLOPS, (1955 GFLOPS), (3606 GFLOPS)
// y checksum 6.358978E+05     hy checksum 6.281680E+04
// dx checksum 6.296622E+00    dhx checksum 2.289960E+05
// dw checksum 5.397419E+07
