#!/bin/bash

cleanup() {
    ./scripts/kill_server.sh
    ./scripts/kill_iox.sh
}

run_tally_test() {
    # Launch client process
    echo $@
    ./scripts/start_client.sh $@
}

test_list=(
    "./build/tests/elementwise"
    "./build/tests/matmul"
    "./build/tests/max_pool"
    "./build/tests/cudnn_test"
    "./build/tests/cublas_test"
    "./build/tests/cublasLt_test"
    "./build/tests/basic_gemm"
    "./build/tests/turing_tensorop_gemm"
    "./build/tests/ampere_tf32_tensorop_gemm"
    "python3 ./tests/tensorflow_samples/tf_basic.py"
    # "python3 ./tests/tensorflow_samples/cifar_train.py"
    "python3 ./tests/pytorch_samples/train.py"
    "python3 ./tests/pytorch_samples/run-triton.py"
    "python3 ./tests/pytorch_samples/run-torch-compile.py"
    "python3 ./tests/pytorch_samples/resnet50-compiled-1.py"
    "python3 ./tests/pytorch_samples/resnet50-compiled-2.py"
    "python3 ./tests/pytorch_samples/run-imagenet.py"
    "python3 ./tests/hidet_samples/run-hidet.py"
    "./tests/cudnn_samples_v8/RNN/RNN"
    "./tests/cudnn_samples_v8/RNN_v8.0/RNN"
    "./tests/cudnn_samples_v8/conv_sample/conv_sample"
    "./tests/cudnn_samples_v8/mnistCUDNN/mnistCUDNN"
    "./tests/cudnn_samples_v8/multiHeadAttention/multiHeadAttention -attnTrain1 -attnDataType0 -attnNumHeads3 -attnBatchSize6 -attnBeamSize1 -attnQsize8 -attnKsize8 -attnVsize8 -attnProjQsize2 -attnProjKsize2 -attnProjVsize2 -attnProjOsize8 -attnResLink0 -attnSeqLenQ3 -attnSeqLenK10"
    "./build/bin/samples"
)

# Set up
trap cleanup ERR
set -e

# Build tally and tests
make
cd tests && cd cudnn_samples_v8 && make && cd .. && cd ..

./scripts/start_iox.sh &
sleep 10

# Launch tally server in the background
./scripts/start_server.sh &
sleep 5

# Run tests with tally-server-client
for item in "${test_list[@]}"; do
    run_tally_test $item
done

cleanup

# # Run tests with offline client 
# for item in "${test_list[@]}"; do
#     echo $item
#     SCHEDULER_POLICY=WORKLOAD_AGNOSTIC_SHARING ./scripts/start_client.sh $item
# done

echo All tests passed!

rm result.txt 2> /dev/null