#!/bin/bash

SCHEDULER_POLICY="NAIVE"

cleanup() {
    ./scripts/kill_tally_server.sh
    ./scripts/kill_iox_roudi.sh
}

run_tally_test() {
    # Launch client process
    echo $@
    ./scripts/start_tally_client.sh $@
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

./scripts/start_iox_roudi.sh &

sleep 5

# Launch tally server in the background
SCHEDULER_POLICY=$SCHEDULER_POLICY ./scripts/start_tally_server.sh &

echo wait for server to start ...
sleep 5

# Run tests
for item in "${test_list[@]}"; do
    run_tally_test $item
done

sleep 5

echo All tests passed!

cleanup

rm result.txt 2> /dev/null