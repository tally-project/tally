#!/bin/bash

kill_iox_server() {
    # stop server
    pid=$(ps -ef | grep iox-roudi | grep -v grep | awk '{print $2}')
    kill -9 $pid &> /dev/null

    sleep 1
}

kill_tally_server() {
    # stop server
    pid=$(ps -ef | grep tally_server | grep -v grep | awk '{print $2}')
    kill -9 $pid &> /dev/null

    sleep 1
}

cleanup() {
    kill_tally_server
    kill_iox_server
}

run_tally_test() {
    # Launch client process
    echo $@
    ./start_client.sh $@
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
    "python3 ./tests/train.py"
    "python3 ./tests/run-triton.py"
    "python3 ./tests/run-torch-compile.py"
    "python3 ./tests/run-hidet.py"
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

./build/iox-roudi &

sleep 5

# Launch tally server in the background
    ./start_server.sh &

echo wait for server to start ...
sleep 5

# Run tests
for item in "${test_list[@]}"; do
    run_tally_test $item
done

sleep 5

echo All tests passed!

kill_tally_server
kill_iox_server