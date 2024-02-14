#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

cleanup() {
    ./scripts/kill_server.sh
    ./scripts/kill_iox.sh
}

run_tally_test() {

    # Launch tally server in the background
    ./scripts/start_server.sh &

    sleep 5

    echo $@
    
    if [[ ! -z "$REPLACE_CUBLAS" ]]; then
        echo "Running with REPLACE_CUBLAS set ..."
    fi

    # Launch client process
    ./scripts/start_client.sh $@

    ./scripts/kill_server.sh
}

test_list=(
    "./build/tests/elementwise"
    "./build/tests/elementwise_with_cond"
    "./build/tests/matmul"
    "./build/tests/cuda-memcpy-test"
    "./build/tests/max_pool"
    "./build/tests/cudnn_test"
    "./build/tests/cublas_test"
    "./build/tests/cublasLt_test"
    "./build/tests/test_nccl"
    "./build/tests/add_driver_api"
    "./build/tests/batched_gemm"
    "./build/tests/cublas_to_cutlass"
    "./build/tests/cutlass_bias_epilogue"
    "./build/tests/matmul_fp16"
    "./build/tests/basic_gemm"
    "./build/tests/batched_gemm_cutlass"
    "./build/tests/turing_tensorop_gemm"
    "./build/tests/ampere_tf32_tensorop_gemm"
    "python3 ./tests/pytorch_samples/train.py"
    "python3 ./tests/pytorch_samples/dropout.py"
    "python3 ./tests/pytorch_samples/run-torch-compile.py"
    "python3 ./tests/pytorch_samples/run-imagenet.py"
    "python3 ./tests/hidet_samples/run-hidet.py"
    "./tests/cudnn_samples_v8/RNN/RNN"
    "./tests/cudnn_samples_v8/RNN_v8.0/RNN"
    "./tests/cudnn_samples_v8/conv_sample/conv_sample"
    "./tests/cudnn_samples_v8/multiHeadAttention/multiHeadAttention -attnTrain1 -attnDataType0 -attnNumHeads3 -attnBatchSize6 -attnBeamSize1 -attnQsize8 -attnKsize8 -attnVsize8 -attnProjQsize2 -attnProjKsize2 -attnProjVsize2 -attnProjOsize8 -attnResLink0 -attnSeqLenQ3 -attnSeqLenK10"
    "./build/bin/samples"
    # "python3 ./tests/tensorflow_samples/tf_basic.py"
    # "python3 ./tests/tensorflow_samples/cifar_train.py"
    # "./tests/cudnn_samples_v8/mnistCUDNN/mnistCUDNN"
)

# Set up
trap cleanup ERR
set -e

# Build tally and tests
make
cd tests && cd cudnn_samples_v8 && make && cd .. && cd ..

./scripts/start_iox.sh &
sleep 5

# Run tests with tally-server-client
for item in "${test_list[@]}"; do
    run_tally_test $item
done

# Run tests again with REPLACE_CUBLAS set
for item in "${test_list[@]}"; do
    REPLACE_CUBLAS=TRUE run_tally_test $item
done

cleanup

echo All tests passed!

rm result.txt 2> /dev/null