#include <cstring>
#include <dlfcn.h>
#include <cassert>

#include "spdlog/spdlog.h"

#include <tally/transform.h>
#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/log.h>
#include <tally/msg_struct.h>
#include <tally/cache.h>
#include <tally/cache_util.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>

std::unique_ptr<TallyServer> TallyServer::server = std::make_unique<TallyServer>();

TallyServer::TallyServer()
{
    register_api_handler();

    __exit = [&](int sig_num) {
        is_quit__.store(true, std::memory_order_release);
        if (send_ipc != nullptr) send_ipc->disconnect();
        if (recv_ipc != nullptr) recv_ipc->disconnect();

        if (sig_num == SIGSEGV) {
            spdlog::info("Tally server received segfault signal.");
        }

        spdlog::info("Tally server shutting down ...");
        exit(0);
    };

    signal(SIGINT  , __exit_wrapper);
    signal(SIGABRT , __exit_wrapper);
    signal(SIGSEGV , __exit_wrapper);
    signal(SIGTERM , __exit_wrapper);
    signal(SIGHUP  , __exit_wrapper);
}

void TallyServer::start(uint32_t interval) {

    auto time_ckpt = std::chrono::steady_clock::now();
    double req_count = 0.; 

    send_ipc = new ipc::channel("server-to-client-230000", ipc::sender);
    recv_ipc = new ipc::channel("client-to-server-230000", ipc::receiver);

    load_cache();

    spdlog::info("Tally server is up ...");

    while (!is_quit__.load(std::memory_order_acquire)) {
        ipc::buff_t buf;
        while (buf.empty()) {
            buf = recv_ipc->recv(interval);

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - time_ckpt).count();
            if (elapsed >= 1000 * 60) {
                std::string log_msg = "Tally request processing speed: " + std::to_string(req_count / (double)elapsed * 1000. * 60.) + "/min";
                spdlog::info(log_msg);
                time_ckpt = now;
                req_count = 0;
            }

            if (is_quit__.load(std::memory_order_acquire)) return;
        }

        char const *dat = buf.get<char const *>();
        MessageHeader_t *msg_header = (MessageHeader_t *) dat;
        void *args = (void *) (dat + sizeof(CUDA_API_ENUM));

        auto handler = cuda_api_handler_map[msg_header->api_id];
        handler(args);
        req_count++;
    }
}

void TallyServer::load_cache()
{
    std::map<size_t, std::vector<CubinData>> cubin_map = TallyCache::cache->cubin_cache.cubin_map;

    for (auto &pair : cubin_map) {
        uint32_t cubin_size = pair.first;
        auto &cubin_vec = pair.second;

        for (auto &cubin_data : cubin_vec) {
            const my__fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) = {
                cubin_data.magic,
                cubin_data.version,
                (const long long unsigned int*) cubin_data.cubin_data.c_str(),
                0
            };
            void **handle = l__cudaRegisterFatBinary((void *) &__fatDeviceText);

            auto kernel_args = cubin_data.kernel_args;

            for (auto &kernel_args_pair : cubin_data.kernel_args) {

                auto &kernel_name = kernel_args_pair.first;
                auto &param_sizes = kernel_args_pair.second;

                // allocate an address for the kernel
                void *kernel_server_addr = malloc(8);

                // Register the kernel with this address
                _kernel_name_to_addr[kernel_name] = kernel_server_addr;
                l__cudaRegisterFunction(handle, (const char*) kernel_server_addr, (char *)kernel_name.c_str(), kernel_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
            
                _kernel_addr_to_args[kernel_server_addr] = param_sizes;
            }

            l__cudaRegisterFatBinaryEnd(handle);

            // For some reason, must call one cuda api call here. Otherwise it won't run.
            int *arr;
            cudaMalloc((void**)&arr, sizeof(int));
            cudaFree(arr);
        }
    }

}

void TallyServer::handle___cudaRegisterFatBinary(void *__args)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinary");
    auto args = (__cudaRegisterFatBinaryArg *) __args;
    bool cached = args->cached;
    magic = args->magic;
    version = args->version;

    struct fatBinaryHeader *header = (struct fatBinaryHeader *) args->data;
    size_t cubin_size = header->headerSize + header->fatSize;
    const char *cubin_data = (const char *) args->data;

    if (cached) {
        cubin_registered = true;
    } else {
        cubin_registered = TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size);
    }

    if (cubin_registered) {
        TALLY_SPD_LOG("Fat binary exists in cache, skipping register");
    }

    // Free data from last time
    // TODO: change this to managed pointer
    if (fatbin_data) {
        free(fatbin_data);
        fatbin_data = nullptr;
    }

    register_queue.clear();

    if (!cubin_registered) {
        // Load necessary data into cache if not exists
        cache_cubin_data(cubin_data, cubin_size, magic, version);

        fatBinSize = cubin_size;
        fatbin_data = (unsigned long long *) malloc(cubin_size);
        memcpy(fatbin_data, args->data, cubin_size);
    }
}

void TallyServer::handle___cudaRegisterFunction(void *__args)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFunction");
    auto args = (struct registerKernelArg *) __args;

    std::string kernel_name {args->data, args->kernel_func_len};
    register_queue.push_back( std::make_pair(args->host_func, kernel_name));
}

void TallyServer::handle___cudaRegisterFatBinaryEnd(void *__args)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinaryEnd");

    void **handle;
    void *kernel_server_addr;
    my__fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8)));
    std::map<std::string, std::vector<uint32_t>> kernel_names_and_param_sizes;

    if (!cubin_registered) {
        __fatDeviceText = { magic, version, fatbin_data, 0 };
        handle = l__cudaRegisterFatBinary((void *) &__fatDeviceText);
        kernel_names_and_param_sizes = TallyCache::cache->cubin_cache.get_kernel_args((const char*) fatbin_data, fatBinSize);
    }

    for (auto &kernel_pair : register_queue) {
        auto &client_addr = kernel_pair.first;
        auto &kernel_name = kernel_pair.second;
        
        if (!cubin_registered) {
            // allocate an address for the kernel
            kernel_server_addr = malloc(8);

            auto &param_sizes = kernel_names_and_param_sizes[kernel_name];

            // Register the kernel with this address
            _kernel_name_to_addr[kernel_name] = kernel_server_addr;
            _kernel_addr_to_args[kernel_server_addr] = param_sizes;
            l__cudaRegisterFunction(handle, (const char*) kernel_server_addr, (char *)kernel_name.c_str(), kernel_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
        }

        // For now, hoping that the client addr does not appear more than once
        // TODO: fix this
        if (_kernel_client_addr_mapping.find(client_addr) != _kernel_client_addr_mapping.end()) {
            assert(_kernel_client_addr_mapping[client_addr] = _kernel_name_to_addr[kernel_name]);
        } else {
            // Associate this client addr with the server address
            _kernel_client_addr_mapping[client_addr] = _kernel_name_to_addr[kernel_name];
        }
    }

    if (!cubin_registered) {
        l__cudaRegisterFatBinaryEnd(handle);
    }

    // For some reason, must call one cuda api call here. Otherwise it won't run.
    int *arr;
    cudaMalloc((void**)&arr, sizeof(int));
    cudaFree(arr);
}

void TallyServer::handle_cudaMemcpy(void *__args)
{
    TALLY_SPD_LOG("Received request: cudaMemcpy");
    auto args = (struct cudaMemcpyArg *) __args;
    struct cudaMemcpyResponse *res;
    size_t res_size = 0;

    if (args->kind == cudaMemcpyHostToDevice) {

        // Only care about dst (pointer to device memory) from the client call
        cudaError_t err = cudaMemcpy(args->dst, args->data, args->count, args->kind);

        res_size = sizeof(cudaError_t);
        res = (struct cudaMemcpyResponse *) malloc(res_size);
        res->err = err;
    } else if (args->kind == cudaMemcpyDeviceToHost){
        res_size = sizeof(cudaError_t) + args->count;
        res = (struct cudaMemcpyResponse *) malloc(res_size);

        // Only care about src (pointer to device memory) from the client call
        cudaError_t err = cudaMemcpy(res->data, args->src, args->count, args->kind);

        res->err = err;
    } else if (args->kind == cudaMemcpyDeviceToDevice) {
        res_size = sizeof(cudaError_t);
        res = (struct cudaMemcpyResponse *) malloc(res_size);
        cudaError_t err = cudaMemcpy(args->dst, args->src, args->count, args->kind);
        res->err = err;

    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    while(!send_ipc->send((void *) res, res_size)) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaMemcpyAsync(void *__args)
{
    TALLY_SPD_LOG("Received request: cudaMemcpyAsync");
    auto args = (struct cudaMemcpyAsyncArg *) __args;
    struct cudaMemcpyAsyncResponse *res;
    size_t res_size = 0;
    cudaError_t err;

    if (args->kind == cudaMemcpyHostToDevice) {

        // Only care about dst (pointer to device memory) from the client call
        err = cudaMemcpyAsync(args->dst, args->data, args->count, args->kind, args->stream);

        res_size = sizeof(cudaError_t);
        res = (struct cudaMemcpyAsyncResponse *) malloc(res_size);
        res->err = err;
    } else if (args->kind == cudaMemcpyDeviceToHost){
        res_size = sizeof(cudaError_t) + args->count;
        res = (struct cudaMemcpyAsyncResponse *) malloc(res_size);

        // Only care about src (pointer to device memory) from the client call
        err = cudaMemcpyAsync(res->data, args->src, args->count, args->kind, args->stream);

        res->err = err;
    } else if (args->kind == cudaMemcpyDeviceToDevice) {
        res_size = sizeof(cudaError_t);
        res = (struct cudaMemcpyAsyncResponse *) malloc(res_size);
        err = cudaMemcpyAsync(args->dst, args->src, args->count, args->kind, args->stream);
        res->err = err;

    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    while(!send_ipc->send((void *) res, res_size)) {
        send_ipc->wait_for_recv(1);
    }

    free(res);
}

void TallyServer::handle_cudaLaunchKernel(void *__args)
{
    TALLY_SPD_LOG("Received request: cudaLaunchKernel");
    auto args = (cudaLaunchKernelArg *) __args;
    void *kernel_server_addr = _kernel_client_addr_mapping[(void *) args->host_func];
    auto &arg_sizes = _kernel_addr_to_args[kernel_server_addr];
    auto argc = arg_sizes.size();

    void *__args_arr[argc];
    int __args_idx = 0;
    int offset = 0;

    for (size_t i = 0; i < argc; i++) {
        __args_arr[__args_idx] = (void *) (args->params + offset);
        ++__args_idx;
        offset += arg_sizes[i];
    }

    auto err = lcudaLaunchKernel((const void *) kernel_server_addr, args->gridDim, args->blockDim, &__args_arr[0], args->sharedMem, args->stream);

    while (!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasSgemm_v2(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasSgemm_v2");

    auto args = (struct cublasSgemm_v2Arg *) __args;

    const float alpha = args->alpha;
    const float beta = args->beta;

    cublasStatus_t err = cublasSgemm_v2(
		args->handle,
		args->transa,
		args->transb,
		args->m,
		args->n,
		args->k,
		&alpha,
		args->A,
		args->lda,
		args->B,
		args->ldb,
		&beta,
		args->C,
		args->ldc
    );
	
    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasLtMatmul(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmul");

    auto args = (struct cublasLtMatmulArg *) __args;

    cublasStatus_t err = cublasLtMatmul(
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
        &(args->algo),
        args->workspace,
        args->workspaceSizeInBytes,
        args->stream
    );

    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasLtMatmulDescSetAttribute(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulDescSetAttribute");

    auto args = (struct cublasLtMatmulDescSetAttributeArg *) __args;

    cublasStatus_t err = cublasLtMatmulDescSetAttribute(
		args->matmulDesc,
        args->attr,
        (void *)args->buf,
        args->sizeInBytes
    );
	
    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasLtMatrixLayoutSetAttribute(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutSetAttribute");

    auto args = (struct cublasLtMatrixLayoutSetAttributeArg *) __args;

    cublasStatus_t err = cublasLtMatrixLayoutSetAttribute(
		args->matLayout,
        args->attr,
        (void *)args->buf,
        args->sizeInBytes
    );
	
    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasLtMatmulPreferenceSetAttribute(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceSetAttribute");

    auto args = (struct cublasLtMatmulPreferenceSetAttributeArg *) __args;

    cublasStatus_t err = cublasLtMatmulPreferenceSetAttribute(
		args->pref,
        args->attr,
        (void *)args->buf,
        args->sizeInBytes
    );
	
    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasLtMatmulAlgoGetHeuristic(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoGetHeuristic");

    auto args = (struct cublasLtMatmulAlgoGetHeuristicArg *) __args;

    int requestedAlgoCount = args->requestedAlgoCount;
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[requestedAlgoCount];
    int returnAlgoCount;

    cublasStatus_t err = cublasLtMatmulAlgoGetHeuristic(
		args->lightHandle,
        args->operationDesc,
        args->Adesc,
        args->Bdesc,
        args->Cdesc,
        args->Ddesc,
        args->preference,
        args->requestedAlgoCount,
        heuristicResultsArray,
        &returnAlgoCount
    );

    uint32_t res_len =  sizeof(cublasLtMatmulAlgoGetHeuristicResponse) + sizeof(cublasLtMatmulHeuristicResult_t) * returnAlgoCount;
    auto res = (struct cublasLtMatmulAlgoGetHeuristicResponse *) std::malloc(res_len);

    res->returnAlgoCount = returnAlgoCount;
    res->err = err;
    memcpy(res->heuristicResultsArray, heuristicResultsArray, sizeof(cublasLtMatmulHeuristicResult_t) * returnAlgoCount);

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }

    free(res);
}

void TallyServer::handle_cudnnBackendSetAttribute(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnBackendSetAttribute");

    auto args = (struct cudnnBackendSetAttributeArg *) __args;

    std::cout << "descriptor: " << args->descriptor << std::endl;
    std::cout << "attributeName: " << args->attributeName << std::endl;
    std::cout << "attributeType: " << args->attributeType << std::endl;
    std::cout << "elementCount: " << args->elementCount << std::endl;

    int32_t type_size = get_cudnn_attribute_size(args->attributeType);
    std::cout << "type_size is " << type_size << std::endl;


    if (args->attributeType == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
        cudnnBackendDescriptor_t *arr = (cudnnBackendDescriptor_t *) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else if (args->attributeType == CUDNN_TYPE_DATA_TYPE) {
        cudnnDataType_t *arr = (cudnnDataType_t *) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else if (args->attributeType == CUDNN_TYPE_CONVOLUTION_MODE) {
        cudnnConvolutionMode_t *arr = (cudnnConvolutionMode_t *) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else if (args->attributeType == CUDNN_TYPE_INT64) {
        int64_t *arr = (int64_t *) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else if (args->attributeType == CUDNN_TYPE_FLOAT) {
        float *arr = (float *) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else if (args->attributeType == CUDNN_TYPE_HANDLE) {
        cudnnHandle_t *arr = (cudnnHandle_t *) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else if (args->attributeType == CUDNN_TYPE_VOID_PTR) {
        void **arr = (void **) args->arrayOfElements;
        for (int i = 0; i < args->elementCount; i++) {
            std::cout << "arrayOfElements[" << i << "]: " << arr[i] << std::endl;
        }
    } else {
        throw std::runtime_error("unhandled attributeType");
    }

    cudnnStatus_t err = cudnnBackendSetAttribute(
		args->descriptor,
        args->attributeName,
        args->attributeType,
        args->elementCount,
        (void *) args->arrayOfElements
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }

    assert(err == CUDNN_STATUS_SUCCESS);
}

void TallyServer::handle_cudnnBackendGetAttribute(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnBackendGetAttribute");

    auto args = (struct cudnnBackendGetAttributeArg *) __args;

    std::cout << "descriptor: " << args->descriptor << std::endl;
    std::cout << "attributeName: " << args->attributeName << std::endl;
    std::cout << "attributeType: " << args->attributeType << std::endl;
    std::cout << "requestedElementCount: " << args->requestedElementCount << std::endl;

    int32_t type_size = get_cudnn_attribute_size(args->attributeType);
    std::cout << "type_size is " << type_size << std::endl;

    int32_t buf_size = type_size * args->requestedElementCount;
    std::cout << "buf_size: " << buf_size << std::endl;

    void *arrayOfElements = malloc(buf_size);
    
    if (args->attributeName == 300) {
        cudnnBackendDescriptor_t desc;

        int64_t * elementCount = (int64_t *)malloc(sizeof(int64_t));

        cudnnStatus_t err;

        try {
            err = cudnnBackendGetAttribute(
                args->descriptor,
                args->attributeName,
                args->attributeType,
                args->requestedElementCount,
                elementCount,
                &desc
            );
        } catch (std::exception &e) {
            std::cout << "caught std::execution" << std::endl;
        }

        std::cout << "returned desc: " << desc << std::endl;    
        std::cout << "returned elementCount: " << *elementCount << std::endl;

        if (err != CUDNN_STATUS_SUCCESS) {
            std::cerr << "cudnnStatus_t not success" << std::endl;
            std::cout << cudnnGetErrorString(err) << std::endl;
        }

        uint32_t res_len =  sizeof(cudnnBackendGetAttributeResponse) + type_size * *elementCount;
        auto res = (struct cudnnBackendGetAttributeResponse *) std::malloc(res_len);

        res->elementCount = *elementCount;
        res->err = err;
        memcpy(res->arrayOfElements, &desc, type_size * *elementCount);
        free(arrayOfElements);

        while(!send_ipc->send((void *) res, res_len)) {
            send_ipc->wait_for_recv(1);
        }
        free(res);



    } else {
        int64_t elementCount;

        cudnnStatus_t err;

        try {
            err = cudnnBackendGetAttribute(
                args->descriptor,
                args->attributeName,
                args->attributeType,
                args->requestedElementCount,
                &elementCount,
                arrayOfElements
            );
        } catch (std::exception &e) {
            std::cout << "caught std::execution" << std::endl;
        }

        std::cout << "returned elementCount: " << elementCount << std::endl;



        if (err != CUDNN_STATUS_SUCCESS) {
            std::cerr << "cudnnStatus_t not success" << std::endl;
            std::cout << cudnnGetErrorString(err) << std::endl;
        }

        assert(err == CUDNN_STATUS_SUCCESS);

        uint32_t res_len =  sizeof(cudnnBackendGetAttributeResponse) + type_size * elementCount;
        auto res = (struct cudnnBackendGetAttributeResponse *) std::malloc(res_len);

        res->elementCount = elementCount;
        res->err = err;
        memcpy(res->arrayOfElements, arrayOfElements, type_size * elementCount);
        free(arrayOfElements);

        while(!send_ipc->send((void *) res, res_len)) {
            send_ipc->wait_for_recv(1);
        }
        free(res);
    }
}

void TallyServer::handle_cudnnActivationForward(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnActivationForward");

    auto args = (struct cudnnActivationForwardArg *) __args;

    cudnnStatus_t err = cudnnActivationForward(
        args->handle,
        args->activationDesc,
        (void *) &(args->alpha),
        args->xDesc,
        args->x,
        (void *) &(args->beta),
        args->yDesc,
        args->y
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }

    if (err != CUDNN_STATUS_SUCCESS) std::cerr << "cudnnStatus_t not success" << std::endl;
}

void TallyServer::handle_cudnnSetTensorNdDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnSetTensorNdDescriptor");

    auto args = (struct cudnnSetTensorNdDescriptorArg *) __args;

    cudnnStatus_t err = cudnnSetTensorNdDescriptor(
        args->tensorDesc,
        args->dataType,
        args->nbDims,
        args->dimA_and_strideA,
        args->dimA_and_strideA + args->nbDims
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }

    if (err != CUDNN_STATUS_SUCCESS) std::cerr << "cudnnStatus_t not success" << std::endl;
}

void TallyServer::handle_cudnnSetConvolutionNdDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnSetConvolutionNdDescriptor");

    auto args = (struct cudnnSetConvolutionNdDescriptorArg *) __args;

    cudnnStatus_t err = cudnnSetConvolutionNdDescriptor(
        args->convDesc,
        args->arrayLength,
        args->padA_and_filterStrideA_and_dilationA,
        args->padA_and_filterStrideA_and_dilationA + args->arrayLength,
        args->padA_and_filterStrideA_and_dilationA + args->arrayLength * 2,
        args->mode,
        args->computeType
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }

    if (err != CUDNN_STATUS_SUCCESS) std::cerr << "cudnnStatus_t not success" << std::endl;
}

void TallyServer::handle_cudnnSetFilterNdDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnSetFilterNdDescriptor");

    auto args = (struct cudnnSetFilterNdDescriptorArg *) __args;

    cudnnStatus_t err = cudnnSetFilterNdDescriptor(
        args->filterDesc,
        args->dataType,
        args->format,
        args->nbDims,
        args->filterDimA
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }

    if (err != CUDNN_STATUS_SUCCESS) std::cerr << "cudnnStatus_t not success" << std::endl;
}

void TallyServer::handle_cudnnConvolutionForward(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnConvolutionForward");

    auto args = (struct cudnnConvolutionForwardArg *) __args;

    cudnnStatus_t err = cudnnConvolutionForward(
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

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}