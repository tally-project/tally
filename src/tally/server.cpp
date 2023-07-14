#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>

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

// Used to check whether an address points to device memory
std::vector<DeviceMemoryKey> dev_addr_map;

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

    send_ipc = new ipc::channel("server-to-client-340000", ipc::sender);
    recv_ipc = new ipc::channel("client-to-server-340000", ipc::receiver);

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

void TallyServer::handle_cudaMalloc(void *__args)
{
	TALLY_SPD_LOG("Received request: cudaMalloc");

	auto args = (struct cudaMallocArg *) __args;
	void *devPtr;

	cudaError_t err = cudaMalloc(&devPtr, args->size);
	struct cudaMallocResponse res {
		devPtr,
		err
    };

    // Keep track that this addr is device memory
    if (err == cudaSuccess) {
        dev_addr_map.push_back( DeviceMemoryKey(devPtr, args->size) );
    }
    
    while(!send_ipc->send((void *) &res, sizeof(struct cudaMallocResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaFree(void *__args)
{
	TALLY_SPD_LOG("Received request: cudaFree");

    auto args = (struct cudaFreeArg *) __args;
    cudaError_t err = cudaFree(
		args->devPtr
    );

    if (err == cudaSuccess) {
        free_dev_addr(dev_addr_map, args->devPtr);
    }

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
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

    void *arrayOfElements =  args->arrayOfElements;

    // In case the values contain CPU pointers
    // Then we will allocate again
    std::vector <void *> allocated_mem;

    if (args->attributeType == CUDNN_TYPE_VOID_PTR) {
        auto pointer_arr = (void **) (args->arrayOfElements);

        for (int i = 0; i < args->elementCount; i++) {
            auto pointer = pointer_arr[i];

            if (pointer == nullptr) {
                continue;
            }
            
            auto found = is_dev_addr(dev_addr_map, pointer);

            // pointer points to CPU memory
            if (!found) {
                uint64_t val = (uint64_t) pointer;

                // Store the value instead of addr
                pointer_arr[i] = std::malloc(sizeof(uint64_t));
                *((uint64_t *) (pointer_arr[i])) = val;
                allocated_mem.push_back(pointer_arr[i]);
            }
        }
    }

    cudnnStatus_t err = cudnnBackendSetAttribute(
		args->descriptor,
        args->attributeName,
        args->attributeType,
        args->elementCount,
        arrayOfElements
    );

    // These memory can be free once cudnnBackendSetAttribute returns
    for (auto &addr : allocated_mem) {
        std::free(addr);
    }

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }

    assert(err == CUDNN_STATUS_SUCCESS);
}

void TallyServer::handle_cudnnBackendGetAttribute(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnBackendGetAttribute");

    auto args = (struct cudnnBackendGetAttributeArg *) __args;


    int32_t type_size = get_cudnn_attribute_size(args->attributeType);
    int32_t buf_size = type_size * args->requestedElementCount;
    assert(buf_size >= 0);

    void *arrayOfElements;

    if (buf_size == 0) {
        arrayOfElements = NULL;
    } else {
        arrayOfElements = args->arrayOfElementsData;
    }

    int64_t elementCount = 0;

    cudnnStatus_t err = cudnnBackendGetAttribute(
        args->descriptor,
        args->attributeName,
        args->attributeType,
        args->requestedElementCount,
        args->elementCount ? (&elementCount) : NULL,
        args->arrayOfElements ? (arrayOfElements) : NULL
    );

    int64_t arrayOfElementsSize;
    if (args->elementCount) {
        arrayOfElementsSize = std::min(elementCount, args->requestedElementCount);
    } else {
        arrayOfElementsSize = args->requestedElementCount;
    }
    
    uint32_t res_len =  sizeof(cudnnBackendGetAttributeResponse) + type_size * arrayOfElementsSize;
    auto res = (struct cudnnBackendGetAttributeResponse *) std::malloc(res_len);

    res->arrayOfElementsSize = arrayOfElementsSize;
    res->elementCount = elementCount;
    res->err = err;
    if (arrayOfElements) {
        memcpy(res->arrayOfElements, arrayOfElements, type_size * arrayOfElementsSize);
    } else {
        assert(arrayOfElementsSize <= 0);
    }
    // free(arrayOfElements);

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }
    free(res); 
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

void TallyServer::handle_cudnnGetConvolutionNdForwardOutputDim(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionNdForwardOutputDim");

    auto args = (struct cudnnGetConvolutionNdForwardOutputDimArg *) __args;

    uint32_t res_len = sizeof(cudnnGetConvolutionNdForwardOutputDimResponse) + sizeof(int) * args->nbDims;
    auto res = (cudnnGetConvolutionNdForwardOutputDimResponse *) malloc(res_len);

    res->err = cudnnGetConvolutionNdForwardOutputDim(
		args->convDesc,
        args->inputTensorDesc,
        args->filterDesc,
        args->nbDims,
        res->tensorOuputDimA
    );
    
    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }

    free(res);
}

void TallyServer::handle_cudnnGetConvolutionForwardAlgorithm_v7(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionForwardAlgorithm_v7");

    auto args = (struct cudnnGetConvolutionForwardAlgorithm_v7Arg *) __args;

    uint32_t res_len = sizeof(cudnnGetConvolutionForwardAlgorithm_v7Response) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * args->requestedAlgoCount;
    auto res = (cudnnGetConvolutionForwardAlgorithm_v7Response *) malloc(res_len);

    res->err = cudnnGetConvolutionForwardAlgorithm_v7(
		args->handle,
        args->srcDesc,
        args->filterDesc,
        args->convDesc,
        args->destDesc,
        args->requestedAlgoCount,
        &res->returnedAlgoCount,
        res->perfResults
    );

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }

    free(res);
}

void TallyServer::handle_cudnnFindConvolutionForwardAlgorithm(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnFindConvolutionForwardAlgorithm");

    auto args = (struct cudnnFindConvolutionForwardAlgorithmArg *) __args;

    uint32_t res_len = sizeof(cudnnFindConvolutionForwardAlgorithmResponse) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * args->requestedAlgoCount;
    auto res = (cudnnFindConvolutionForwardAlgorithmResponse *) malloc(res_len);

    res->err = cudnnFindConvolutionForwardAlgorithm(
		args->handle,
        args->xDesc,
        args->wDesc,
        args->convDesc,
        args->yDesc,
        args->requestedAlgoCount,
        &res->returnedAlgoCount,
        res->perfResults
    );

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }

    free(res);
}

void TallyServer::handle_cudnnAddTensor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnAddTensor");

    auto args = (struct cudnnAddTensorArg *) __args;

    cudnnStatus_t err = cudnnAddTensor(
		args->handle,
        (void *) &(args->alpha),
        args->aDesc,
        args->A,
        (void *) &(args->beta),
        args->cDesc,
        args->C
    );
	
    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnSetPoolingNdDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnSetPoolingNdDescriptor");

    auto args = (struct cudnnSetPoolingNdDescriptorArg *) __args;

    cudnnStatus_t err = cudnnSetPoolingNdDescriptor(
        args->poolingDesc,
        args->mode,
        args->maxpoolingNanOpt,
        args->nbDims,
        args->windowDimA_paddingA_strideA,
        args->windowDimA_paddingA_strideA + args->nbDims,
        args->windowDimA_paddingA_strideA + args->nbDims * 2
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnGetPoolingNdDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnGetPoolingNdDescriptor");

    auto args = (struct cudnnGetPoolingNdDescriptorArg *) __args;

    int *windowDimA = (int *) malloc(sizeof(int) * args->nbDimsRequested);
    int *paddingA = (int *) malloc(sizeof(int) * args->nbDimsRequested);
    int *strideA = (int *) malloc(sizeof(int) * args->nbDimsRequested);

    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt;
    int nbDims;

    cudnnStatus_t err = cudnnGetPoolingNdDescriptor(
        args->poolingDesc,
        args->nbDimsRequested,
        &mode,
        &maxpoolingNanOpt,
        &nbDims,
        windowDimA,
        paddingA,
        strideA
    );

    uint32_t res_len = sizeof(cudnnGetPoolingNdDescriptorResponse) + sizeof(int) * nbDims * 3;
    auto res = (cudnnGetPoolingNdDescriptorResponse *) malloc(res_len);

    res->err = err;
    res->mode = mode;
    res->maxpoolingNanOpt = maxpoolingNanOpt;
    res->nbDims = nbDims;
    memcpy(res->windowDimA_paddingA_strideA, windowDimA, sizeof(int) * nbDims);
    memcpy(res->windowDimA_paddingA_strideA + sizeof(int) * nbDims, paddingA, sizeof(int) * nbDims);
    memcpy(res->windowDimA_paddingA_strideA + sizeof(int) * nbDims * 2, strideA, sizeof(int) * nbDims);

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnGetPoolingNdForwardOutputDim(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnGetPoolingNdForwardOutputDim");

    auto args = (struct cudnnGetPoolingNdForwardOutputDimArg *) __args;

    uint32_t res_len = sizeof(cudnnGetPoolingNdForwardOutputDimResponse) + sizeof(int) * args->nbDims;
    auto res = (cudnnGetPoolingNdForwardOutputDimResponse *) malloc(res_len);

    res->err = cudnnGetPoolingNdForwardOutputDim(
        args->poolingDesc,
        args->inputTensorDesc,
        args->nbDims,
        res->outputTensorDimA
    );

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnPoolingForward(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnPoolingForward");

    auto args = (struct cudnnPoolingForwardArg *) __args;

    cudnnStatus_t err = cudnnPoolingForward(
		args->handle,
        args->poolingDesc,
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
}

void TallyServer::handle_cublasSgemv_v2(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasSgemv_v2");

    auto args = (struct cublasSgemv_v2Arg *) __args;

    cublasStatus_t err = cublasSgemv_v2(
		args->handle,
        args->trans,
        args->m,
        args->n,
        &args->alpha,
        args->A,
        args->lda,
        args->x,
        args->incx,
        &args->beta,
        args->y,
        args->incy
    );

    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnLRNCrossChannelForward(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnLRNCrossChannelForward");

    auto args = (struct cudnnLRNCrossChannelForwardArg *) __args;

    cudnnStatus_t err = cudnnLRNCrossChannelForward(
		args->handle,
        args->normDesc,
        args->lrnMode,
        &(args->alpha),
        args->xDesc,
        args->x,
        &(args->beta),
        args->yDesc,
        args->y
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnSoftmaxForward(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnSoftmaxForward");

    auto args = (struct cudnnSoftmaxForwardArg *) __args;

    cudnnStatus_t err = cudnnSoftmaxForward(
        args->handle,
        args->algo,
        args->mode,
        &(args->alpha),
        args->xDesc,
        args->x,
        &(args->beta),
        args->yDesc,
        args->y
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnTransformTensor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnTransformTensor");

    auto args = (struct cudnnTransformTensorArg *) __args;

    cudnnStatus_t err = cudnnTransformTensor(
        args->handle,
        &(args->alpha),
        args->xDesc,
        args->x,
        &(args->beta),
        args->yDesc,
        args->y
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasSgemmEx(void *__args)
{
	TALLY_SPD_LOG("Received request: cublasSgemmEx");

    auto args = (struct cublasSgemmExArg *) __args;

    cublasStatus_t err = cublasSgemmEx(
        args->handle,
        args->transa,
        args->transb,
        args->m,
        args->n,
        args->k,
        &(args->alpha),
        args->A,
        args->Atype,
        args->lda,
        args->B,
        args->Btype,
        args->ldb,
        &(args->beta),
        args->C,
        args->Ctype,
        args->ldc
    );

    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnSetSeqDataDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnSetSeqDataDescriptor");

    auto args = (struct cudnnSetSeqDataDescriptorArg *) __args;

    cudnnStatus_t err = cudnnSetSeqDataDescriptor(
        args->seqDataDesc,
        args->dataType,
        args->nbDims,
        args->dimA,
        args->axes,
        args->seqLengthArraySize,
        args->seqLengthArray,
        const_cast<void *>(args->paddingFill)
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnGetSeqDataDescriptor(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnGetSeqDataDescriptor");

    auto args = (struct cudnnGetSeqDataDescriptorArg *) __args;

    cudnnDataType_t dataType;
    int nbDims;
    size_t seqLengthArraySize;
    int *dimA = (int *) malloc(sizeof(int) * args->nbDimsRequested);
    cudnnSeqDataAxis_t *axes = (cudnnSeqDataAxis_t *) malloc(sizeof(cudnnSeqDataAxis_t) * args->nbDimsRequested);
    int *seqLengthArray = (int *) malloc(sizeof(int) * args->seqLengthSizeRequested);

    cudnnStatus_t err = cudnnGetSeqDataDescriptor(
        args->seqDataDesc,
        &dataType,
        &nbDims,
        args->nbDimsRequested,
        dimA,
        axes,
        &seqLengthArraySize,
        args->seqLengthSizeRequested,
        seqLengthArray,
        const_cast<void *>(args->paddingFill)
    );

    uint32_t res_len = sizeof(cudnnGetSeqDataDescriptorResponse) + sizeof(int) * nbDims + sizeof(cudnnSeqDataAxis_t) * nbDims + sizeof(int) * seqLengthArraySize;
    auto res = (cudnnGetSeqDataDescriptorResponse *) malloc(res_len);

    res->err = err;
    res->dataType = dataType;
    res->nbDims = nbDims;
    res->seqLengthArraySize = seqLengthArraySize;

    memcpy(res->dimA_axes_seqLengthArray, dimA, sizeof(int) * nbDims);
    memcpy(res->dimA_axes_seqLengthArray + sizeof(int) * nbDims, axes, sizeof(cudnnSeqDataAxis_t) * nbDims);
    memcpy(res->dimA_axes_seqLengthArray + sizeof(int) * nbDims + sizeof(cudnnSeqDataAxis_t) * nbDims, seqLengthArray, sizeof(int) * seqLengthArraySize);

    while(!send_ipc->send((void *) res, res_len)) {
        send_ipc->wait_for_recv(1);
    }

    free(dimA);
    free(axes);
    free(seqLengthArray);
    free(res);
}

void TallyServer::handle_cudnnMultiHeadAttnForward(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnForward");

    auto args = (struct cudnnMultiHeadAttnForwardArg *) __args;

    cudnnStatus_t err = cudnnMultiHeadAttnForward(
        args->handle,
        args->attnDesc,
        args->currIdx,
        args->loWinIdx_hiWinIdx,
        args->loWinIdx_hiWinIdx + args->winIdxLen,
        args->devSeqLengthsQO,
        args->devSeqLengthsKV,
        args->qDesc,
        args->queries,
        args->residuals,
        args->kDesc,
        args->keys,
        args->vDesc,
        args->values,
        args->oDesc,
        args->out,
        args->weightSizeInBytes,
        args->weights,
        args->workSpaceSizeInBytes,
        args->workSpace,
        args->reserveSpaceSizeInBytes,
        args->reserveSpace
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnMultiHeadAttnBackwardData(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnBackwardData");

    auto args = (struct cudnnMultiHeadAttnBackwardDataArg *) __args;

    cudnnStatus_t err = cudnnMultiHeadAttnBackwardData(
        args->handle,
        args->attnDesc,
        args->loWinIdx_hiWinIdx,
        args->loWinIdx_hiWinIdx + args->winIdxLen,
        args->devSeqLengthsDQDO,
        args->devSeqLengthsDKDV,
        args->doDesc,
        args->dout,
        args->dqDesc,
        args->dqueries,
        args->queries,
        args->dkDesc,
        args->dkeys,
        args->keys,
        args->dvDesc,
        args->dvalues,
        args->values,
        args->weightSizeInBytes,
        args->weights,
        args->workSpaceSizeInBytes,
        args->workSpace,
        args->reserveSpaceSizeInBytes,
        args->reserveSpace
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnMultiHeadAttnBackwardWeights(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnBackwardWeights");

    auto args = (struct cudnnMultiHeadAttnBackwardWeightsArg *) __args;
    cudnnStatus_t err = cudnnMultiHeadAttnBackwardWeights(
		args->handle,
		args->attnDesc,
		args->addGrad,
		args->qDesc,
		args->queries,
		args->kDesc,
		args->keys,
		args->vDesc,
		args->values,
		args->doDesc,
		args->dout,
		args->weightSizeInBytes,
		args->weights,
		args->dweights,
		args->workSpaceSizeInBytes,
		args->workSpace,
		args->reserveSpaceSizeInBytes,
		args->reserveSpace
    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudnnReorderFilterAndBias(void *__args)
{
	TALLY_SPD_LOG("Received request: cudnnReorderFilterAndBias");

    auto args = (struct cudnnReorderFilterAndBiasArg *) __args;
    cudnnStatus_t err = cudnnReorderFilterAndBias(
		args->handle,
		args->filterDesc,
		args->reorderType,
		args->filterData,
		args->reorderedFilterData,
		args->reorderBias,
		args->biasData,
		args->reorderedBiasData

    );

    while(!send_ipc->send((void *) &err, sizeof(cudnnStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}