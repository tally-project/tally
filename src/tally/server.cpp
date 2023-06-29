#include <cstring>

#include <tally/server.h>
#include <tally/cuda_api.h>
#include <tally/kernel_slice.h>
#include <tally/util.h>

TallyServer tally_server;

void TallyServer::handle_cudaMalloc(void *__args)
{
    auto args = (struct cudaMallocArg *) __args;

    // The client "supposedly" should remember this address
    // So not book-keeping it at the moment
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, args->size);

    struct cudaMallocResponse res { devPtr, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaMallocResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaFree(void *__args)
{
    auto args = (struct cudaFreeArg *) __args;

    cudaError_t err = cudaFree(args->devPtr);

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaMemcpy(void *__args)
{
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
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    while(!send_ipc->send((void *) res, res_size)) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaLaunchKernel(void *__args)
{
    auto args = (cudaLaunchKernelArg *) __args;
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    assert(lcudaLaunchKernel);

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

    auto err = lcudaLaunchKernel((const void *) kernel_server_addr, args->gridDim, args->blockDim, &__args_arr[0], args->sharedMem, cudaStreamDefault);

    while (!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_fatCubin(void *__args)
{
    auto args = (fatBinArg *) __args;
    magic = args->magic;
    version = args->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) args->data;
    fatBinSize = fbh->headerSize + fbh->fatSize;

    fatbin_data = (unsigned long long *) malloc(fatBinSize);
    memcpy(fatbin_data, args->data, fatBinSize);
}

void TallyServer::handle_register_kernel(void *__args)
{
    auto args = (struct registerKernelArg *) __args;
    std::string kernel_name {args->data, args->kernel_func_len};
    register_queue.push_back( std::make_pair(args->host_func, kernel_name));
}

void TallyServer::handle_fatCubin_end()
{
	assert(l__cudaRegisterFatBinaryEnd);
    assert(l__cudaRegisterFatBinary);
    assert(l__cudaRegisterFunction);

    const my__fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) = { magic, version, fatbin_data, 0 };
    void **handle = l__cudaRegisterFatBinary((void *)&__fatDeviceText);

    void *kernel_server_addr;

    for (auto &kernel_pair : register_queue) {
        auto &client_addr = kernel_pair.first;
        auto &kernel_name = kernel_pair.second;

        // allocate an address for the purpose
        // TODO: free this at when?
        kernel_server_addr = malloc(8);

        // Bookkeeping the mapping between clinet kernel addr and server kernel addr
        _kernel_name_to_addr[kernel_name] = kernel_server_addr;
        _kernel_client_addr_mapping[client_addr] = kernel_server_addr;
        l__cudaRegisterFunction(handle, (const char*) kernel_server_addr, (char *)kernel_name.c_str(), kernel_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
    }

    l__cudaRegisterFatBinaryEnd(handle);

    std::ofstream cubin_file("/tmp/tmp.cubin", std::ios::binary); // Open the file in binary mode
    cubin_file.write(reinterpret_cast<const char*>(fatbin_data), fatBinSize);
    cubin_file.close();

    const char* command = "cuobjdump /tmp/tmp.cubin -elf > /tmp/tmp_cubin.elf";
    system(command);

    std::string elf_filename = "/tmp/tmp_cubin.elf";
    auto kernel_names_and_param_sizes = get_kernel_names_and_param_sizes_from_elf(elf_filename);

    for (auto &pair : kernel_names_and_param_sizes) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;

        _kernel_addr_to_args[_kernel_name_to_addr[kernel_name]] = param_sizes;
    }

    // For some reason, must call one cuda api call here. Otherwise it won't run.
    int *arr;
    cudaMalloc((void**)&arr, sizeof(int));
    cudaFree(arr);
}



