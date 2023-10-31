
#ifndef TALLY_CUBLAS_TRACER_H
#define TALLY_CUBLAS_TRACER_H

#include <map>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <tally/log.h>

struct cublasCtx {

	cublasHandle_t handle;
	cublasMath_t mode = CUBLAS_DEFAULT_MATH;
	cudaStream_t stream;

};

class cublasTracer {

private:
	std::unordered_map<cublasHandle_t, cublasCtx> handle_map;

public:
	cublasTracer(){}
	~cublasTracer(){}

	void handle_cublasCreate_v2(cublasHandle_t handle) {
		cublasCtx ctx;
		ctx.handle = handle;
		handle_map[handle] = ctx;
	}

	void handle_cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
		auto &ctx = handle_map[handle];	
		ctx.stream = stream;
	}

	void handle_cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
		auto &ctx = handle_map[handle];	
		ctx.mode = mode;
	}

	void handle_cublasDestroy_v2(cublasHandle_t handle) {
		handle_map.erase(handle);	
	}

	cublasCtx get_cublasCtx(cublasHandle_t handle) {
		return handle_map[handle];
	}
};


#endif // TALLY_CUBLAS_TRACER_H
