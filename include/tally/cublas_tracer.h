
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
	void* workspace = nullptr;
	size_t workspaceSizeInBytes = 0;
};

struct cublasLtCtx {
	cublasLtHandle_t handle;
};

struct cublasLtMatmulDescCtx {
	cublasLtMatmulDesc_t handle;
	cublasComputeType_t computeType;
	cudaDataType_t scaleType;

	cublasLtPointerMode_t cublaslt_matmul_desc_pointer_mode = CUBLASLT_POINTER_MODE_HOST;
	cublasOperation_t cublaslt_matmul_desc_transa = CUBLAS_OP_N;
	cublasOperation_t cublaslt_matmul_desc_transb = CUBLAS_OP_N;
	cublasLtEpilogue_t cublaslt_matmul_desc_epilogue = CUBLASLT_EPILOGUE_DEFAULT;
	void *cublaslt_matmul_desc_bias_pointer = NULL;
};

struct cublasLtMatrixLayoutCtx {
	cublasLtMatrixLayout_t  handle;
	cudaDataType  type;
	uint64_t  rows;
	uint64_t  cols;
	int64_t  ld;
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

	void handle_cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
		auto &ctx = handle_map[handle];	
		ctx.mode = mode;
	}

	void handle_cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
		auto &ctx = handle_map[handle];	
		ctx.stream = stream;
		ctx.workspace = nullptr;
		ctx.workspaceSizeInBytes = 0;
	}

	void handle_cublasSetWorkspace_v2(cublasHandle_t handle, void*  workspace, size_t  workspaceSizeInBytes) {
		auto &ctx = handle_map[handle];
		ctx.workspace = workspace;
		ctx.workspaceSizeInBytes = workspaceSizeInBytes;
	}

	void handle_cublasDestroy_v2(cublasHandle_t handle) {
		handle_map.erase(handle);	
	}

	void handle_cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
		auto &ctx = handle_map[handle];
		*mode = ctx.mode;
	}

	cublasCtx get_cublasCtx(cublasHandle_t handle) {
		return handle_map[handle];
	}
};

class cublasLtTracer {
private:
	std::unordered_map<cublasLtHandle_t, cublasLtCtx> handle_map;

public:
	cublasLtTracer(){}
	~cublasLtTracer(){}

	void handle_cublasLtCreate(cublasLtHandle_t handle) {
		cublasLtCtx ctx;
		ctx.handle = handle;
		handle_map[handle] = ctx;
	}

	void handle_cublasLtDestroy(cublasLtHandle_t handle) {
		handle_map.erase(handle);
	}

	cublasLtCtx get_cublasLtCtx(cublasLtHandle_t handle) {
		return handle_map[handle];
	}
};

class cublasLtMatmulDescTracer {
private:
	std::unordered_map<cublasLtMatmulDesc_t, cublasLtMatmulDescCtx> handle_map;

public:
	cublasLtMatmulDescTracer(){}
	~cublasLtMatmulDescTracer(){}

	void handle_cublasLtMatmulDescCreate(cublasLtMatmulDesc_t handle, cublasComputeType_t computeType, cudaDataType_t scaleType) {
		cublasLtMatmulDescCtx ctx;
		ctx.handle = handle;
		ctx.computeType = computeType;
		ctx.scaleType = scaleType;
		handle_map[handle] = ctx;

		// std::cout << "tracer: cublasLtMatmulDescCreate" << std::endl;
		// std::cout << "computeType: " << (int) computeType << std::endl;
		// std::cout << "scaleType: " << (int) scaleType << std::endl;
	}

	void handle_cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t handle, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes) {
		
		// std::cout << "tracer: cublasLtMatmulDescSetAttribute" << std::endl;

		auto &ctx = handle_map[handle];
		if (attr == CUBLASLT_MATMUL_DESC_TRANSA) {
			assert(sizeInBytes == sizeof(cublasOperation_t));
			ctx.cublaslt_matmul_desc_transa = *((cublasOperation_t *) buf);
			// std::cout << "CUBLASLT_MATMUL_DESC_TRANSA" << std::endl;
		} else if (attr == CUBLASLT_MATMUL_DESC_TRANSB) {
			assert(sizeInBytes == sizeof(cublasOperation_t));
			ctx.cublaslt_matmul_desc_transb = *((cublasOperation_t *) buf);
			// std::cout << "CUBLASLT_MATMUL_DESC_TRANSB" << std::endl;
		} else if (attr == CUBLASLT_MATMUL_DESC_EPILOGUE) {
			assert(sizeInBytes == sizeof(cublasLtEpilogue_t));
			ctx.cublaslt_matmul_desc_epilogue = *((cublasLtEpilogue_t *) buf);
			// std::cout << "CUBLASLT_MATMUL_DESC_EPILOGUE" << std::endl;
			// std::cout << "ctx.cublaslt_matmul_desc_epilogue: " << (int) ctx.cublaslt_matmul_desc_epilogue << std::endl;
		} else if (attr == CUBLASLT_MATMUL_DESC_BIAS_POINTER) {
			assert(sizeInBytes == sizeof(void *));
			ctx.cublaslt_matmul_desc_bias_pointer = *((void **) buf);
			// std::cout << "CUBLASLT_MATMUL_DESC_BIAS_POINTER" << std::endl;
		} else if (attr == CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET) {
			// Ignore this
		} else {
			throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublasLtMatmulDescAttributes_t " + std::to_string((int) attr) + " is yet handled.");
		}
	}

	void handle_cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t handle) {
		handle_map.erase(handle);
	}

	cublasLtMatmulDescCtx get_cublasLtMatmulDescCtx(cublasLtMatmulDesc_t handle) {
		return handle_map[handle];
	}
};

class cublasLtMatrixLayoutTracer {
private:
	std::unordered_map<cublasLtMatrixLayout_t, cublasLtMatrixLayoutCtx> handle_map;

public:
	cublasLtMatrixLayoutTracer(){}
	~cublasLtMatrixLayoutTracer(){}

	void handle_cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t  handle, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld) {
		
		// std::cout << "tracer: cublasLtMatrixLayoutCreate" << std::endl;
		// std::cout << "type: " << (int) type << std::endl;
		// std::cout << "rows: " << (int) rows << std::endl;
		// std::cout << "cols: " << (int) cols << std::endl;
		// std::cout << "ld: " << (int) ld << std::endl;
		
		cublasLtMatrixLayoutCtx ctx;
		ctx.handle = handle;
		ctx.type = type;
		ctx.rows = rows;
		ctx.cols = cols;
		ctx.ld = ld;
		handle_map[handle] = ctx;
	}

	void handle_cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t  handle) {
		handle_map.erase(handle);	
	}

	cublasLtMatrixLayoutCtx get_cublasLtMatrixLayoutCtx(cublasLtMatrixLayout_t handle) {
		return handle_map[handle];
	}
};

#endif // TALLY_CUBLAS_TRACER_H
