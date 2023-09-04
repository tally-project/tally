#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <sys/time.h>

template <typename scalar_t, typename accscalar_t>
__global__ void max_pool_forward_nchw(const int nthreads, const scalar_t* bottom_data,
    const int64_t channels, const int64_t height,
    const int64_t width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask) {

  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (int index=_i_n_d_e_x; _i_n_d_e_x < (nthreads); _i_n_d_e_x+=blockDim.x * gridDim.x, index=_i_n_d_e_x) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, static_cast<int>(height));
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, static_cast<int>(width));
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    accscalar_t maxval = -static_cast<float>(INFINITY);; // -Infinity
    int maxidx = hstart * width + wstart;
    const scalar_t* btm_data = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        scalar_t val = btm_data[h * width + w];
        if ((static_cast<accscalar_t>(val) > maxval) || std::isnan(val)) {
          maxidx = h * width + w;
          maxval = static_cast<accscalar_t>(val);
        }
      }
    }

    top_data[index] = static_cast<accscalar_t>(maxval);
    top_mask[index] = maxidx;
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void max_pool_forward_nchw_sliced(const int nthreads, const scalar_t* bottom_data,
    const int64_t channels, const int64_t height,
    const int64_t width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask, dim3 blockOffset) {

  int64_t _i_n_d_e_x = (blockIdx.x + blockOffset.x) * blockDim.x + threadIdx.x;
  for (int index=_i_n_d_e_x; _i_n_d_e_x < (nthreads); _i_n_d_e_x+=blockDim.x * gridDim.x, index=_i_n_d_e_x) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, static_cast<int>(height));
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, static_cast<int>(width));
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    accscalar_t maxval = -static_cast<float>(INFINITY);; // -Infinity
    int maxidx = hstart * width + wstart;
    const scalar_t* btm_data = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        scalar_t val = btm_data[h * width + w];
        if ((static_cast<accscalar_t>(val) > maxval) || std::isnan(val)) {
          maxidx = h * width + w;
          maxval = static_cast<accscalar_t>(val);
        }
      }
    }

    top_data[index] = static_cast<accscalar_t>(maxval);
    top_mask[index] = maxidx;
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void max_pool_forward_nchw_PTB(const int nthreads, const scalar_t* bottom_data,
    const int64_t channels, const int64_t height,
    const int64_t width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask, dim3 original_gridSize) {

    uint32_t num_thread_blocks = original_gridSize.x * original_gridSize.y * original_gridSize.z;
    uint32_t xy_tbs = original_gridSize.x * original_gridSize.y;
    dim3 newBlockIdx(0, 0, 0);

    for (int tb_idx = blockIdx.x; tb_idx < num_thread_blocks; tb_idx += gridDim.x) {

        newBlockIdx.z = tb_idx / xy_tbs;
        newBlockIdx.y = (tb_idx - newBlockIdx.z * xy_tbs) / original_gridSize.x;
        newBlockIdx.x = (tb_idx - newBlockIdx.z * xy_tbs) - newBlockIdx.y * original_gridSize.x;

        int64_t _i_n_d_e_x = newBlockIdx.x * blockDim.x + threadIdx.x;
        for (int index=_i_n_d_e_x; _i_n_d_e_x < (nthreads); _i_n_d_e_x+=blockDim.x * gridDim.x, index=_i_n_d_e_x) {
            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int c = (index / pooled_width / pooled_height) % channels;
            int n = index / pooled_width / pooled_height / channels;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, static_cast<int>(height));
            int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, static_cast<int>(width));
            while(hstart < 0)
            hstart += dilation_h;
            while(wstart < 0)
            wstart += dilation_w;
            accscalar_t maxval = -static_cast<float>(INFINITY);; // -Infinity
            int maxidx = hstart * width + wstart;
            const scalar_t* btm_data = bottom_data + (n * channels + c) * height * width;
            for (int h = hstart; h < hend; h += dilation_h) {
            for (int w = wstart; w < wend; w += dilation_w) {
                scalar_t val = btm_data[h * width + w];
                if ((static_cast<accscalar_t>(val) > maxval) || std::isnan(val)) {
                maxidx = h * width + w;
                maxval = static_cast<accscalar_t>(val);
                }
            }
            }

            top_data[index] = static_cast<accscalar_t>(maxval);
            top_mask[index] = maxidx;
        }
    }
}

__host__ void runMaxPool(uint32_t input_size, uint32_t output_size, float *input, float *output, int64_t *top_mask)
{
    // baseline
    float *d_input, *d_output;
    int64_t *d_top_mask;

    // PTB output
    float *d_output_ptb;
    int64_t *d_top_mask_ptb;
    float *h_output_ptb = new float[output_size];
    int64_t *h_top_mask_ptb = new int64_t[output_size];

    // Allocate device memory
    cudaMalloc((void **)&d_input, input_size * sizeof(float));
    cudaMalloc((void **)&d_output, output_size * sizeof(float));
    cudaMalloc((void **)&d_top_mask, output_size * sizeof(int64_t));
    cudaMalloc((void **)&d_output_ptb, output_size * sizeof(float));
    cudaMalloc((void **)&d_top_mask_ptb, output_size * sizeof(int64_t));

    // Copy input matrices from host to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim((output_size + 256 - 1) / 256);

    std::cout << "original grid_dim: " << grid_dim.x << " " << grid_dim.y << " " << grid_dim.z << std::endl;

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    max_pool_forward_nchw<float, float><<<grid_dim, block_dim, 0, cudaStreamDefault>>>(
            output_size, d_input, 3, 128, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, d_output, d_top_mask);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Original Kernel execution time: %.2f ms\n", milliseconds);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(top_mask, d_top_mask, output_size * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Depend on number of PTBs/SM
    dim3 PTB_grid_dim(82 * 4);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    max_pool_forward_nchw_PTB<float, float><<<PTB_grid_dim, block_dim, 0, cudaStreamDefault>>>(
            output_size, d_input, 3, 128, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, d_output_ptb, d_top_mask_ptb, grid_dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("PTB Kernel execution time: %.2f ms\n", milliseconds);

    cudaMemcpy(h_output_ptb, d_output_ptb, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_mask_ptb, d_top_mask_ptb, output_size * sizeof(int64_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < output_size; i++) {
        assert(h_output_ptb[i] == output[i]);
        assert(h_top_mask_ptb[i] == top_mask[i]);
    }

    dim3 new_grid_dim;
    dim3 blockOffset(0, 0, 0);
    uint32_t threads_per_slice = 196608;
    uint32_t threads_per_block = block_dim.x * block_dim.y * block_dim.z;

    uint32_t num_blocks = (threads_per_slice + threads_per_block - 1) / threads_per_block;
    if (num_blocks <= grid_dim.x) {
        new_grid_dim = dim3(num_blocks, 1, 1);
    } else {
        uint32_t num_blocks_y = (num_blocks + grid_dim.x - 1) / grid_dim.x;
        if (num_blocks_y <= grid_dim.y) {
            new_grid_dim = dim3(grid_dim.x, num_blocks_y, 1);
        } else {
            uint32_t num_blocks_z = (num_blocks_y + grid_dim.y - 1) / grid_dim.y;
            new_grid_dim = dim3(grid_dim.x, grid_dim.y, std::min(num_blocks_z, grid_dim.z));
        }
    }

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    while (blockOffset.x < grid_dim.x && blockOffset.y < grid_dim.y && blockOffset.z < grid_dim.z) {

        // This ensure that you won't go over the original grid size
        dim3 curr_grid_dim (
            std::min(grid_dim.x - blockOffset.x, new_grid_dim.x),
            std::min(grid_dim.y - blockOffset.y, new_grid_dim.y),
            std::min(grid_dim.z - blockOffset.z, new_grid_dim.z)
        );

        std::cout << "curr_grid_dim: " << curr_grid_dim.x << " " << curr_grid_dim.y << " " << curr_grid_dim.z << std::endl;

        max_pool_forward_nchw_sliced<float, float><<<curr_grid_dim, block_dim, 0, cudaStreamDefault>>>(
            output_size, d_input, 3, 128, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, d_output_ptb, d_top_mask_ptb, blockOffset);

        blockOffset.x += new_grid_dim.x;

        if (blockOffset.x >= grid_dim.x) {
            blockOffset.x = 0;
            blockOffset.y += new_grid_dim.y;

            if (blockOffset.y >= grid_dim.y) {
                blockOffset.y = 0;
                blockOffset.z += new_grid_dim.z;
            }
        }

        break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Sliced Kernel execution time: %.2f ms\n", milliseconds);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_top_mask);
    cudaFree(d_output_ptb);
    cudaFree(d_top_mask_ptb);

    free(h_output_ptb);
    free(h_top_mask_ptb);
}

int main()
{
    // bool ptb = false;
    uint32_t input_size = 1024 * 3 * 128 * 128;
    uint32_t output_size = 1024 * 3 * 64 * 64;
    float* input = new float[input_size];
    float* output = new float[output_size];
    int64_t* top_mask = new int64_t[output_size];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < input_size; ++i) {
        input[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    runMaxPool(input_size, output_size, input, output, top_mask);

    delete[] input;
    delete[] output;
    delete[] top_mask;

    return 0;
}
