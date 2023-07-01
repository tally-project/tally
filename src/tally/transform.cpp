#include <string>
#include <map>
#include <vector>
#include <regex>
#include <unordered_map>

#include <cuda.h>

#include <tally/util.h>
#include <tally/transform.h>
#include <tally/generated/cuda_api.h>

#include <boost/timer/progress_display.hpp>
#include <boost/regex.hpp>

std::unique_ptr<CubinCache> CubinCache::cache = std::make_unique<CubinCache>();
std::unique_ptr<Transform> Transform::tracer = std::make_unique<Transform>();

LaunchConfig LaunchConfig::tune(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    LaunchConfig best_config;
    std::vector<LaunchConfig> candidates;

    float best_time_ms = std::numeric_limits<float>::max();
    float time_ms;

    std::cout << "[Profile result]" <<std::endl;
    std::cout << "\tKernel:" << Transform::tracer->host_func_to_kernel_name_map[func] << std::endl;

    // default config - use_original=true
    LaunchConfig base_config;
    base_config.launch(func, gridDim, blockDim, args, sharedMem, stream, true, &time_ms);
    std::cout << "\tBaseline: Time: " << time_ms << std::endl;

    // some sliced configs
    for (uint32_t _threads_per_block : { 129560, 161280, 174080, 184320, 196608 }) {
        for (bool _use_cuda_graph : { true, false }) {
            LaunchConfig config(false, true, false, _use_cuda_graph, _threads_per_block);
            candidates.push_back(config);
        }
    }

    uint32_t num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(_num_blocks_per_sm * num_threads_per_block < 1580) {
        LaunchConfig config(false, false, true, false, 0, _num_blocks_per_sm);
        candidates.push_back(config);
        _num_blocks_per_sm++;
    }
    
    for (auto &config : candidates) {
        config.launch(func, gridDim, blockDim, args, sharedMem, stream, true, &time_ms);
        std::cout << "\t" << config << " Time: " << time_ms << std::endl;
        if (time_ms < best_time_ms) {
            best_config = config;
            best_time_ms = time_ms;
        }
    }

    return best_config;
}

cudaError_t LaunchConfig::launch(
    const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream,
    bool run_profile, float *elapsed_time_ms)
{
    cudaEvent_t _start, _stop;

    if (use_original) {

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaDeviceSynchronize();

            cudaEventRecord(_start);
        }

        auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
        }

        return err;
    } else if (use_sliced) {

        uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;

        auto cu_func = Transform::tracer->sliced_kernel_map[func].first;
        auto num_args = Transform::tracer->sliced_kernel_map[func].second;

        assert(cu_func);

        CudaGraphCall *cuda_graph_call = nullptr;

        cudaStream_t _stream;

        if (use_cuda_graph) {

            // Try to use cuda graph first 
            for (auto *call : Transform::tracer->cuda_graph_vec) {
                if (call->equals(func, args, num_args - 1, gridDim, blockDim)) {
                    cuda_graph_call = call;
                    break;
                }
            }

            if (!cuda_graph_call) {
                // Otherwise, construct one
                cuda_graph_call = new CudaGraphCall(func, args, num_args - 1, gridDim, blockDim);
                Transform::tracer->cuda_graph_vec.push_back(cuda_graph_call);
            }

            _stream = Transform::tracer->stream;
        } else {
            _stream = stream;
        }

        dim3 new_grid_dim;
        dim3 blockOffset(0, 0, 0);

        uint32_t num_blocks = (threads_per_slice + threads_per_block - 1) / threads_per_block;
        if (num_blocks <= gridDim.x) {
            new_grid_dim = dim3(num_blocks, 1, 1);
        } else {
            uint32_t num_blocks_y = (num_blocks + gridDim.x - 1) / gridDim.x;
            if (num_blocks_y <= gridDim.y) {
                new_grid_dim = dim3(gridDim.x, num_blocks_y, 1);
            } else {
                uint32_t num_blocks_z = (num_blocks_y + gridDim.y - 1) / gridDim.y;
                new_grid_dim = dim3(gridDim.x, gridDim.y, std::min(num_blocks_z, gridDim.z));
            }
        }

        if (use_cuda_graph) {
            cudaStreamBeginCapture(Transform::tracer->stream, cudaStreamCaptureModeGlobal);
        } else {

            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaDeviceSynchronize();

                cudaEventRecord(_start);
            }
        }

        CUresult res;
        while (blockOffset.x < gridDim.x && blockOffset.y < gridDim.y && blockOffset.z < gridDim.z) {

            void *KernelParams[num_args];
            for (size_t i = 0; i < num_args - 1; i++) {
                KernelParams[i] = args[i];
            }
            KernelParams[num_args - 1] = &blockOffset;

            // This ensure that you won't go over the original grid size
            dim3 curr_grid_dim (
                std::min(gridDim.x - blockOffset.x, new_grid_dim.x),
                std::min(gridDim.y - blockOffset.y, new_grid_dim.y),
                std::min(gridDim.z - blockOffset.z, new_grid_dim.z)
            );

            res = lcuLaunchKernel(cu_func, curr_grid_dim.x, curr_grid_dim.y, curr_grid_dim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, _stream, KernelParams, NULL);

            if (res != CUDA_SUCCESS) {
                std::cerr << "Encountering res != CUDA_SUCCESS" << std::endl;
                return cudaErrorInvalidValue;
            }

            blockOffset.x += new_grid_dim.x;

            if (blockOffset.x >= gridDim.x) {
                blockOffset.x = 0;
                blockOffset.y += new_grid_dim.y;

                if (blockOffset.y >= gridDim.y) {
                    blockOffset.y = 0;
                    blockOffset.z += new_grid_dim.z;
                }
            }
        }

        if (use_cuda_graph) {
            cudaStreamEndCapture(Transform::tracer->stream, &(cuda_graph_call->graph));

            if (!cuda_graph_call->instantiated) {
                cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                cuda_graph_call->instantiated = true;
            } else {
                // graph already exists; try to apply changes
                cudaGraphExecUpdateResult update;

                if (cudaGraphExecUpdate(cuda_graph_call->instance, cuda_graph_call->graph, NULL, &update) != cudaSuccess) 
                {
                    cudaGraphExecDestroy(cuda_graph_call->instance);
                    cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                }
            }

            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaDeviceSynchronize();

                cudaEventRecord(_start);
            }

            auto res = cudaGraphLaunch(cuda_graph_call->instance, stream);

            if (run_profile) {
                cudaEventRecord(_stop);
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            }

            return res;
        } else {

            if (run_profile) {
                cudaEventRecord(_stop);
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            }

            return cudaSuccess;
        }
    } else if (use_ptb) {
        auto cu_func = Transform::tracer->ptb_kernel_map[func].first;
        auto num_args = Transform::tracer->ptb_kernel_map[func].second;

        assert(cu_func);

        // Depend on number of PTBs/SM
        dim3 PTB_grid_dim(82 * num_blocks_per_sm);

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 1; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 1] = &gridDim;

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaDeviceSynchronize();

            cudaEventRecord(_start);
        }

        auto res = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
        }


        if (res != CUDA_SUCCESS) {
            std::cerr << "Encountering res != CUDA_SUCCESS" << std::endl;
            return cudaErrorInvalidValue;
        }

        return cudaSuccess;
        
    } else {
        throw std::runtime_error("Invalid launch config.");
    }
}

void write_binary_to_file(std::string path, const char* data, uint32_t size)
{
    std::ofstream file(path, std::ios::binary); // Open the file in binary mode
    file.write(data, size);
    file.close();
}

void write_str_to_file(std::string path, std::string str)
{
    std::ofstream file(path);
    file << str;
    file.close();
}

/* Extract all ptx code from cubin file, 
   return a vector of the generated file names */
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path)
{
    exec("cuobjdump -xptx all " + cubin_path);
    auto output = exec("cuobjdump " + cubin_path + " -lptx");

    std::stringstream ss(output.first);
    std::vector<std::string> names;
    std::string line;

    while (std::getline(ss, line, '\n')) {
        if (containsSubstring(line, ".ptx")) {
            auto split_str = splitOnce(line, ":");
            auto ptx_file_name = strip(split_str.second);
            names.push_back(ptx_file_name);
        }
    }

    return names;
}

std::string gen_ptb_ptx(std::string ptx_path)
{
    std::cout << "Generating PTB version of " << ptx_path << std::endl;
    std::ifstream t(ptx_path);
    if (!t.is_open()) {
        std::cerr << ptx_path << " not found." << std::endl;
        return "";
    }
    std::string ptx_code_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::stringstream ss(ptx_code_str);
    std::string line;
    boost::smatch matches;

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    boost::regex pred_reg_decl_pattern("\\.reg \\.pred %p<(\\d+)>;");
    boost::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    boost::regex kernel_param_pattern("^placeholder$");

    std::string PTB_RETURN_BLOCK_NAME = "L__PTB_RETURN";
    std::string PTB_LOOP_BLOCK_NAME = "L__PTB_LOOP";

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    uint32_t num_pred_regs = 0;
    bool use_block_idx_xyz[3] = { false, false, false };
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    std::string kernel_name;

    uint32_t num_additional_b32 = 15;
    uint32_t num_additional_pred_regs = 2;

    std::string origGridDim_x_reg = "%r" + std::to_string(num_b32_regs);
    std::string origGridDim_y_reg = "%r" + std::to_string(num_b32_regs + 1);
    std::string origGridDim_z_reg = "%r" + std::to_string(num_b32_regs + 2);
    std::string threadIdx_x_reg = "%r" + std::to_string(num_b32_regs + 3);
    std::string blockDim_x_reg = "%r" + std::to_string(num_b32_regs + 4);
    std::string gridDim_x_reg = "%r" + std::to_string(num_b32_regs + 5);
    std::string xy_tbs_reg = "%r" + std::to_string(num_b32_regs + 6);
    std::string num_thread_blocks_reg = "%r" + std::to_string(num_b32_regs + 7);
    std::string tb_idx_reg = "%r" + std::to_string(num_b32_regs + 8);
    std::string newBlockIdx_z_reg = "%r" + std::to_string(num_b32_regs + 9);
    std::string newBlockIdx_z_mul_xy_tbs_reg = "%r" + std::to_string(num_b32_regs + 10);
    std::string tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg = "%r" + std::to_string(num_b32_regs + 11);
    std::string newBlockIdx_y_reg = "%r" + std::to_string(num_b32_regs + 12);
    std::string newBlockIdx_y_mul_origGridDim_x_reg = "%r" + std::to_string(num_b32_regs + 13);
    std::string newBlockIdx_x_reg = "%r" + std::to_string(num_b32_regs + 14);

    std::string tb_idx_ge_num_thread_blocks_reg = "%p" + std::to_string(num_pred_regs);
    std::string tb_idx_lt_num_thread_blocks_reg = "%p" + std::to_string(num_pred_regs + 1);

    std::string ptb_ptx_code = "";

    boost::timer::progress_display progress(ptx_code_str.size());
    while (std::getline(ss, line, '\n')) {

        progress += line.size() + 1;
        if (boost::regex_search(line, matches, kernel_name_pattern)) {
            record_kernel = true;
            kernel_name = matches[2];
            kernel_param_pattern = boost::regex("\\.param (.+) " + kernel_name + "_param_(\\d+)");
            num_params = 0;
            num_b32_regs = 0;
            num_pred_regs = 0;
            kernel_lines.clear();
        }
        
        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            ptb_ptx_code += line + "\n";
        }
        
        if (boost::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
            continue;
        }

        if (boost::regex_search(line, matches, b32_reg_decl_pattern)) {
            num_b32_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, pred_reg_decl_pattern)) {
            num_pred_regs = std::stoi(matches[1]);
            continue;
        }

        if (record_kernel && line == "ret;") {

            // Ignore such kernels for now!
            if (num_params == 0) {
                continue;
            }

            record_kernel = false;
            boost::regex last_param_pattern("\\.param (.+) " + kernel_name + "_param_" + std::to_string(num_params - 1));
            std::map<std::string, std::string> reg_replacement_rules;

            for (auto &kernel_line : kernel_lines) {

                if (boost::regex_search(kernel_line, matches, last_param_pattern)) {
                    ptb_ptx_code += kernel_line + ",\n";
                    ptb_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                    continue;
                }

                // Perform actions at the top
                if (kernel_line == "{") {

                    ptb_ptx_code += "{\n";
                    ptb_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";
                    ptb_ptx_code += ".reg .pred %p<" + std::to_string(num_pred_regs + num_additional_pred_regs) + ">;\n";

                    // Load origGridDim.x
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_x_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "];\n";
                    // Load origGridDim.y
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_y_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+4];\n";
                    // Load origGridDim.z
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_z_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+8];\n";

                    // threadIdx.x
                    ptb_ptx_code += "mov.u32 " + threadIdx_x_reg + ", %tid.x;\n";
                    // blockDim.x
                    ptb_ptx_code += "mov.u32 " + blockDim_x_reg + ", %ntid.x;\n";
                    // gridDim.x
                    ptb_ptx_code += "mov.u32 " + gridDim_x_reg + ", %nctaid.x;\n";

                    // xy_tbs = origGridDim.x * origGridDim.y
                    ptb_ptx_code += "mul.lo.s32 " + xy_tbs_reg + ", " + origGridDim_x_reg + ", " + origGridDim_y_reg + ";\n";
                    // num_thread_blocks = origGridDim.x * origGridDim.y * origGridDim.z
                    ptb_ptx_code += "mul.lo.s32 " + num_thread_blocks_reg + ", " + origGridDim_z_reg + ", " + xy_tbs_reg + ";\n";

                    // tb_idx = blockIdx.x
                    ptb_ptx_code += "mov.u32 " + tb_idx_reg + ", %ctaid.x;\n";
                    // tb_idx >= num_thread_blocks
                    ptb_ptx_code += "setp.ge.u32 " + tb_idx_ge_num_thread_blocks_reg + ", " + tb_idx_reg + ", " + num_thread_blocks_reg + ";\n";
                    // branch to return if tb_idx >= num_thread_blocks
                    ptb_ptx_code += "@" + tb_idx_ge_num_thread_blocks_reg + " bra $" + PTB_RETURN_BLOCK_NAME + ";\n\n";

                    ptb_ptx_code += "$" + PTB_LOOP_BLOCK_NAME + ":\n";
                    
                    // newBlockIdx.z
                    ptb_ptx_code += "div.u32 " + newBlockIdx_z_reg + ", " + tb_idx_reg + ", " + xy_tbs_reg + ";\n";
                    // newBlockIdx.z * xy_tbs
                    ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                    // tb_idx - newBlockIdx.z * xy_tbs
                    ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + tb_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                    // newBlockIdx.y
                    ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + origGridDim_x_reg + ";\n";
                    // newBlockIdx.y * origGridDim.x
                    ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_y_mul_origGridDim_x_reg + ", " + newBlockIdx_y_reg + ", " + origGridDim_x_reg + ";\n";
                    // newBlockIdx.x
                    ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";

                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, pred_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, block_idx_pattern)) {
                    std::string block_idx_match_dim = matches[2];
                    uint32_t block_idx_match_reg = std::stoi(matches[1]);
   
                    if (block_idx_match_dim == "x") {
                        reg_replacement_rules["%r" + std::to_string(block_idx_match_reg) + "(?!\\d)"] = newBlockIdx_x_reg;
                    } else if (block_idx_match_dim == "y") {
                        reg_replacement_rules["%r" + std::to_string(block_idx_match_reg) + "(?!\\d)"] = newBlockIdx_y_reg;
                    } else if (block_idx_match_dim == "z") {
                        reg_replacement_rules["%r" + std::to_string(block_idx_match_reg) + "(?!\\d)"] = newBlockIdx_z_reg;
                    }
                    continue;
                }
            
                // instead of return, now in a loop
                if (kernel_line == "ret;") {

                    // tb_idx += gridDim.x
                    ptb_ptx_code += "add.s32 " + tb_idx_reg + ", " + tb_idx_reg + ", " + gridDim_x_reg + ";\n";
                    // tb_idx < num_thread_blocks
                    ptb_ptx_code += "setp.lt.u32 " + tb_idx_lt_num_thread_blocks_reg + ", " + tb_idx_reg + ", " + num_thread_blocks_reg + ";\n";
                    // branch to L__PTB_LOOP if tb_idx < num_thread_blocks
                    ptb_ptx_code += "@" + tb_idx_lt_num_thread_blocks_reg + " bra $" + PTB_LOOP_BLOCK_NAME + ";\n\n";

                    // Add return block
                    ptb_ptx_code += "$" + PTB_RETURN_BLOCK_NAME + ":\n";
                    ptb_ptx_code += "ret;\n";

                    continue;
                }

                for (const auto& pair : reg_replacement_rules) {
                    boost::regex pattern(pair.first);
                    kernel_line = boost::regex_replace(kernel_line, pattern, pair.second);
                }

                ptb_ptx_code += kernel_line + "\n";
            }
        }
    }

    return ptb_ptx_code;
}

std::string gen_sliced_ptx(std::string ptx_path)
{
    std::cout << "Generating sliced version of " << ptx_path << std::endl;
    std::ifstream t(ptx_path);
    if (!t.is_open()) {
        std::cerr << ptx_path << " not found." << std::endl;
        return "";
    }
    std::string ptx_code_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::stringstream ss(ptx_code_str);
    std::string line;
    
    std::string ptb_ptx_code = "";

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    boost::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    boost::regex kernel_param_pattern("^placeholder$");

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    bool use_block_idx_xyz[3] = { false, false, false };
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    std::string kernel_name;

    boost::timer::progress_display progress(ptx_code_str.size());

    while (std::getline(ss, line, '\n')) {
        progress += line.size() + 1;

        boost::smatch matches;
        if (boost::regex_search(line, matches, kernel_name_pattern)) {
            record_kernel = true;
            kernel_name = matches[2];

            kernel_param_pattern = boost::regex("\\.param (.+) " + kernel_name + "_param_(\\d+)");
            num_params = 0;
            num_b32_regs = 0;
            kernel_lines.clear();
            
            // reset to false 
            memset(use_block_idx_xyz, 0, 3);

            kernel_lines.push_back(line);
            continue;
        }

        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            ptb_ptx_code += line + "\n";
        }

        if (boost::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
            continue;
        }

        if (boost::regex_search(line, matches, b32_reg_decl_pattern)) {
            num_b32_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, block_idx_pattern)) {
            std::string block_idx_match_dim = matches[2];
            if (block_idx_match_dim == "x") {
                use_block_idx_xyz[0] = true;
            } else if (block_idx_match_dim == "y") {
                use_block_idx_xyz[1] = true;
            } else if (block_idx_match_dim == "z") {
                use_block_idx_xyz[2] = true;
            }
            continue;
        }

        if (record_kernel && line == "ret;") {
            record_kernel = false;

            boost::regex last_param_pattern("\\.param (.+) " + kernel_name + "_param_" + std::to_string(num_params - 1));
            uint32_t num_additional_b32 = 0;
    
            uint32_t block_offset_xyz_reg[3];
            uint32_t new_block_idx_xyz_reg[3];

            uint32_t curr_reg = num_b32_regs;

            for (size_t i = 0; i < 3; i++) {
                if (use_block_idx_xyz[i]) {
                    num_additional_b32 += 2;
                    block_offset_xyz_reg[i] = curr_reg;
                    new_block_idx_xyz_reg[i] = curr_reg + 1;
                    curr_reg += 2;
                }
            }

            std::map<std::string, std::string> reg_replacement_rules;
    
            for (auto &kernel_line : kernel_lines) {
                
                if (boost::regex_search(kernel_line, matches, last_param_pattern)) {
                    ptb_ptx_code += kernel_line + ",\n";
                    ptb_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    ptb_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, block_idx_pattern)) {
                    std::string block_idx_match_dim = matches[2];
                    uint32_t block_idx_match_reg = std::stoi(matches[1]);

                    int32_t idx = -1;
                    if (block_idx_match_dim == "x") {
                        idx = 0;
                    } else if (block_idx_match_dim == "y") {
                        idx = 1;
                    } else if (block_idx_match_dim == "z") {
                        idx = 2;
                    }

                    ptb_ptx_code += kernel_line + "\n";
                    ptb_ptx_code += "ld.param.u32 %r" + std::to_string(block_offset_xyz_reg[idx]) + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+" + std::to_string(idx * 4) + "];\n";
                    ptb_ptx_code += "add.u32 %r" + std::to_string(new_block_idx_xyz_reg[idx]) + ", %r" + std::to_string(block_idx_match_reg) + ", %r" + std::to_string(block_offset_xyz_reg[idx]) + ";\n";
                    reg_replacement_rules["%r" + std::to_string(block_idx_match_reg) + "(?!\\d)"] = "%r" + std::to_string(new_block_idx_xyz_reg[idx]);

                    continue;
                }

                for (const auto& pair : reg_replacement_rules) {
                    boost::regex pattern(pair.first);
                    kernel_line = boost::regex_replace(kernel_line, pattern, pair.second);
                }

                ptb_ptx_code += kernel_line + "\n";
            }
            continue;
        }
    }

    // std::cout << "Processing done" << std::endl;

    return ptb_ptx_code;
}

std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> register_kernels_from_ptx_fatbin(
    std::vector<std::pair<std::string, std::string>> &ptx_fatbin_strs,
    std::map<std::string, const void *> &kernel_name_map
)
{
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> kernel_map;

    for (auto &ptx_fatbin_pair : ptx_fatbin_strs) {
        
        auto &ptx_str = ptx_fatbin_pair.first;
        auto &fatbin_str = ptx_fatbin_pair.second;

        CUmodule cudaModule;
        lcuModuleLoadDataEx(&cudaModule, fatbin_str.c_str(), 0, 0, 0);

        auto kernel_names_and_nparams = get_kernel_names_and_nparams_from_ptx(ptx_str);
        
        for (auto &name_and_nparams : kernel_names_and_nparams) {

            auto &kernel_name = name_and_nparams.first;
            auto num_params = name_and_nparams.second;
            auto host_func = kernel_name_map[kernel_name];

            CUfunction function;
            lcuModuleGetFunction(&function, cudaModule, kernel_name.c_str());

            kernel_map[host_func] = std::make_pair(function, num_params);
        }
    }

    return kernel_map;
}

std::vector<std::pair<std::string, uint32_t>> get_kernel_names_and_nparams_from_ptx(std::string &ptx_str)
{
    std::vector<std::pair<std::string, uint32_t>> kernel_names_and_nparams;

    std::stringstream ss(ptx_str);
    std::string line;

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex kernel_param_pattern("^placeholder$");
    uint32_t num_params = 0;
    std::string kernel_name;

    while (std::getline(ss, line, '\n')) {

        boost::smatch matches;
        if (boost::regex_search(line, matches, kernel_name_pattern)) {
            kernel_name = matches[2];
            kernel_param_pattern = boost::regex("\\.param (.+) " + kernel_name + "_param_(\\d+)");
            num_params = 0;
        }

        if (boost::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
        }

        if (line == "ret;") {
            kernel_names_and_nparams.push_back(std::make_pair(kernel_name, num_params));
        }
    }

    return kernel_names_and_nparams;
}

std::vector<std::pair<std::string, std::vector<uint32_t>>> get_kernel_names_and_param_sizes_from_elf(std::string elf_file_name)
{
    // key: func_name, val: [ <ordinal, size> ]
    using ordinal_size_pair = std::pair<uint32_t, uint32_t>;
    std::vector<std::pair<std::string, std::vector<uint32_t>>> kernel_names_and_param_sizes;

    std::ifstream elf_file(elf_file_name);

    std::string line;
    while (std::getline(elf_file, line)) {
        if (startsWith(line, ".nv.info.")) {
            std::string kernel_name = line.substr(9);
            std::vector<ordinal_size_pair> params_info;

            while (std::getline(elf_file, line)) {
                if (containsSubstring(line, "EIATTR_KPARAM_INFO")) {
                    
                } else if (containsSubstring(line, "Ordinal :")) {
                    auto split_by_ordinal = splitOnce(line, "Ordinal :");
                    auto split_by_offset = splitOnce(split_by_ordinal.second, "Offset  :");
                    auto split_by_size = splitOnce(split_by_offset.second, "Size    :");

                    auto ordinal_str = strip(split_by_offset.first);
                    auto size_str = strip(split_by_size.second);

                    uint32_t arg_ordinal = std::stoi(ordinal_str, nullptr, 16);
                    uint32_t arg_size = std::stoi(size_str, nullptr, 16);

                    params_info.push_back(std::make_pair(arg_ordinal, arg_size));

                } else if (line.empty()) {
                    break;
                }
            }

            // Sort by ordinal
            std::sort(
                params_info.begin(),
                params_info.end(),
                [](ordinal_size_pair a, ordinal_size_pair b) {
                    return a.first < b.first;
                }
            );

            // Store the size
            std::vector<uint32_t> param_sizes;

            for (auto &pair : params_info) {
                param_sizes.push_back(pair.second);
            }

            kernel_names_and_param_sizes.push_back(std::make_pair(kernel_name, param_sizes));
        }
    }    

    elf_file.close();

    return kernel_names_and_param_sizes;
}

std::string get_fatbin_str_from_ptx_str(std::string ptx_str)
{
    write_str_to_file("/tmp/output.ptx", ptx_str);
    auto res = exec("nvcc /tmp/output.ptx --fatbin -arch sm_86 -o /tmp/output.fatbin");

    if (res.second != 0) {
        throw std::runtime_error("Fail to compile PTX.");
    }

    std::ifstream ifs("/tmp/output.fatbin", std::ios::binary);
    auto fatbin_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    std::remove("/tmp/output.ptx");
    std::remove("/tmp/output.fatbin");

    return fatbin_str;
}