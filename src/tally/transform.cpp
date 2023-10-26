#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>

#include <tally/env.h>
#include <tally/util.h>
#include <tally/transform.h>

#include <boost/timer/progress_display.hpp>
#include <boost/regex.hpp>

// Generating preemptive using no shmem PTB version of a PTX file
std::string gen_preemptive_ptb_ptx(std::string &kernel_ptx_str)
{
    std::stringstream ss(kernel_ptx_str);
    std::string line;
    boost::smatch matches;

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex b16_reg_decl_pattern("\\.reg \\.b16 %rs<(\\d+)>;");
    boost::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    boost::regex b64_reg_decl_pattern("\\.reg \\.b64 %rd<(\\d+)>;");
    boost::regex pred_reg_decl_pattern("\\.reg \\.pred %p<(\\d+)>;");
    boost::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    boost::regex kernel_param_pattern("^\\.param");

    std::string PTB_MAIN_BLOCK_NAME = "L__PTB_MAIN";
    std::string PTB_CHECK_LEADER_BLOCK_NAME = "L__PTB_CHECK_LEADER";
    std::string PTB_CHECK_INDEX_BLOCK_NAME = "L__PTB_CHECK_INDEX";
    std::string PTB_ATOMIC_ADD_BLOCK_NAME = "L__PTB_ATOMIC_ADD";
    std::string PTB_RETURN_BLOCK_NAME = "L__PTB_RETURN";

    uint32_t num_params = 0;
    uint32_t num_b16_regs = 0;
    uint32_t num_b32_regs = 0;
    uint32_t num_b64_regs = 0;
    uint32_t num_pred_regs = 0;
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    int32_t brace_counter = 0;
    int32_t brace_encountered = false;
    std::string kernel_name;

    bool use_BlockIdx_x = false;
    bool use_BlockIdx_y = false;
    bool use_BlockIdx_z = false;

    uint32_t num_additional_b16 = 0;
    uint32_t num_additional_b32 = 0;
    uint32_t num_additional_b64 = 0;
    uint32_t num_additional_pred_regs = 0;

    auto allocate_new_b16_reg = [&num_b16_regs, &num_additional_b16]() {
        uint32_t new_b16_reg = num_b16_regs + num_additional_b16;
        num_additional_b16++;
        return new_b16_reg;
    };

    auto allocate_new_b32_reg = [&num_b32_regs, &num_additional_b32]() {
        uint32_t new_b32_reg = num_b32_regs + num_additional_b32;
        num_additional_b32++;
        return new_b32_reg;
    };

    auto allocate_new_b64_reg = [&num_b64_regs, &num_additional_b64]() {
        uint32_t new_b64_reg = num_b64_regs + num_additional_b64;
        num_additional_b64++;
        return new_b64_reg;
    };

    auto allocate_new_pred_reg = [&num_pred_regs, &num_additional_pred_regs]() {
        uint32_t new_pred_reg = num_pred_regs + num_additional_pred_regs;
        num_additional_pred_regs++;
        return new_pred_reg;
    };

    std::string ptb_ptx_code = "";

    // boost::timer::progress_display progress(ptx_code_str.size());
    while (std::getline(ss, line, '\n')) {
        // progress += line.size() + 1;
        if (boost::regex_search(line, matches, kernel_name_pattern)) {
            record_kernel = true;
            kernel_name = matches[2];
            num_params = 0;
            num_b16_regs = 0;
            num_b32_regs = 0;
            num_b64_regs = 0;
            num_pred_regs = 0;
            num_additional_b16 = 0;
            num_additional_b32 = 0;
            num_additional_b64 = 0;
            num_additional_pred_regs = 0;
            brace_counter = 0;
            brace_encountered = false;
            use_BlockIdx_x = false;
            use_BlockIdx_y = false;
            use_BlockIdx_z = false;
            kernel_lines.clear();
        }
        
        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            ptb_ptx_code += line + "\n";
        }

        int32_t numLeftBrace = countLeftBrace(line);
        int32_t numRightBrace = countRightBrace(line);

        brace_counter += numLeftBrace;
        brace_counter -= numRightBrace;
        if (!brace_encountered && numLeftBrace > 0) {
            brace_encountered = true;
        }
        
        if (boost::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
            continue;
        }

        if (boost::regex_search(line, matches, b16_reg_decl_pattern)) {
            num_b16_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, b32_reg_decl_pattern)) {
            num_b32_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, b64_reg_decl_pattern)) {
            num_b64_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, pred_reg_decl_pattern)) {
            num_pred_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, block_idx_pattern)) {
            std::string block_idx_match_dim = matches[2];
            if (block_idx_match_dim == "x") {
                use_BlockIdx_x = true;
            } else if (block_idx_match_dim == "y") {
                use_BlockIdx_y = true;
            } else if (block_idx_match_dim == "z") {
                use_BlockIdx_z = true;
            }

            continue;
        }

        if (record_kernel && brace_encountered && brace_counter == 0) {

            record_kernel = false;

            // Ignore such kernels for now!
            if (num_params == 0) {
                continue;
            }

            // Now must be at end of kernel

            // 16-bit registers
            std::string retreat_val_reg = "%rs" + std::to_string(allocate_new_b16_reg());

            // 64-bit registers
            std::string global_idx_param_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string global_idx_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string retreat_param_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string retreat_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_arr_param_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_arr_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_addr_offset_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_addr_reg = "%rd" + std::to_string(allocate_new_b64_reg());

            // 32-bit registers
            std::string origGridDim_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string origGridDim_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string origGridDim_z_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string threadIdx_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string threadIdx_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string threadIdx_z_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string threadIdx_x_or_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string leader_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string xy_tbs_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string num_thread_blocks_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string num_thread_blocks_plus_1_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string tb_idx_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_z_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_z_mul_xy_tbs_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_y_mul_origGridDim_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string blockIdx_x_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string atomic_add_res_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string curr_idx_reg = "%r" + std::to_string(allocate_new_b32_reg());

            // predicate registers
            std::string check_leader_pred_reg = "%p" + std::to_string(allocate_new_pred_reg());
            std::string check_index_pred_reg = "%p" + std::to_string(allocate_new_pred_reg());
            std::string check_retreat_pred_reg = "%p" + std::to_string(allocate_new_pred_reg());

            std::map<std::string, std::string> reg_replacement_rules;
            reg_replacement_rules["%ctaid.x"] = newBlockIdx_x_reg;
            reg_replacement_rules["%ctaid.y"] = newBlockIdx_y_reg;
            reg_replacement_rules["%ctaid.z"] = newBlockIdx_z_reg;

            brace_counter = 0;
            brace_encountered = false;
            int count_num_params = 0;
            
            for (auto &kernel_line : kernel_lines) {

                int32_t numLeftBrace = countLeftBrace(kernel_line);
                int32_t numRightBrace = countRightBrace(kernel_line);

                brace_counter += numLeftBrace;
                brace_counter -= numRightBrace;
                assert(brace_counter >= 0);

                if (boost::regex_search(kernel_line, matches, kernel_param_pattern)) {
                    count_num_params += 1;

                    if (count_num_params == num_params) {
                        ptb_ptx_code += kernel_line + ",\n";
                        ptb_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12],\n";
                        ptb_ptx_code += ".param .u64 " + kernel_name + "_param_" + std::to_string(num_params + 1) + ",\n";
                        ptb_ptx_code += ".param .u64 " + kernel_name + "_param_" + std::to_string(num_params + 2) + ",\n";
                        ptb_ptx_code += ".param .u64 " + kernel_name + "_param_" + std::to_string(num_params + 3) + "\n";
                        count_num_params = 0;
                        continue;
                    }

                    ptb_ptx_code += kernel_line + "\n";

                    continue;
                }

                if (strip(kernel_line) == "{") {

                    if (brace_encountered) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    ptb_ptx_code += kernel_line + "\n";
    
                    brace_encountered = true;

                    // Perform actions at the top
                    ptb_ptx_code += ".reg .b16 %rs<" + std::to_string(num_b16_regs + num_additional_b16) + ">;\n";
                    ptb_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";
                    ptb_ptx_code += ".reg .b64 %rd<" + std::to_string(num_b64_regs + num_additional_b64) + ">;\n";
                    ptb_ptx_code += ".reg .pred %p<" + std::to_string(num_pred_regs + num_additional_pred_regs) + ">;\n";

                    // Load origGridDim.x
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_x_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "];\n";
                    // Load origGridDim.y
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_y_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+4];\n";
                    // Load origGridDim.z
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_z_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+8];\n";
                    // Load global_idx param
                    ptb_ptx_code += "ld.param.u64 " + global_idx_param_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params + 1) + "];\n";
                    // Load retreat param
                    ptb_ptx_code += "ld.param.u64 " + retreat_param_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params + 2) + "];\n";
                    // Load curr_idx_arr param
                    ptb_ptx_code += "ld.param.u64 " + curr_idx_arr_param_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params + 3) + "];\n";

                    // Convert addr for global_idx param
                    ptb_ptx_code += "cvta.to.global.u64 " + global_idx_reg + ", " + global_idx_param_reg + ";\n";
                    // Convert addr for retreat param
                    ptb_ptx_code += "cvta.to.global.u64 " + retreat_reg + ", " + retreat_param_reg + ";\n";
                    // Convert addr for curr_idx_arr param
                    ptb_ptx_code += "cvta.to.global.u64 " + curr_idx_arr_reg + ", " + curr_idx_arr_param_reg + ";\n";

                    // threadIdx.x
                    ptb_ptx_code += "mov.u32 " + threadIdx_x_reg + ", %tid.x;\n";
                    // threadIdx.y
                    ptb_ptx_code += "mov.u32 " + threadIdx_y_reg + ", %tid.y;\n";
                    // threadIdx.z
                    ptb_ptx_code += "mov.u32 " + threadIdx_z_reg + ", %tid.z;\n";

                    // threadIdx.x || threadIdx.y
                    ptb_ptx_code += "or.b32 " + threadIdx_x_or_y_reg + ", " + threadIdx_x_reg + ", " + threadIdx_y_reg + ";\n";
                    // leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
                    ptb_ptx_code += "or.b32 " + leader_reg + ", " + threadIdx_x_or_y_reg + ", " + threadIdx_z_reg + ";\n";

                    // xy_tbs = origGridDim.x * origGridDim.y
                    ptb_ptx_code += "mul.lo.s32 " + xy_tbs_reg + ", " + origGridDim_x_reg + ", " + origGridDim_y_reg + ";\n";
                    // num_thread_blocks = origGridDim.x * origGridDim.y * origGridDim.z
                    ptb_ptx_code += "mul.lo.s32 " + num_thread_blocks_reg + ", " + origGridDim_z_reg + ", " + xy_tbs_reg + ";\n";

                    // load blockIdx.x as threads's index to curr_idx arr
                    ptb_ptx_code += "mov.u32 " + blockIdx_x_reg + ", %ctaid.x;\n";
                    // offset = blockIdx.x * 4
                    ptb_ptx_code += "mul.wide.u32 " + curr_idx_addr_offset_reg + ", " + blockIdx_x_reg + ", 4;\n";
                    // addr = curr_idx_arr + offset
                    ptb_ptx_code += "add.s64 " + curr_idx_addr_reg + ", " + curr_idx_arr_reg + ", " + curr_idx_addr_offset_reg + ";\n";

                    // Always branch to CHECK LEADER BLOCK first
                    ptb_ptx_code += "bra.uni $" + PTB_CHECK_LEADER_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                    // Wrap original code inside the MAIN block
                    ptb_ptx_code += "$" + PTB_MAIN_BLOCK_NAME + ":\n";
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b16_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b64_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, pred_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, kernel_name_pattern)) {
                    auto new_kernel_name = kernel_name + "_tally_preemptive_ptb";
                    ptb_ptx_code += replace_substring(kernel_line, kernel_name, new_kernel_name) + "\n";
                    continue;
                }
            
                // instead of return, branch to loop condition check
                if (strip(kernel_line) == "ret;") {
                    ptb_ptx_code += "bra.uni $" + PTB_CHECK_LEADER_BLOCK_NAME + ";\n";
                    continue;
                }

                // Potentially end of kernel
                if (strip(kernel_line) == "}") {
                    
                    if (brace_counter != 0) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    ptb_ptx_code += "\n";

                    // Check leader block
                    ptb_ptx_code += "$" + PTB_CHECK_LEADER_BLOCK_NAME + ":\n";
                    // if (leader)
                    ptb_ptx_code += "setp.ne.s32 " + check_leader_pred_reg + ", " + leader_reg + ", 0;\n";
                    // Branch to check index if not leader
                    ptb_ptx_code += "@" + check_leader_pred_reg + " bra $" + PTB_CHECK_INDEX_BLOCK_NAME + ";\n";
                    
                    ptb_ptx_code += "\n";

                    // Load retreat value
                    ptb_ptx_code += "ld.volatile.global.u8 " + retreat_val_reg + ", [" + retreat_reg + "];\n";
                    // Check if retreat
                    ptb_ptx_code += "setp.eq.s16 " + check_retreat_pred_reg + ", " + retreat_val_reg + ", 0;\n";
                    // Branch to atomic add if not retreat
                    ptb_ptx_code += "@" + check_retreat_pred_reg + " bra $" + PTB_ATOMIC_ADD_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                    // Compute num_thread_blocks + 1
                    ptb_ptx_code += "add.s32 " + num_thread_blocks_plus_1_reg + ", " + num_thread_blocks_reg + ", 1;\n";
                    // Store volatile_curr_idx = num_thread_blocks + 1
                    ptb_ptx_code += "st.global.u32 [" + curr_idx_addr_reg + "], " + num_thread_blocks_plus_1_reg + ";\n";
                    // Branch to check index block
                    ptb_ptx_code += "bra.uni $" + PTB_CHECK_INDEX_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                    // Start atomic add block
                    ptb_ptx_code += "$" + PTB_ATOMIC_ADD_BLOCK_NAME + ":\n";
                    // curr_idx = atomicAdd(global_idx, 1);
                    ptb_ptx_code += "atom.global.add.u32 " + atomic_add_res_reg + ", [" + global_idx_reg + "], 1;\n";
                    // store curr_idx
                    ptb_ptx_code += "st.global.u32 [" + curr_idx_addr_reg + "], " + atomic_add_res_reg + ";\n";

                    ptb_ptx_code += "\n";

                    // Check curr_idx index block
                    ptb_ptx_code += "$" + PTB_CHECK_INDEX_BLOCK_NAME + ":\n";
                    // __syncthreads();
                    ptb_ptx_code += "bar.sync 0;\n";
                    // load curr_idx from volatile mem
                    ptb_ptx_code += "ld.global.u32 " + curr_idx_reg + ", [" + curr_idx_addr_reg + "];\n";
                    // if curr_idx > num_thread_blocks
                    ptb_ptx_code += "setp.ge.u32 " + check_index_pred_reg + ", " + curr_idx_reg + ", " + num_thread_blocks_reg + ";\n";
                    // return if curr_idx > num_thread_blocks
                    ptb_ptx_code += "@" + check_index_pred_reg + " bra $" + PTB_RETURN_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                     // newBlockIdx.z
                    if (use_BlockIdx_z) {
                        ptb_ptx_code += "div.u32 " + newBlockIdx_z_reg + ", " + curr_idx_reg + ", " + xy_tbs_reg + ";\n";
                    }
                    
                    if (use_BlockIdx_y) {
                        if (use_BlockIdx_z) {
                            // newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                            // tb_idx - newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + curr_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                            // newBlockIdx.y
                            ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + origGridDim_x_reg + ";\n";
                        } else {
                            // newBlockIdx.y
                            ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + curr_idx_reg + ", " + origGridDim_x_reg + ";\n";
                        }
                    }

                    if (use_BlockIdx_x) {
                        if (use_BlockIdx_y) {
                            // newBlockIdx.y * origGridDim.x
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_y_mul_origGridDim_x_reg + ", " + newBlockIdx_y_reg + ", " + origGridDim_x_reg + ";\n";
                            
                            if (use_BlockIdx_z) {
                                // newBlockIdx.x
                                ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";
                            } else {
                                 // newBlockIdx.x
                                ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + curr_idx_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";
                            }
                        } else if (use_BlockIdx_z) {
                            // newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                            // tb_idx - newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + curr_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                            // newBlockIdx.x
                            ptb_ptx_code += "mov.u32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                        } else {
                            // newBlockIdx.x
                            ptb_ptx_code += "mov.u32 " + newBlockIdx_x_reg + ", " + curr_idx_reg + ";\n";
                        }
                    }
                    
                    // Branch to MAIN block
                    ptb_ptx_code += "bra.uni $" + PTB_MAIN_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                    // Add return block
                    ptb_ptx_code += "$" + PTB_RETURN_BLOCK_NAME + ":\n";
                    ptb_ptx_code += "ret;\n";

                    ptb_ptx_code += kernel_line + "\n";
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

// Generating dynamic PTB version of a PTX file
std::string gen_dynamic_ptb_ptx(std::string &kernel_ptx_str)
{
    std::stringstream ss(kernel_ptx_str);
    std::string line;
    boost::smatch matches;

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    boost::regex b64_reg_decl_pattern("\\.reg \\.b64 %rd<(\\d+)>;");
    boost::regex pred_reg_decl_pattern("\\.reg \\.pred %p<(\\d+)>;");
    boost::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    boost::regex kernel_param_pattern("^\\.param");

    std::string PTB_MAIN_BLOCK_NAME = "L__PTB_MAIN";
    std::string PTB_CHECK_LEADER_BLOCK_NAME = "L__PTB_CHECK_LEADER";
    std::string PTB_CHECK_INDEX_BLOCK_NAME = "L__PTB_CHECK_INDEX";
    std::string PTB_RETURN_BLOCK_NAME = "L__PTB_RETURN";

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    uint32_t num_b64_regs = 0;
    uint32_t num_pred_regs = 0;
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    int32_t brace_counter = 0;
    int32_t brace_encountered = false;
    std::string kernel_name;

    bool use_BlockIdx_x = false;
    bool use_BlockIdx_y = false;
    bool use_BlockIdx_z = false;

    uint32_t num_additional_b32 = 0;
    uint32_t num_additional_b64 = 0;
    uint32_t num_additional_pred_regs = 0;

    auto allocate_new_b32_reg = [&num_b32_regs, &num_additional_b32]() {
        uint32_t new_b32_reg = num_b32_regs + num_additional_b32;
        num_additional_b32++;
        return new_b32_reg;
    };

    auto allocate_new_b64_reg = [&num_b64_regs, &num_additional_b64]() {
        uint32_t new_b64_reg = num_b64_regs + num_additional_b64;
        num_additional_b64++;
        return new_b64_reg;
    };

    auto allocate_new_pred_reg = [&num_pred_regs, &num_additional_pred_regs]() {
        uint32_t new_pred_reg = num_pred_regs + num_additional_pred_regs;
        num_additional_pred_regs++;
        return new_pred_reg;
    };

    std::string ptb_ptx_code = "";

    // boost::timer::progress_display progress(ptx_code_str.size());
    while (std::getline(ss, line, '\n')) {
        // progress += line.size() + 1;
        if (boost::regex_search(line, matches, kernel_name_pattern)) {
            record_kernel = true;
            kernel_name = matches[2];
            num_params = 0;
            num_b32_regs = 0;
            num_b64_regs = 0;
            num_pred_regs = 0;
            num_additional_b32 = 0;
            num_additional_b64 = 0;
            num_additional_pred_regs = 0;
            brace_counter = 0;
            brace_encountered = false;
            use_BlockIdx_x = false;
            use_BlockIdx_y = false;
            use_BlockIdx_z = false;
            kernel_lines.clear();
        }
        
        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            ptb_ptx_code += line + "\n";
        }

        int32_t numLeftBrace = countLeftBrace(line);
        int32_t numRightBrace = countRightBrace(line);

        brace_counter += numLeftBrace;
        brace_counter -= numRightBrace;
        if (!brace_encountered && numLeftBrace > 0) {
            brace_encountered = true;
        }
        
        if (boost::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
            continue;
        }

        if (boost::regex_search(line, matches, b32_reg_decl_pattern)) {
            num_b32_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, b64_reg_decl_pattern)) {
            num_b64_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, pred_reg_decl_pattern)) {
            num_pred_regs = std::stoi(matches[1]);
            continue;
        }

        if (boost::regex_search(line, matches, block_idx_pattern)) {
            std::string block_idx_match_dim = matches[2];
            if (block_idx_match_dim == "x") {
                use_BlockIdx_x = true;
            } else if (block_idx_match_dim == "y") {
                use_BlockIdx_y = true;
            } else if (block_idx_match_dim == "z") {
                use_BlockIdx_z = true;
            }

            continue;
        }

        if (record_kernel && brace_encountered && brace_counter == 0) {

            record_kernel = false;

            // Ignore such kernels for now!
            if (num_params == 0) {
                continue;
            }

            // Now must be at end of kernel

            // 64-bit registers
            std::string global_idx_param_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string global_idx_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_arr_param_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_arr_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_addr_offset_reg = "%rd" + std::to_string(allocate_new_b64_reg());
            std::string curr_idx_addr_reg = "%rd" + std::to_string(allocate_new_b64_reg());

            // 32-bit registers
            std::string origGridDim_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string origGridDim_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string origGridDim_z_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string threadIdx_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string threadIdx_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string threadIdx_z_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string threadIdx_x_or_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string leader_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string xy_tbs_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string num_thread_blocks_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string tb_idx_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_z_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_z_mul_xy_tbs_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_y_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_y_mul_origGridDim_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string newBlockIdx_x_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string blockIdx_x_reg = "%r" + std::to_string(allocate_new_b32_reg());

            std::string atomic_add_res_reg = "%r" + std::to_string(allocate_new_b32_reg());
            std::string curr_idx_reg = "%r" + std::to_string(allocate_new_b32_reg());

            // predicate registers
            std::string check_leader_pred_reg = "%p" + std::to_string(allocate_new_pred_reg());
            std::string check_index_pred_reg = "%p" + std::to_string(allocate_new_pred_reg());

            std::map<std::string, std::string> reg_replacement_rules;
            reg_replacement_rules["%ctaid.x"] = newBlockIdx_x_reg;
            reg_replacement_rules["%ctaid.y"] = newBlockIdx_y_reg;
            reg_replacement_rules["%ctaid.z"] = newBlockIdx_z_reg;

            brace_counter = 0;
            brace_encountered = false;
            int count_num_params = 0;
            
            for (auto &kernel_line : kernel_lines) {

                int32_t numLeftBrace = countLeftBrace(kernel_line);
                int32_t numRightBrace = countRightBrace(kernel_line);

                brace_counter += numLeftBrace;
                brace_counter -= numRightBrace;
                assert(brace_counter >= 0);

                if (boost::regex_search(kernel_line, matches, kernel_param_pattern)) {
                    count_num_params += 1;

                    if (count_num_params == num_params) {
                        ptb_ptx_code += kernel_line + ",\n";
                        ptb_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12],\n";
                        ptb_ptx_code += ".param .u64 " + kernel_name + "_param_" + std::to_string(num_params + 1) + ",\n";
                        ptb_ptx_code += ".param .u64 " + kernel_name + "_param_" + std::to_string(num_params + 2) + "\n";
                        count_num_params = 0;
                        continue;
                    }

                    ptb_ptx_code += kernel_line + "\n";

                    continue;
                }

                if (strip(kernel_line) == "{") {

                    if (brace_encountered) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    ptb_ptx_code += kernel_line + "\n";
    
                    brace_encountered = true;

                    // Perform actions at the top
                    ptb_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";
                    ptb_ptx_code += ".reg .b64 %rd<" + std::to_string(num_b64_regs + num_additional_b64) + ">;\n";
                    ptb_ptx_code += ".reg .pred %p<" + std::to_string(num_pred_regs + num_additional_pred_regs) + ">;\n";

                    // Load origGridDim.x
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_x_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "];\n";
                    // Load origGridDim.y
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_y_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+4];\n";
                    // Load origGridDim.z
                    ptb_ptx_code += "ld.param.u32 " + origGridDim_z_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+8];\n";
                    // Load global_idx param
                    ptb_ptx_code += "ld.param.u64 " + global_idx_param_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params + 1) + "];\n";
                    // Load curr_idx_arr param
                    ptb_ptx_code += "ld.param.u64 " + curr_idx_arr_param_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params + 2) + "];\n";

                    // Convert addr for global_idx param
                    ptb_ptx_code += "cvta.to.global.u64 " + global_idx_reg + ", " + global_idx_param_reg + ";\n";
                    // Convert addr for retreat param
                    ptb_ptx_code += "cvta.to.global.u64 " + curr_idx_arr_reg + ", " + curr_idx_arr_param_reg + ";\n";

                    // threadIdx.x
                    ptb_ptx_code += "mov.u32 " + threadIdx_x_reg + ", %tid.x;\n";
                    // threadIdx.y
                    ptb_ptx_code += "mov.u32 " + threadIdx_y_reg + ", %tid.y;\n";
                    // threadIdx.z
                    ptb_ptx_code += "mov.u32 " + threadIdx_z_reg + ", %tid.z;\n";

                    // threadIdx.x || threadIdx.y
                    ptb_ptx_code += "or.b32 " + threadIdx_x_or_y_reg + ", " + threadIdx_x_reg + ", " + threadIdx_y_reg + ";\n";
                    // leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
                    ptb_ptx_code += "or.b32 " + leader_reg + ", " + threadIdx_x_or_y_reg + ", " + threadIdx_z_reg + ";\n";

                    // xy_tbs = origGridDim.x * origGridDim.y
                    ptb_ptx_code += "mul.lo.s32 " + xy_tbs_reg + ", " + origGridDim_x_reg + ", " + origGridDim_y_reg + ";\n";
                    // num_thread_blocks = origGridDim.x * origGridDim.y * origGridDim.z
                    ptb_ptx_code += "mul.lo.s32 " + num_thread_blocks_reg + ", " + origGridDim_z_reg + ", " + xy_tbs_reg + ";\n";

                    // load blockIdx.x as threads's index to curr_idx arr
                    ptb_ptx_code += "mov.u32 " + blockIdx_x_reg + ", %ctaid.x;\n";
                    // offset = blockIdx.x * 4
                    ptb_ptx_code += "mul.wide.u32 " + curr_idx_addr_offset_reg + ", " + blockIdx_x_reg + ", 4;\n";
                    // addr = curr_idx_arr + offset
                    ptb_ptx_code += "add.s64 " + curr_idx_addr_reg + ", " + curr_idx_arr_reg + ", " + curr_idx_addr_offset_reg + ";\n";

                    // Always branch to CHECK LEADER BLOCK first
                    ptb_ptx_code += "bra.uni $" + PTB_CHECK_LEADER_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                    // Wrap original code inside the MAIN block
                    ptb_ptx_code += "$" + PTB_MAIN_BLOCK_NAME + ":\n";
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b64_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, pred_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, kernel_name_pattern)) {
                    auto new_kernel_name = kernel_name + "_tally_dynamic_ptb";
                    ptb_ptx_code += replace_substring(kernel_line, kernel_name, new_kernel_name) + "\n";
                    continue;
                }
            
                // instead of return, branch to loop condition check
                if (strip(kernel_line) == "ret;") {
                    ptb_ptx_code += "bra.uni $" + PTB_CHECK_LEADER_BLOCK_NAME + ";\n";
                    continue;
                }

                // Potentially end of kernel
                if (strip(kernel_line) == "}") {
                    
                    if (brace_counter != 0) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    ptb_ptx_code += "\n";

                    // Check leader block
                    ptb_ptx_code += "$" + PTB_CHECK_LEADER_BLOCK_NAME + ":\n";
                    // if (leader)
                    ptb_ptx_code += "setp.ne.s32 " + check_leader_pred_reg + ", " + leader_reg + ", 0;\n";
                    // Branch to check index if not leader
                    ptb_ptx_code += "@" + check_leader_pred_reg + " bra $" + PTB_CHECK_INDEX_BLOCK_NAME + ";\n";
                    
                    ptb_ptx_code += "\n";

                    // curr_idx = atomicAdd(global_idx, 1);
                    ptb_ptx_code += "atom.global.add.u32 " + atomic_add_res_reg + ", [" + global_idx_reg + "], 1;\n";
                    // store curr_idx
                    ptb_ptx_code += "st.global.u32 [" + curr_idx_addr_reg + "], " + atomic_add_res_reg + ";\n";

                    ptb_ptx_code += "\n";

                    // Check curr_idx index block
                    ptb_ptx_code += "$" + PTB_CHECK_INDEX_BLOCK_NAME + ":\n";
                    // __syncthreads();
                    ptb_ptx_code += "bar.sync 0;\n";
                    // load curr_idx from volatile mem
                    ptb_ptx_code += "ld.global.u32 " + curr_idx_reg + ", [" + curr_idx_addr_reg + "];\n";
                    // if curr_idx > num_thread_blocks
                    ptb_ptx_code += "setp.ge.u32 " + check_index_pred_reg + ", " + curr_idx_reg + ", " + num_thread_blocks_reg + ";\n";
                    // return if curr_idx > num_thread_blocks
                    ptb_ptx_code += "@" + check_index_pred_reg + " bra $" + PTB_RETURN_BLOCK_NAME + ";";

                    ptb_ptx_code += "\n";

                    // newBlockIdx.z
                    if (use_BlockIdx_z) {
                        ptb_ptx_code += "div.u32 " + newBlockIdx_z_reg + ", " + curr_idx_reg + ", " + xy_tbs_reg + ";\n";
                    }
                    
                    if (use_BlockIdx_y) {
                        if (use_BlockIdx_z) {
                            // newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                            // tb_idx - newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + curr_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                            // newBlockIdx.y
                            ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + origGridDim_x_reg + ";\n";
                        } else {
                            // newBlockIdx.y
                            ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + curr_idx_reg + ", " + origGridDim_x_reg + ";\n";
                        }
                    }

                    if (use_BlockIdx_x) {
                        if (use_BlockIdx_y) {
                            // newBlockIdx.y * origGridDim.x
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_y_mul_origGridDim_x_reg + ", " + newBlockIdx_y_reg + ", " + origGridDim_x_reg + ";\n";
                            
                            if (use_BlockIdx_z) {
                                // newBlockIdx.x
                                ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";
                            } else {
                                 // newBlockIdx.x
                                ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + curr_idx_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";
                            }
                        } else if (use_BlockIdx_z) {
                            // newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                            // tb_idx - newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + curr_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                            // newBlockIdx.x
                            ptb_ptx_code += "mov.u32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                        } else {
                            // newBlockIdx.x
                            ptb_ptx_code += "mov.u32 " + newBlockIdx_x_reg + ", " + curr_idx_reg + ";\n";
                        }
                    }
                    
                    // Branch to MAIN block
                    ptb_ptx_code += "bra.uni $" + PTB_MAIN_BLOCK_NAME + ";\n";

                    ptb_ptx_code += "\n";

                    // Add return block
                    ptb_ptx_code += "$" + PTB_RETURN_BLOCK_NAME + ":\n";
                    ptb_ptx_code += "ret;\n";

                    ptb_ptx_code += kernel_line + "\n";
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


std::string gen_ptb_ptx(std::string &kernel_ptx_str)
{
    std::stringstream ss(kernel_ptx_str);
    std::string line;
    boost::smatch matches;

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    boost::regex pred_reg_decl_pattern("\\.reg \\.pred %p<(\\d+)>;");
    boost::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    boost::regex kernel_param_pattern("^\\.param");

    std::string PTB_RETURN_BLOCK_NAME = "L__PTB_RETURN";
    std::string PTB_LOOP_BLOCK_NAME = "L__PTB_LOOP";
    std::string PTB_LOOP_CONDITION_BLOCK_NAME = "L__PTB_LOOP_CONDITION";

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    uint32_t num_pred_regs = 0;
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    int32_t brace_counter = 0;
    int32_t brace_encountered = false;
    std::string kernel_name;

    bool use_BlockIdx_x = false;
    bool use_BlockIdx_y = false;
    bool use_BlockIdx_z = false;

    uint32_t num_additional_b32 = 15;
    uint32_t num_additional_pred_regs = 2;

    std::string ptb_ptx_code = "";

    // boost::timer::progress_display progress(ptx_code_str.size());
    while (std::getline(ss, line, '\n')) {

        // progress += line.size() + 1;
        if (boost::regex_search(line, matches, kernel_name_pattern)) {
            record_kernel = true;
            kernel_name = matches[2];
            num_params = 0;
            num_b32_regs = 0;
            num_pred_regs = 0;
            brace_counter = 0;
            brace_encountered = false;
            use_BlockIdx_x = false;
            use_BlockIdx_y = false;
            use_BlockIdx_z = false;
            kernel_lines.clear();
        }
        
        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            ptb_ptx_code += line + "\n";
        }

        int32_t numLeftBrace = countLeftBrace(line);
        int32_t numRightBrace = countRightBrace(line);

        brace_counter += numLeftBrace;
        brace_counter -= numRightBrace;
        if (!brace_encountered && numLeftBrace > 0) {
            brace_encountered = true;
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

        if (boost::regex_search(line, matches, block_idx_pattern)) {
            std::string block_idx_match_dim = matches[2];
            if (block_idx_match_dim == "x") {
                use_BlockIdx_x = true;
            } else if (block_idx_match_dim == "y") {
                use_BlockIdx_y = true;
            } else if (block_idx_match_dim == "z") {
                use_BlockIdx_z = true;
            }

            continue;
        }

        if (record_kernel && brace_encountered && brace_counter == 0) {

            record_kernel = false;

            // Ignore such kernels for now!
            if (num_params == 0) {
                continue;
            }

            // Now must be at end of kernel
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

            std::map<std::string, std::string> reg_replacement_rules;
            reg_replacement_rules["%ctaid.x"] = newBlockIdx_x_reg;
            reg_replacement_rules["%ctaid.y"] = newBlockIdx_y_reg;
            reg_replacement_rules["%ctaid.z"] = newBlockIdx_z_reg;

            brace_counter = 0;
            brace_encountered = false;
            int count_num_params = 0;

            for (auto &kernel_line : kernel_lines) {

                int32_t numLeftBrace = countLeftBrace(kernel_line);
                int32_t numRightBrace = countRightBrace(kernel_line);

                brace_counter += numLeftBrace;
                brace_counter -= numRightBrace;
                assert(brace_counter >= 0);

                if (boost::regex_search(kernel_line, matches, kernel_param_pattern)) {
                    count_num_params += 1;

                    if (count_num_params == num_params) {
                        ptb_ptx_code += kernel_line + ",\n";
                        ptb_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                        count_num_params = 0;
                        continue;
                    }

                    ptb_ptx_code += kernel_line + "\n";

                    continue;
                }

                if (strip(kernel_line) == "{") {

                    if (brace_encountered) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    ptb_ptx_code += kernel_line + "\n";
    
                    brace_encountered = true;

                    // Perform actions at the top
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
                    if (use_BlockIdx_z) {
                        ptb_ptx_code += "div.u32 " + newBlockIdx_z_reg + ", " + tb_idx_reg + ", " + xy_tbs_reg + ";\n";
                    }
                    
                    if (use_BlockIdx_y) {
                        if (use_BlockIdx_z) {
                            // newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                            // tb_idx - newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + tb_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                            // newBlockIdx.y
                            ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + origGridDim_x_reg + ";\n";
                        } else {
                            // newBlockIdx.y
                            ptb_ptx_code += "div.u32 " + newBlockIdx_y_reg + ", " + tb_idx_reg + ", " + origGridDim_x_reg + ";\n";
                        }
                    }

                    if (use_BlockIdx_x) {
                        if (use_BlockIdx_y) {
                            // newBlockIdx.y * origGridDim.x
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_y_mul_origGridDim_x_reg + ", " + newBlockIdx_y_reg + ", " + origGridDim_x_reg + ";\n";
                            
                            if (use_BlockIdx_z) {
                                // newBlockIdx.x
                                ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";
                            } else {
                                 // newBlockIdx.x
                                ptb_ptx_code += "sub.s32 " + newBlockIdx_x_reg + ", " + tb_idx_reg + ", " + newBlockIdx_y_mul_origGridDim_x_reg + ";\n";
                            }
                        } else if (use_BlockIdx_z) {
                            // newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "mul.lo.s32 " + newBlockIdx_z_mul_xy_tbs_reg + ", " + newBlockIdx_z_reg + ", " + xy_tbs_reg + ";\n";
                            // tb_idx - newBlockIdx.z * xy_tbs
                            ptb_ptx_code += "sub.s32 " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ", " + tb_idx_reg + ", " + newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                            // newBlockIdx.x
                            ptb_ptx_code += "mov.u32 " + newBlockIdx_x_reg + ", " + tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg + ";\n";
                        } else {
                            // newBlockIdx.x
                            ptb_ptx_code += "mov.u32 " + newBlockIdx_x_reg + ", " + tb_idx_reg + ";\n";
                        }
                    }

                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, pred_reg_decl_pattern)) {
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, kernel_name_pattern)) {
                    auto new_kernel_name = kernel_name + "_tally_ptb";
                    ptb_ptx_code += replace_substring(kernel_line, kernel_name, new_kernel_name) + "\n";
                    continue;
                }
            
                // instead of return, branch to loop condition check
                if (strip(kernel_line) == "ret;") {
                    ptb_ptx_code += "bra.uni $" + PTB_LOOP_CONDITION_BLOCK_NAME + ";\n";
                    continue;
                }

                // Potentially end of kernel
                if (strip(kernel_line) == "}") {
                    
                    if (brace_counter != 0) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    // End of kernel
                    ptb_ptx_code += "$" + PTB_LOOP_CONDITION_BLOCK_NAME + ":\n";
                    // tb_idx += gridDim.x
                    ptb_ptx_code += "add.s32 " + tb_idx_reg + ", " + tb_idx_reg + ", " + gridDim_x_reg + ";\n";
                    // tb_idx < num_thread_blocks
                    ptb_ptx_code += "setp.lt.u32 " + tb_idx_lt_num_thread_blocks_reg + ", " + tb_idx_reg + ", " + num_thread_blocks_reg + ";\n";
                    // branch to L__PTB_LOOP if tb_idx < num_thread_blocks
                    ptb_ptx_code += "@" + tb_idx_lt_num_thread_blocks_reg + " bra $" + PTB_LOOP_BLOCK_NAME + ";\n\n";

                    // Add return block
                    ptb_ptx_code += "$" + PTB_RETURN_BLOCK_NAME + ":\n";
                    ptb_ptx_code += "ret;\n";

                    ptb_ptx_code += kernel_line + "\n";
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

// Generating combined version of a PTX file
std::string gen_transform_ptx(std::string &ptx_path)
{
    std::ifstream t(ptx_path);
    if (!t.is_open()) {
        std::cerr << ptx_path << " not found." << std::endl;
        return "";
    }
    std::string ptx_code_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::stringstream ss(ptx_code_str);
    std::string line;

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");

    bool record_kernel = false;
    int32_t brace_counter = 0;
    bool brace_encountered = false;

    std::string final_ptx_str = "";
    std::string kernel_func_str = "";

    final_ptx_str = ptx_code_str;

    // // Iterate ptx file, and extract each kernel function
    // while (std::getline(ss, line, '\n')) {

    //     boost::smatch matches;
    //     if (boost::regex_search(line, matches, kernel_name_pattern)) {
    //         record_kernel = true;
    //     }

    //     if (record_kernel) {
    //         kernel_func_str += line + "\n";
    //     } else {
    //         final_ptx_str += line + '\n';
    //     }

    //     int32_t numLeftBrace = countLeftBrace(line);
    //     int32_t numRightBrace = countRightBrace(line);

    //     brace_counter += numLeftBrace;
    //     brace_counter -= numRightBrace;
    //     if (!brace_encountered && numLeftBrace > 0) {
    //         brace_encountered = true;
    //     }

    //     if (record_kernel && brace_encountered && brace_counter == 0) {
    //         record_kernel = false;
    //         brace_encountered = false;

    //         final_ptx_str += kernel_func_str + "\n";
    //         final_ptx_str += gen_ptb_ptx(kernel_func_str) + "\n";
    //         final_ptx_str += gen_dynamic_ptb_ptx(kernel_func_str) + "\n";
    //         final_ptx_str += gen_preemptive_ptb_ptx(kernel_func_str) + "\n";

    //         kernel_func_str = "";
    //     }
    // }

    return final_ptx_str;
}