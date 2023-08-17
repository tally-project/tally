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
    std::string PTB_LOOP_CONDITION_BLOCK_NAME = "L__PTB_LOOP_CONDITION";

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    uint32_t num_pred_regs = 0;
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    int32_t brace_counter = 0;
    int32_t brace_encountered = false;
    std::string kernel_name;

    uint32_t num_additional_b32 = 15;
    uint32_t num_additional_pred_regs = 2;

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
            brace_counter = 0;
            brace_encountered = false;
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
            boost::regex last_param_pattern("\\.param (.+) " + kernel_name + "_param_" + std::to_string(num_params - 1));

            for (auto &kernel_line : kernel_lines) {

                int32_t numLeftBrace = countLeftBrace(kernel_line);
                int32_t numRightBrace = countRightBrace(kernel_line);

                brace_counter += numLeftBrace;
                brace_counter -= numRightBrace;
                assert(brace_counter >= 0);

                if (boost::regex_search(kernel_line, matches, last_param_pattern)) {
                    ptb_ptx_code += kernel_line + ",\n";
                    ptb_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                    continue;
                }

                if (strip(kernel_line) == "{") {

                    if (brace_encountered) {
                        ptb_ptx_code += kernel_line + "\n";
                        continue;
                    }

                    // if (kernel_name == "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1ELb0ENSD_9NoPermuteEEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1ELb0ESJ_EENSL_ISQ_fSE_Li0ESS_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSM_fSE_fSE_NSW_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS15_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2ENS8_6thread14UnaryTransform8IdentityEEES1F_bEENS_8epilogue11threadblock8EpilogueIS7_S16_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1M_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESJ_Lb0EEENS1H_4warp20FragmentIteratorSimtISY_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSM_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S14_EENS1R_16TileIteratorSimtISY_S1Y_fSE_S14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi4EEENS1H_6thread17LinearCombinationIfLi1EffLNS25_9ScaleType4KindE0ELS1B_2EfEENSB_ILi0ELi17EEELi1ELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE") {
                    //     ptb_ptx_code += ".maxnreg 256\n";
                    // }

                    // if (kernel_name == "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi16EEENS_9transform11threadblock28PredicatedTileAccessIteratorINS_11MatrixShapeILi128ELi16EEEfNS_6layout8RowMajorELi1ENS8_29PitchLinearWarpRakedThreadMapINS_16PitchLinearShapeILi16ELi128EEELi128ENSG_ILi4ELi8EEELi4EEENS_5ArrayIfLi4ELb1EEELb0ENSD_9NoPermuteEEENS9_25RegularTileAccessIteratorISC_fNSD_37RowMajorTensorOpMultiplicandCrosswiseILi32ELi16EEELi0ESJ_Li16EEELNS_4arch14CacheOperation4KindE1ENSA_INSB_ILi16ELi128EEEfNSD_11ColumnMajorELi0ESJ_SL_Lb0ESM_EENSO_ISV_fNSD_40ColumnMajorTensorOpMultiplicandCrosswiseILi32ELi16EEELi1ESJ_Li16EEELSU_1EfSE_NS4_9MmaPolicyINS1_4warp11MmaTensorOpINS6_ILi64ELi64ELi16EEEfSQ_fSZ_fSE_NS12_17MmaTensorOpPolicyINSS_3MmaINS6_ILi16ELi8ELi8EEELi32ENS_10tfloat32_tESE_S18_SW_fSE_NSS_13OpMultiplyAddEEENSB_ILi1ELi1EEEEELi1ELb0EbEENSB_ILi0ELi0EEES1E_Li1EEELi4ELNS1_23SharedMemoryClearOptionE0EbEENS_8epilogue11threadblock8EpilogueIS7_S1D_Li1ENS1J_22PredicatedTileIteratorINS1J_26OutputTileOptimalThreadMapINS1J_15OutputTileShapeILi128ELi8ELi2ELi1ELi1EEENS1N_ILi1ELi8ELi1ELi1ELi8EEELi128ELi4ELi32EEEfLb0ESM_Lb0EEENS1I_4warp24FragmentIteratorTensorOpIS14_S17_fSL_SE_EENS1S_20TileIteratorTensorOpIS14_S17_fSE_EENS1J_18SharedLoadIteratorINS1Q_18CompactedThreadMapEfLi16EEENS1I_6thread17LinearCombinationIfLi4EffLNS20_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EfEENSB_ILi0ELi8EEELi2ELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE") {
                    //     ptb_ptx_code += ".maxnreg 256\n";
                    // }

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

std::string gen_original_ptx(std::string ptx_path)
{
    std::cout << "Generating original version of " << ptx_path << std::endl;
    std::ifstream t(ptx_path);
    if (!t.is_open()) {
        std::cerr << ptx_path << " not found." << std::endl;
        return "";
    }
    std::string ptx_code_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    return ptx_code_str;
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
    
    std::string sliced_ptx_code = "";

    boost::regex kernel_name_pattern("(\\.visible\\s+)?\\.entry (\\w+)");
    boost::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    boost::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    boost::regex kernel_param_pattern("^placeholder$");

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    uint32_t num_additional_b32 = 9;
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    int32_t brace_counter = 0;
    int32_t brace_encountered = false;
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
            brace_counter = 0;
            brace_encountered = false;
            kernel_lines.clear();

            kernel_lines.push_back(line);
            continue;
        }

        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            sliced_ptx_code += line + "\n";
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

        if (boost::regex_search(line, matches, block_idx_pattern)) {
            std::string block_idx_match_dim = matches[2];
            continue;
        }

        if (record_kernel && brace_encountered && brace_counter == 0) {

            record_kernel = false;

            // Ignore such kernels for now!
            if (num_params == 0) {
                continue;
            }

            brace_encountered = false;

            boost::regex last_param_pattern("\\.param (.+) " + kernel_name + "_param_" + std::to_string(num_params - 1));

            std::string blockOffset_x_reg = "%r" + std::to_string(num_b32_regs);
            std::string blockOffset_y_reg = "%r" + std::to_string(num_b32_regs + 1);
            std::string blockOffset_z_reg = "%r" + std::to_string(num_b32_regs + 2);

            std::string blockIdx_x_reg = "%r" + std::to_string(num_b32_regs + 3);
            std::string blockIdx_y_reg = "%r" + std::to_string(num_b32_regs + 4);
            std::string blockIdx_z_reg = "%r" + std::to_string(num_b32_regs + 5);

            std::string new_blockIdx_x_reg = "%r" + std::to_string(num_b32_regs + 6);
            std::string new_blockIdx_y_reg = "%r" + std::to_string(num_b32_regs + 7);
            std::string new_blockIdx_z_reg = "%r" + std::to_string(num_b32_regs + 8);

            std::map<std::string, std::string> reg_replacement_rules;
            reg_replacement_rules["%ctaid.x"] = new_blockIdx_x_reg;
            reg_replacement_rules["%ctaid.y"] = new_blockIdx_y_reg;
            reg_replacement_rules["%ctaid.z"] = new_blockIdx_z_reg;
    
            for (auto &kernel_line : kernel_lines) {

                if (strip(kernel_line) == "{") {

                    sliced_ptx_code += kernel_line + "\n";

                    if (brace_encountered) {
                        continue;
                    }
    
                    brace_encountered = true;

                    // Perform actions at the top
                    sliced_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";

                    // Load blockOffset.x
                    sliced_ptx_code += "ld.param.u32 " + blockOffset_x_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "];\n";
                    // Load blockOffset.y
                    sliced_ptx_code += "ld.param.u32 " + blockOffset_y_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+4];\n";
                    // Load blockOffset.z
                    sliced_ptx_code += "ld.param.u32 " + blockOffset_z_reg + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+8];\n";

                    // Load blockIdx.x
                    sliced_ptx_code += "mov.u32 " + blockIdx_x_reg + ", %ctaid.x;\n";
                    // Load blockIdx.y
                    sliced_ptx_code += "mov.u32 " + blockIdx_y_reg + ", %ctaid.y;\n";
                    // Load blockIdx.z
                    sliced_ptx_code += "mov.u32 " + blockIdx_z_reg + ", %ctaid.z;\n";

                    // new_blockIdx.x = blockIdx.x + blockOffset.x
                    sliced_ptx_code += "add.u32 " + new_blockIdx_x_reg + ", " + blockIdx_x_reg + ", " + blockOffset_x_reg + ";\n";
                    // new_blockIdx.y = blockIdx.y + blockOffset.y
                    sliced_ptx_code += "add.u32 " + new_blockIdx_y_reg + ", " + blockIdx_y_reg + ", " + blockOffset_y_reg + ";\n";
                    // new_blockIdx.x = blockIdx.x + blockOffset.x
                    sliced_ptx_code += "add.u32 " + new_blockIdx_z_reg + ", " + blockIdx_z_reg + ", " + blockOffset_z_reg + ";\n";

                    continue;
                }

                if (boost::regex_search(kernel_line, matches, last_param_pattern)) {
                    sliced_ptx_code += kernel_line + ",\n";
                    sliced_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    continue;
                }

                for (const auto& pair : reg_replacement_rules) {
                    boost::regex pattern(pair.first);
                    kernel_line = boost::regex_replace(kernel_line, pattern, pair.second);
                }

                sliced_ptx_code += kernel_line + "\n";
            }
            continue;
        }
    }

    return sliced_ptx_code;
}