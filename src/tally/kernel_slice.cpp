#include <string>
#include <map>
#include <vector>
#include <regex>

#include <cuda.h>

#include <tally/util.h>
#include <tally/kernel_slice.h>
#include <tally/cuda_api.h>

std::string write_cubin_to_file(const char* data, uint32_t size)
{
    std::string file_name;
    int i = 0;
    while (true) {
        file_name = "output" + std::to_string(i) + ".cubin";
        std::ifstream infile(file_name);
        if (infile.good()) {
            i++;
            continue;
        }
        infile.close();

        std::ofstream file(file_name, std::ios::binary); // Open the file in binary mode
        file.write(reinterpret_cast<const char*>(data), size);
        file.close();

        break;
    }

    return file_name;
}

/* Extract all ptx code from cubin file, 
   return a vector of the generated file names */
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path)
{
    exec("cuobjdump -xptx all " + cubin_path);

    auto output = exec("cuobjdump " + cubin_path + " -lptx");
    std::stringstream ss(output);
    std::vector<std::string> names;
    std::string line;

    while (std::getline(ss, line, '\n')) {
        auto split_str = splitOnce(line, ":");
        auto ptx_file_name = strip(split_str.second);
        names.push_back(ptx_file_name);
    }

    return names;
}

void gen_sliced_ptx(std::string ptx_path, std::string output_path)
{
    std::ifstream t(ptx_path);
    if (!t.is_open()) {
        std::cerr << ptx_path << " not found." << std::endl;
        return;
    }
    std::string ptx_code_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::stringstream ss(ptx_code_str);
    std::string line;
    
    std::string sliced_ptx_code = "";

    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    std::string kernel_name;

    std::regex kernel_name_pattern("\\.visible \\.entry (\\w+)");
    std::regex b32_reg_decl_pattern("\\.reg \\.b32 %r<(\\d+)>;");
    std::regex block_idx_pattern("mov\\.u32 %r(\\d+), %ctaid\\.([xyz])");
    std::regex kernel_param_pattern("^placeholder$");

    uint32_t num_params = 0;
    uint32_t num_b32_regs = 0;
    bool use_block_idx_xyz[3] = { false, false, false };

    while (std::getline(ss, line, '\n')) {
        std::smatch matches;
        if (std::regex_search(line, matches, kernel_name_pattern)) {
            record_kernel = true;
            kernel_name = matches[1];
            kernel_param_pattern = std::regex("\\.param (.+) " + kernel_name + "_param_(\\d+)");
            num_params = 0;
            num_b32_regs = 0;
            
            // reset to false 
            memset(use_block_idx_xyz, 0, 3);
        }

        if (record_kernel) {
            kernel_lines.push_back(line);
        } else {
            sliced_ptx_code += line + "\n";
        }

        if (std::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
        }

        if (std::regex_search(line, matches, b32_reg_decl_pattern)) {
            num_b32_regs = std::stoi(matches[1]);
        }

        if (std::regex_search(line, matches, block_idx_pattern)) {
            std::string block_idx_match_dim = matches[2];
            if (block_idx_match_dim == "x") {
                use_block_idx_xyz[0] = true;
            } else if (block_idx_match_dim == "y") {
                use_block_idx_xyz[1] = true;
            } else if (block_idx_match_dim == "z") {
                use_block_idx_xyz[2] = true;
            }
        }

        if (line == "}") {
            record_kernel = false;

            std::regex last_param_pattern("\\.param \\.(\\w+) " + kernel_name + "_param_" + std::to_string(num_params - 1));
            std::regex last_ld_param_pattern("ld\\.param(.+)\\[" + kernel_name + "_param_" + std::to_string(num_params - 1) + "\\];");
            uint32_t num_additional_b32 = 0;
            for (size_t i = 0; i < 3; i++) {
                if (use_block_idx_xyz[i]) {
                    num_additional_b32 += 2;
                }
            }

            uint32_t block_offset_xyz_reg[3] { num_b32_regs, num_b32_regs + 2, num_b32_regs + 4 };
            uint32_t new_block_idx_xyz_reg[3] { num_b32_regs + 1,  num_b32_regs + 3, num_b32_regs + 5 };

            std::map<std::string, std::string> reg_replacement_rules;
    
            for (auto &kernel_line : kernel_lines) {
                
                if (std::regex_search(kernel_line, matches, last_param_pattern)) {
                    sliced_ptx_code += kernel_line + ",\n";
                    sliced_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                    continue;
                }

                if (std::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    sliced_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";
                    continue;
                }

                if (std::regex_search(kernel_line, matches, last_ld_param_pattern)) {
                    sliced_ptx_code += kernel_line + "\n";
                    
                    for (size_t i = 0; i < 3; i++) {
                        if (use_block_idx_xyz[i]) {
                            sliced_ptx_code += "ld.param.u32 %r" + std::to_string(block_offset_xyz_reg[i]) + ", [" + kernel_name + "_param_4+" + std::to_string(i * 4) + "];\n";
                        }
                    }

                    continue;
                }

                if (std::regex_search(kernel_line, matches, block_idx_pattern)) {
                    std::string block_idx_match_dim = matches[2];
                    uint32_t block_idx_match_reg = std::stoi(matches[1]);

                    sliced_ptx_code += kernel_line + "\n";

                    int32_t idx = -1;
                    if (block_idx_match_dim == "x") {
                        idx = 0;
                    } else if (block_idx_match_dim == "y") {
                        idx = 1;
                    } else if (block_idx_match_dim == "z") {
                        idx = 2;
                    }

                    sliced_ptx_code += "add.u32 %r" + std::to_string(new_block_idx_xyz_reg[idx]) + ", %r" + std::to_string(block_idx_match_reg) + ", %r" + std::to_string(block_offset_xyz_reg[idx]) + ";\n";
                    reg_replacement_rules["%r" + std::to_string(block_idx_match_reg)] = "%r" + std::to_string(new_block_idx_xyz_reg[idx]);

                    continue;
                }

                for (const auto& pair : reg_replacement_rules) {
                    std::regex pattern(pair.first);
                    kernel_line = std::regex_replace(kernel_line, pattern, pair.second);
                }

                sliced_ptx_code += kernel_line + "\n";
            }
        }
    }

    std::ofstream outputFile(output_path);  // Open the file for writing

    if (outputFile.is_open()) {
        outputFile << sliced_ptx_code;  // Write the content to the file
        outputFile.close();  // Close the file
    } else {
        std::cerr << "Unable to open " << output_path << std::endl;
    }
}


std::map<std::string, std::pair<CUfunction, uint32_t>> register_ptx(std::string ptx_path)
{
    std::map<std::string, std::pair<CUfunction, uint32_t>> name_to_func_map;

    std::ifstream t(ptx_path);
    if (!t.is_open()) {
        std::cerr << ptx_path << " not found." << std::endl;
        return name_to_func_map;
    }
    std::string ptx_code_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::stringstream ss(ptx_code_str);
    std::string line;

    std::regex kernel_name_pattern("\\.visible \\.entry (\\w+)");
    std::regex kernel_param_pattern("^placeholder$");
    uint32_t num_params = 0;
    std::string kernel_name;

    while (std::getline(ss, line, '\n')) {

        std::smatch matches;
        if (std::regex_search(line, matches, kernel_name_pattern)) {
            kernel_name = matches[1];
            kernel_param_pattern = std::regex("\\.param (.+) " + kernel_name + "_param_(\\d+)");
            num_params = 0;
        }

        if (std::regex_search(line, matches, kernel_param_pattern)) {
            num_params += 1;
        }

        if (line == "}") {
            name_to_func_map[kernel_name] = std::make_pair(nullptr, num_params);
        }
    }

    CUmodule cudaModule;
    lcuModuleLoadDataEx(&cudaModule, ptx_code_str.c_str(), 0, 0, 0);

    for (auto &pair : name_to_func_map) {
        auto kernel_name = pair.first;

        CUfunction function;
        lcuModuleGetFunction(&function, cudaModule, kernel_name.c_str());
        name_to_func_map[kernel_name].first = function;
    }
 
    return name_to_func_map;
}