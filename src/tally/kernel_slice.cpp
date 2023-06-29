#include <string>
#include <map>
#include <vector>
#include <regex>
#include <unordered_map>

#include <cuda.h>

#include <tally/util.h>
#include <tally/kernel_slice.h>
#include <tally/cuda_api.h>

#include <boost/timer/progress_display.hpp>
#include <boost/regex.hpp>

CubinCache cubin_cache = CubinCache();

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

std::string gen_sliced_ptx(std::string ptx_path)
{
    // std::cout << "Processing " << ptx_path << std::endl;
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
    bool use_block_idx_xyz[3] = { false, false, false };
    bool record_kernel = false;
    std::vector<std::string> kernel_lines;
    std::string kernel_name;

    // boost::timer::progress_display progress(ptx_code_str.size());

    while (std::getline(ss, line, '\n')) {
        // progress += line.size() + 1;

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
            sliced_ptx_code += line + "\n";
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
                    sliced_ptx_code += kernel_line + ",\n";
                    sliced_ptx_code += ".param .align 4 .b8 " + kernel_name + "_param_" + std::to_string(num_params) + "[12]\n";
                    continue;
                }

                if (boost::regex_search(kernel_line, matches, b32_reg_decl_pattern)) {
                    sliced_ptx_code += ".reg .b32 %r<" + std::to_string(num_b32_regs + num_additional_b32) + ">;\n";
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

                    sliced_ptx_code += kernel_line + "\n";
                    sliced_ptx_code += "ld.param.u32 %r" + std::to_string(block_offset_xyz_reg[idx]) + ", [" + kernel_name + "_param_" + std::to_string(num_params) + "+" + std::to_string(idx * 4) + "];\n";
                    sliced_ptx_code += "add.u32 %r" + std::to_string(new_block_idx_xyz_reg[idx]) + ", %r" + std::to_string(block_idx_match_reg) + ", %r" + std::to_string(block_offset_xyz_reg[idx]) + ";\n";
                    reg_replacement_rules["%r" + std::to_string(block_idx_match_reg) + "(?!\\d)"] = "%r" + std::to_string(new_block_idx_xyz_reg[idx]);

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

    // std::cout << "Processing done" << std::endl;

    return sliced_ptx_code;
}

std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> register_kernels_from_ptx_fatbin(
    std::vector<std::pair<std::string, std::string>> &sliced_ptx_fatbin_strs,
    std::map<std::string, const void *> &kernel_name_map
)
{
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> kernel_map;

    for (auto &ptx_fatbin_pair : sliced_ptx_fatbin_strs) {
        
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