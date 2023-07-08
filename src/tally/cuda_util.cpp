#include <fstream>
#include <sstream>
#include <unistd.h>

#include <boost/regex.hpp>

#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/generated/cuda_api.h>

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

        if (strip(line) == "ret;") {
            kernel_names_and_nparams.push_back(std::make_pair(kernel_name, num_params));
        }
    }

    return kernel_names_and_nparams;
}

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf(std::string elf_path)
{
    std::ifstream elf_file(elf_path);
    if (!elf_file.is_open()) {
        std::cerr << elf_path << " not found." << std::endl;
        throw std::runtime_error("file not found");
    }
    
    std::string elf_code_str((std::istreambuf_iterator<char>(elf_file)), std::istreambuf_iterator<char>());
    std::stringstream ss(elf_code_str);

    // key: func_name, val: [ <ordinal, size> ]
    using ordinal_size_pair = std::pair<uint32_t, uint32_t>;
    std::map<std::string, std::vector<uint32_t>> kernel_names_and_param_sizes;
    std::string line;

    while (std::getline(ss, line, '\n')) {
        if (startsWith(line, ".nv.info.")) {
            std::string kernel_name = line.substr(9);
            std::vector<ordinal_size_pair> params_info;

            while (std::getline(ss, line, '\n')) {
                if (containsSubstring(line, "EIATTR_KPARAM_INFO")) {
                    
                } else if (containsSubstring(line, "Ordinal :")) {
                    auto split_by_ordinal = splitOnce(line, "Ordinal");
                    auto split_by_offset = splitOnce(split_by_ordinal.second, "Offset");
                    auto split_by_size = splitOnce(split_by_offset.second, "Size");

                    auto ordinal_str = strip_space_and_colon(split_by_offset.first);
                    auto size_str = strip_space_and_colon(split_by_size.second);

                    uint32_t arg_ordinal;
                    uint32_t arg_size;

                    try {
                        arg_ordinal = std::stoi(ordinal_str, nullptr, 16);
                        arg_size = std::stoi(size_str, nullptr, 16);
                    } catch (const std::exception& e) {
                        std::cerr << "Fail to run stoi" << std::endl;
                        std::cerr << "line: " << line << std::endl;
                        std::cerr << "ordinal_str: " << ordinal_str << std::endl;
                        std::cerr << "size_str: " << size_str << std::endl;
                        std::cerr << "elf_code_str: " << std::endl;
                        std::cerr << elf_code_str << std::endl;

                        throw e;
                    }

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

            kernel_names_and_param_sizes[kernel_name] = param_sizes;
        }
    }    

    elf_file.close();

    return kernel_names_and_param_sizes;
}