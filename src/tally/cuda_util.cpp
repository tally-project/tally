#include <fstream>
#include <sstream>
#include <unistd.h>

#include <boost/regex.hpp>

#include <tally/log.h>
#include <tally/env.h>
#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cuda_launch.h>
#include <tally/generated/cuda_api.h>

bool CUDA_SPECS_INITIALIZED = false;

std::vector<std::string> CUDA_COMPUTE_CAPABILITIES = {"86", "80"};
std::string CUDA_COMPUTE_CAPABILITY;
int CUDA_NUM_SM;
int CUDA_MAX_NUM_THREADS_PER_SM;
int CUDA_MAX_NUM_REGISTERS_PER_SM;
int CUDA_MAX_SHM_BYTES_PER_SM;

uint32_t FATBIN_MAGIC_NUMBER = 3126193488;

void implicit_init_cuda_ctx()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
}

void register_cuda_specs()
{
    if (!CUDA_SPECS_INITIALIZED) {
        
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        // a list of candidate cuda compute capabilities 
        CUDA_COMPUTE_CAPABILITY = std::to_string(deviceProp.major) + std::to_string(deviceProp.minor);
        CUDA_NUM_SM = deviceProp.multiProcessorCount;
        CUDA_MAX_NUM_THREADS_PER_SM = deviceProp.maxThreadsPerMultiProcessor;
        CUDA_MAX_NUM_REGISTERS_PER_SM = deviceProp.regsPerMultiprocessor;
        CUDA_MAX_SHM_BYTES_PER_SM = deviceProp.sharedMemPerMultiprocessor;

        if (!std::getenv("PRIORITY_PTB_MAX_NUM_THREADS_PER_SM")) {

            switch(CUDA_MAX_NUM_THREADS_PER_SM) {
                case 2048:
                    PRIORITY_PTB_MAX_NUM_THREADS_PER_SM = 1536;
                    break;
                case 1536:
                    PRIORITY_PTB_MAX_NUM_THREADS_PER_SM = 1024;
                    break;
            }
        }

        CUDA_SPECS_INITIALIZED = true;
    }
}

std::vector<std::string> get_candidate_cuda_compute_capabilities()
{
    auto it = std::find(
        CUDA_COMPUTE_CAPABILITIES.begin(),
        CUDA_COMPUTE_CAPABILITIES.end(),
        CUDA_COMPUTE_CAPABILITY
    );

    // return the capabilities <= target
    return std::vector<std::string>(it, CUDA_COMPUTE_CAPABILITIES.end());
}

CUDA_MODULE_TYPE get_cuda_module_type(const void * image)
{
    // Test if it is fatbin
    auto fbh = (fatBinaryHeader *) image;
    if (fbh->magic == FATBIN_MAGIC_NUMBER) {
        return CUDA_MODULE_TYPE::FATBIN;
    }

    // Test if it is in-memory elf format
    auto hdr = (Elf64_Ehdr *) image;
    if (hdr->e_ident[EI_MAG0] == ELFMAG0 && hdr->e_ident[EI_MAG1] == ELFMAG1 ||
        hdr->e_ident[EI_MAG2] == ELFMAG2 && hdr->e_ident[EI_MAG3] == ELFMAG3) {
        return CUDA_MODULE_TYPE::ELF;
    }

    // Test if it is ptx string
    std::string image_str((char *)image);
    if (containsSubstring(image_str, ".target")) {
        return CUDA_MODULE_TYPE::PTX_STRING;
    }

    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Cannot identify cuda module.");
}

std::string get_fatbin_str_from_ptx_str(std::string &ptx_str)
{
    write_str_to_file("/tmp/output.ptx", ptx_str);

    std::string compute_cap = std::string(CUDA_COMPUTE_CAPABILITY);

    std::string virtual_arch = "-gencode arch=compute_" + compute_cap + ",code=compute_" + compute_cap;
    std::string real_arch = "-gencode arch=compute_" + compute_cap + ",code=sm_" + compute_cap;

    std::string compile_cmd = "nvcc /tmp/output.ptx --fatbin " + virtual_arch + " " +
                              real_arch + " -o /tmp/output.fatbin";
                            
    auto res = exec(compile_cmd);

    if (res.second != 0) {
        throw std::runtime_error("Fail to compile PTX.");
    }

    std::ifstream ifs("/tmp/output.fatbin", std::ios::binary);
    auto fatbin_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    // std::remove("/tmp/output.ptx");
    // std::remove("/tmp/output.fatbin");

    return fatbin_str;
}

/* Extract all ptx code from cubin file, 
   return a vector of the generated file names */
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path)
{
    std::vector<std::string> ptx_file_names;
    auto candidate_cuda_compute_capabilities = get_candidate_cuda_compute_capabilities();
    bool has_failed = false;

    for (auto &capability : candidate_cuda_compute_capabilities) {

        auto arch_str = "-arch sm_" + std::string(capability);
        auto output = exec("cuobjdump " + cubin_path + " -lptx " + arch_str);

        std::stringstream ss(output.first);
        std::string line;

        while (std::getline(ss, line, '\n')) {
            if (containsSubstring(line, ".ptx")) {
                auto split_str = splitOnce(line, ":");
                auto ptx_file_name = strip(split_str.second);
                ptx_file_names.push_back(ptx_file_name);
            }
        }

        if (ptx_file_names.empty()) {
            // TALLY_SPD_WARN("Fail to find ptx code from " + cubin_path + " for compute capability " + capability);
            has_failed = true;
            continue;
        }

        if (has_failed) {
            TALLY_SPD_WARN("Fall back to use PTX code for compute capability " + capability);
        }

        exec("cuobjdump " + arch_str + " -xptx all " + cubin_path);
        break;
    }

    return ptx_file_names;
}

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf_str(std::string elf_code_str)
{
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

    return kernel_names_and_param_sizes;
}

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf(std::string elf_path)
{
    std::ifstream elf_file(elf_path);
    if (!elf_file.is_open()) {
        std::cerr << elf_path << " not found in get_kernel_names_and_param_sizes_from_elf" << std::endl;
        throw std::runtime_error("get_kernel_names_and_param_sizes_from_elf file not found");
    }
    
    std::string elf_code_str((std::istreambuf_iterator<char>(elf_file)), std::istreambuf_iterator<char>());
    elf_file.close();
    
    return get_kernel_names_and_param_sizes_from_elf_str(elf_code_str);
}