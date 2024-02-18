#include <tally/cache_struct.h>
#include <tally/cache.h>

void CubinData::compile() {

    if (compiled) {
        return;
    }

    std::string fatbin_tmp_path = get_tmp_file_path(".fatbin");
    write_binary_to_file(fatbin_tmp_path, cubin_data.c_str(), cubin_size);
    
    // Extract PTX code from cubin file
    auto ptx_file_names = gen_ptx_from_cubin(fatbin_tmp_path);

    if (ptx_file_names.size() > 1) {
        throw std::runtime_error("Currently assuming one fatbin only contains one ptx file");
    }

    if (ptx_file_names.size() > 0) {

        auto ptx_file_name = ptx_file_names[0];

        // Generate transformed version from the PTX code
        TALLY_SPD_LOG_ALWAYS("Generating transformed code for " + ptx_file_name + " ...");

        ptx_str = gen_transform_ptx(ptx_file_name);
        fatbin_str = get_fatbin_str_from_ptx_str(ptx_str);

        // Delete ptx
        std::remove(ptx_file_name.c_str());

    } else {

        TALLY_SPD_WARN("No PTX file found in fatbin data - kernel transformations will be disabled.");

        // ptx code will be empty
        ptx_str = "";

        std::ifstream ifs(fatbin_tmp_path, std::ios::binary);
        fatbin_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    }

    // Delete cubin file
    std::remove(fatbin_tmp_path.c_str());

    compiled = true;
}