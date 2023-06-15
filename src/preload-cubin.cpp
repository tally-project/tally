
#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <cxxabi.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <thread>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

class Preload {

public:
    Preload(){}
    ~Preload(){}
};

Preload tracer;

std::string demangleFunc(std::string mangledName)
{
    int status;
    char *demangled_name = abi::__cxa_demangle(mangledName.c_str(), nullptr, nullptr, &status);
    
    if (status == 0) {
        std::string demangled_name_str(demangled_name);
        free(demangled_name);
        return demangled_name_str;
    } else {
        return mangledName;
    }
}

std::string exec(std::string cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::pair<std::string, std::string> splitOnce(const std::string& str, const std::string& delimiter) {
    std::size_t pos = str.find(delimiter);
    if (pos != std::string::npos) {
        std::string first = str.substr(0, pos);
        std::string second = str.substr(pos + delimiter.length());
        return {first, second};
    }
    return {str, ""};
}

std::string strip(const std::string& str) {
    std::string result = str;
    
    // Remove leading whitespace
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    
    // Remove trailing whitespace
    result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), result.end());

    return result;
}

extern "C" { 

struct __align__(8) fatBinaryHeader
{
    unsigned int           magic;
    unsigned short         version;
    unsigned short         headerSize;
    unsigned long long int fatSize;
};

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    static void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
    if (!l__cudaRegisterFunction) {
        l__cudaRegisterFunction = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    assert(l__cudaRegisterFunction);

    std::cout << "running l__cudaRegisterFunction" << std::endl;
    std::cout << demangleFunc(std::string(deviceFun)) << std::endl;

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {

    __fatBinC_Wrapper_t *wp = (__fatBinC_Wrapper_t *) fatCubin;

    int magic = wp->magic;
    int version = wp->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;

    std::cout << "fatCubin_data_size_bytes: " << fatCubin_data_size_bytes << std::endl;

    int i = 0;
    while (true) {
        std::string file_name = "output" + std::to_string(i) + ".cubin";
        std::ifstream infile(file_name);
        if (infile.good()) {
            i++;
            continue;
        }
        infile.close();

        std::ofstream file(file_name, std::ios::binary); // Open the file in binary mode
        file.write(reinterpret_cast<const char*>(wp->data), fatCubin_data_size_bytes);
        file.close();

        exec("cuobjdump -xptx all " + file_name);
        auto output = exec("cuobjdump " + file_name + " -lptx");
        std::stringstream ss(output);
        std::vector<std::string> lines;
        std::string line;

        while (std::getline(ss, line, '\n')) {
            lines.push_back(line);
        }

        // Print the split lines
        for (const auto& line : lines) {
            auto split_str = splitOnce(line, ":");
            auto ptx_file_name = strip(split_str.second);
            std::cout << ptx_file_name << std::endl;

            std::ifstream ptx_file(ptx_file_name);
            std::string ptx_code((std::istreambuf_iterator<char>(ptx_file)), std::istreambuf_iterator<char>());
            std::cout << ptx_code << std::endl;
            ptx_file.close();

        }

        break;
    }

    static void** (*l__cudaRegisterFatBinary) (void *);
    if (!l__cudaRegisterFatBinary) {
        l__cudaRegisterFatBinary = (void** (*) (void *)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    }
    assert(l__cudaRegisterFatBinary);

    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
	static void (*l__cudaRegisterFatBinaryEnd) (void **);
	if (!l__cudaRegisterFatBinaryEnd) {
		l__cudaRegisterFatBinaryEnd = (void (*) (void **)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");
	}
	assert(l__cudaRegisterFatBinaryEnd);

	l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}
        

}