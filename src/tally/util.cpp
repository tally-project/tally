#include <string>
#include <cxxabi.h>
#include <iostream>
#include <array>
#include <fstream>
#include <unistd.h>

#include <tally/util.h>

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

bool is_process_running(pid_t pid) {
    struct stat st;
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d", pid);

    if (stat(path, &st) == 0)
        return true;
    
    return false;
}

std::pair<std::string, int> exec(std::string cmd) {
    std::array<char, 128> buffer;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error executing command: " << cmd << std::endl;
        return std::make_pair("", -1);
    }

    std::string result;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    
    int status = pclose(pipe);
    if (status == -1) {
        return std::make_pair("", -1);
    }

    return std::make_pair(result, WEXITSTATUS(status));
}

void launch_shell(std::string cmd)
{
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error executing command: " << cmd << std::endl;
    }
    pclose(pipe);
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

int32_t countLeftBrace(const std::string& str) {
    int32_t count = 0;
    for (char c : str) {
        if (c == '{') {
            count += 1;
        }
    }
    return count;
}

int32_t countRightBrace(const std::string& str) {
    int32_t count = 0;
    for (char c : str) {
        if (c == '}') {
            count += 1;
        }
    }
    return count;
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

std::string strip_space_and_colon(const std::string& input) {
    std::string result;
    for (char c : input) {
        if (c != ' ' && c != ':') {
            result += c;
        }
    }
    return result;
}

bool startsWith(const std::string& str, const std::string& prefix) {
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool containsSubstring(const std::string& str, const std::string& substring) {
    return str.find(substring) != std::string::npos;
}

void write_str_to_file(std::string path, std::string str)
{
    std::ofstream file(path);
    file << str;
    file.close();
}

void write_binary_to_file(std::string path, const char* data, uint32_t size)
{
    std::ofstream file(path, std::ios::binary); // Open the file in binary mode
    file.write(data, size);
    file.close();
}

std::string get_tmp_file_path(std::string suffix)
{
    std::string tmp_file = "/tmp/tmp_" + std::to_string(getpid()) + suffix;
    return tmp_file;
}