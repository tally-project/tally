#ifndef TALLY_UTIL_H
#define TALLY_UTIL_H

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

std::string demangleFunc(std::string mangledName);
bool is_process_running(pid_t pid);
std::pair<std::string, int> exec(std::string cmd);
void launch_shell(std::string cmd);
std::pair<std::string, std::string> splitOnce(const std::string& str, const std::string& delimiter);
int32_t countLeftBrace(const std::string& str);
int32_t countRightBrace(const std::string& str);
std::string strip(const std::string& str);
bool startsWith(const std::string& str, const std::string& prefix);
bool containsSubstring(const std::string& str, const std::string& substring);
void write_str_to_file(std::string path, std::string str);
void write_binary_to_file(std::string path, const char* data, uint32_t size);
std::string strip_space_and_colon(const std::string& input);
std::string get_tmp_file_path(std::string suffix, int file_name=-1);

template <typename T>
void merge_vec(std::vector<T>& dest, const std::vector<T>& src) {
    dest.insert(dest.end(), src.begin(), src.end());
}

inline std::string get_process_name(int pid) {
    std::stringstream ss;
    ss << "/proc/" << pid << "/cmdline";
    std::ifstream commFile(ss.str());
    std::string processName;
    std::stringstream buffer;
    buffer << commFile.rdbuf();

    return buffer.str();
}

inline std::string replace_substring(std::string& input, const std::string& oldStr, const std::string& newStr) {

    std::string res = input;

    size_t startPos = 0;
    while ((startPos = res.find(oldStr, startPos)) != std::string::npos) {
        res.replace(startPos, oldStr.length(), newStr);
        startPos += newStr.length();
    }

    return res;
}

#endif // TALLY_UTIL_H