#ifndef UTIL_H
#define UTIL_H

inline std::string demangleFunc(std::string mangledName)
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

inline std::string exec(std::string cmd) {
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

inline std::pair<std::string, std::string> splitOnce(const std::string& str, const std::string& delimiter) {
    std::size_t pos = str.find(delimiter);
    if (pos != std::string::npos) {
        std::string first = str.substr(0, pos);
        std::string second = str.substr(pos + delimiter.length());
        return {first, second};
    }
    return {str, ""};
}

inline std::string strip(const std::string& str) {
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

inline bool startsWith(const std::string& str, const std::string& prefix) {
    return str.compare(0, prefix.length(), prefix) == 0;
}

inline bool containsSubstring(const std::string& str, const std::string& substring) {
    return str.find(substring) != std::string::npos;
}


#endif // UTIL_H