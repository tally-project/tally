#ifndef TALLY_CONSTS_H
#define TALLY_CONSTS_H

// Hacky way to check whether the process should use the preload library
#define NO_INIT_PROCESS_KEYWORDS_VEC \
    const std::vector<std::string> no_init_process_keywords = { \
        "-std=c++", \
        "--c++", \
        "-arch", \
        "--cicc-cmdline", \
        "/usr/bin/ld", \
        "-opt=", \
        "--64", \
        "/usr/bin/gcc", \
        "as-I", \
        "--fatbin", \
        "ptxas", \
        "nvcc", \
        "g++", \
        "gcc", \
        "fatbinary", \
        "ldconfig" \
    };

#endif // TALLY_CONSTS_H