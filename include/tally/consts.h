#ifndef TALLY_CONSTS_H
#define TALLY_CONSTS_H

#include <string>
#include <vector>

// Hacky way to check whether the process should use the preload library
// Later we will change this to laze-loading of client
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
        "ldconfig", \
        "cuobjdump", \
        "uname", \
        "file -b", \
        "lscpu", \
        "sysctl", \
        "dmesg", \
        "ninja", \
        "/usr/bin/python3 /home/zhaowe58/.local/lib/python3.10/site-packages/cpuinfo/cpuinfo.py --json", \
        "cat /proc/cpuinfo", \
        "nvidia-smi" \
    };

#endif // TALLY_CONSTS_H