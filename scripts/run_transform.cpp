#include <string>
#include <iostream>

#include <tally/transform.h>
#include <tally/util.h>

int main() {

    auto ptx_file = std::string("tmp_910508.1.sm_86.ptx");
    auto transformed = gen_transform_ptx(ptx_file);
    std::cout << transformed << std::endl;

    return 0;
}