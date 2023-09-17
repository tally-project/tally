#include <string>
#include <iostream>

#include <tally/transform.h>
#include <tally/util.h>

int main() {

    auto transformed = gen_ptb_ptx("a.1.sm_86.ptx");
    std::cout << transformed << std::endl;

    return 0;
}