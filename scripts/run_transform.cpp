#include <string>
#include <iostream>

#include <tally/transform.h>
#include <tally/util.h>

int main() {

    auto transformed = gen_preemptive_ptb_ptx("elementwise.1.sm_86.ptx");
    std::cout << transformed << std::endl;

    return 0;
}