#include <string>
#include <iostream>

#include <tally/transform.h>
#include <tally/util.h>

int main() {

    std::ifstream t("elementwise_with_cond.ptx");
    std::string str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());

    // auto ptx_file = std::string("elementwise_with_cond.ptx");
    auto transformed = gen_sync_aware_kernel(str);
    std::cout << transformed << std::endl;

    return 0;
}