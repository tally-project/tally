#include <iostream>

#include <tally/signal.h>
#include <tally/env.h>

void register_signal_handler()
{
    if (!TALLY_INITIALIZED) {
        register_env_vars();
    }

    std::cout << "Registering Signal handlers" << std::endl;

    if (PROFILE_KERNEL_TO_KERNEL_PERF) {
        std::signal(SIGUSR1, warmup_signal_handler);
        std::cout << "Registered warmup signal at SIGUSR1" << std::endl;
    }
}

void warmup_signal_handler(int num)
{
    PROFILE_WARMED_UP = true;
    std::cout << "Profile is warmed up" << std::endl;
}