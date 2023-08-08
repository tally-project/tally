#include <iostream>

#include <tally/signal.h>
#include <tally/env.h>

void register_signal_handler()
{
    if (!TALLY_INITIALIZED) {
        register_env_vars();
    }

    // std::signal(SIGUSR1, warmup_signal_handler);
}

void warmup_signal_handler(int num)
{}