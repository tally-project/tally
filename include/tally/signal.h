#ifndef TALLY_SIGNAL_H
#define TALLY_SIGNAL_H

#include <tally/env.h>

#include <csignal>

void __attribute__((constructor)) register_signal_handler();

void warmup_signal_handler(int num);

#endif // TALLY_SIGNAL_H