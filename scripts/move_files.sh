#!/bin/bash

mv tally_client.cpp tally_profile_cpu.cpp tally_profile_gpu.cpp ../src
mv cuda_api.cpp ../src/tally
mv cuda_api.h ../include/tally