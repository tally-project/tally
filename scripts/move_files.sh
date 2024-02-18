#!/bin/bash

mv tally_profile_cpu.cpp tally_profile_gpu.cpp ../src/tally/preload
mv tally_client.cpp  ../src/tally/preload/generated
mv cuda_api.cpp server.cpp ../src/tally/generated
mv server.h cuda_api.h cuda_api_enum.h msg_struct.h ../include/tally/generated