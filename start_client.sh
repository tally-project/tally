#!/bin/bash

cp .tally_cache .tally_cache_client
LD_PRELOAD=./build/libtally_client.so $@