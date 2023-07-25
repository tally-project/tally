#!/bin/bash

TALLY_CACHE=.tally_cache
TALLY_CACHE_CLIENT=.tally_cache_client

# Copy cache if not already exists
if [ -e "$TALLY_CACHE" ]; then
    if [ -e "$TALLY_CACHE_CLIENT" ]; then
        if ! cmp -s "$TALLY_CACHE" "$TALLY_CACHE_CLIENT"; then
            cp .tally_cache .tally_cache_client
        fi
    else
        cp .tally_cache .tally_cache_client
    fi
fi

LD_PRELOAD=./build/libtally_client.so $@