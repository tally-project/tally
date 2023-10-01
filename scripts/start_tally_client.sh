#!/bin/bash

TALLY_CACHE=${HOME%%/}/.cache/tally/.tally_cache
TALLY_CACHE_CLIENT=${HOME%%/}/.cache/tally/.tally_cache_client

TALLY_CLIENT_LIB=${HOME%%/}/tally/build/libtally_client.so

if [[ ! -z "$TALLY_HOME" ]]; then
    TALLY_CLIENT_LIB=${TALLY_HOME%%/}/build/libtally_client.so
fi

# Copy cache if not already exists
if [ -e "$TALLY_CACHE" ]; then
    if [ -e "$TALLY_CACHE_CLIENT" ]; then
        if ! cmp -s "$TALLY_CACHE" "$TALLY_CACHE_CLIENT"; then
            cp $TALLY_CACHE $TALLY_CACHE_CLIENT
        fi
    else
        cp $TALLY_CACHE $TALLY_CACHE_CLIENT
    fi
fi

LD_PRELOAD=$TALLY_CLIENT_LIB $@