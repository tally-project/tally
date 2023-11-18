#!/bin/bash

TALLY_CLIENT_LIB=libtally_client_local.so

TALLY_CLIENT_LIB_PATH=${HOME%%/}/tally/build/$TALLY_CLIENT_LIB

if [[ ! -z "$TALLY_HOME" ]]; then
    TALLY_CLIENT_LIB_PATH=${TALLY_HOME%%/}/build/$TALLY_CLIENT_LIB
fi

LD_PRELOAD=$TALLY_CLIENT_LIB_PATH $@