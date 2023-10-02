#!/bin/bash

TALLY_SERVER_BIN=${HOME%%/}/tally/build/tally_server

if [[ ! -z "$TALLY_HOME" ]]; then
    TALLY_SERVER_BIN=${TALLY_HOME%%/}/build/tally_server
fi

$TALLY_SERVER_BIN