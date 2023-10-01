#!/bin/bash

IOX_ROUDI_BIN=${HOME%%/}/tally/build/iox-roudi

if [[ ! -z "$TALLY_HOME" ]]; then
    IOX_ROUDI_BIN=${TALLY_HOME%%/}/tally/build/iox-roudi
fi

$IOX_ROUDI_BIN