#!/bin/bash

if pgrep tally_server > /dev/null; then
    exit 0
else
    exit 1
fi