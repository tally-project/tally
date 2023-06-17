#!/bin/bash

LD_PRELOAD=./libpreload_slice.so ./tests/elementwise
LD_PRELOAD=./libpreload_slice.so ./tests/matmul