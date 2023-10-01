#!/bin/bash

pid=$(ps -ef | grep iox-roudi | grep -v grep | awk '{print $2}')
kill -9 $pid > /dev/null 2>&1