#!/bin/bash

pid=$(ps -ef | grep iox-roudi | grep -v grep | awk '{print $2}')
kill -15 $pid > /dev/null 2>&1

if [ ! -z "$pid" ]
then
    while [ -e /proc/$pid ]
    do
        sleep 0.1
    done
fi