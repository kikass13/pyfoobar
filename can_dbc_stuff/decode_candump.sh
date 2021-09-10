#!/bin/bash

DBC_FILE=$1
CANDUMP_LOG=$2

modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

(candump vcan0 | cantools decode $DBC_FILE) &
PID=$!
cat $CANDUMP_LOG | canplayer -t vcan0=can0 

sleep 1

kill $PID