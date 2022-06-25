#!/bin/bash

process=$1
log=$2
while true; do
	top -H -b -n1 | grep $process | wc -l >> $log
	sleep 10
done
