#!/bin/bash

root=$(pwd)

echo "Attempting to build null implementation" 
cd src/first_implementation
rm -rf build; mkdir -p build; cd build
cmake ../ > /dev/null; make
cd $root
