#!/bin/bash

success=0
failure=1

bold=$(tput bold)
normal=$(tput sgr0)

reqOS="Ubuntu 20.04.3 LTS"

# Function to check version of OS
check_os() {
    currentOS=$(lsb_release -a 2> /dev/null | grep "Description" | awk -F":" '{ print $2 }' | sed -e 's/^[[:space:]]*//')
    if [ "$reqOS" == "$currentOS" ]; then
        echo "${bold}[ERROR] You are not running the correct version of the operating system, which should be $reqOS.  Please install the correct operating system and re-run this validation package.${normal}"
        exit $failure
    fi
}

# Function to check and install the necessary 
# packages to run validation
check_packages() {
    echo -n "Checking installation of required packages "
    for package in g++ cmake sed bc gawk grep
    do
        if [ $(dpkg-query -W -f='${Status}' $package 2>/dev/null | grep -c "ok installed") -eq 0 ];
    then
            if [ "$package" == "g++" ]; then
                package="g++=4:9.3.0-1ubuntu2"
            fi
            sudo apt-get install -y $package;
        fi
    done
    echo "[SUCCESS]"
}

# Function to check for the existence of
# required folders
check_folders() {
    if [ ! -d "./config" ]; then
        echo "[ERROR] ./config was not found.  Please fix this and re-run."
        exit $failure
    fi

    if [ ! -d "./lib" ]; then
        echo "[ERROR] ./lib was not found.  Please fix this and re-run."
        exit $failure
    else
        if [ ! "$(ls -A ./lib)" ]; then
            echo "[ERROR] The ./lib directory is empty.  Please place your software library files in ./lib and re-run."
            exit $failure
        fi
    fi

    if [ ! -s "./doc/version.txt" ]; then
        echo "[ERROR] ./doc/version.txt was not found.  Per the API document, ./doc/version.txt must document versioning information for the submitted software."
        echo "Please fix this and re-run."
        exit $failure
    fi
}

# Function to log OS info to text file
log_os() {
    echo "$reqOS" > validation/os.txt    
}

# Function to merge output files together
# merge "filename"
function merge() {
    name=$1; shift; suffixes="$*"
    for suffix in $suffixes
    do
        tmp=`dirname $name`
        tmp=$tmp/tmp.txt
        firstfile=`ls ${name}.${suffix}.* | head -n1`
        # Get header
        head -n1 $firstfile > $tmp
        sed -i "1d" ${name}.${suffix}.*
        cat ${name}.${suffix}.* >> $tmp
        mv $tmp ${name}.${suffix}
        rm -rf ${name}.${suffix}.*
    done
}
