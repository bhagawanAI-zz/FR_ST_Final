#!/bin/bash

source ../common/scripts/utils.sh

# Make sure there aren't any zombie processes
# left over from previous validation run
kill -9 $(ps -aef | grep "count_thread" | awk '{ print $2 }') 2> /dev/null

configDir=config
if [ ! -e "$configDir" ]; then
	echo "${bold}[ERROR] Missing $configDir folder!${normal}"
	exit $failure	
fi

outputDir=validation
templatesDir=$outputDir/templates
rm -rf $outputDir 
mkdir -p $templatesDir

# Usage: ../bin/validate11 createTemplate|match [-x enroll|verif] -c configDir -o outputDir -h outputStem -i inputFile -t numForks -j templatesDir
#
#   createTemplate|createMultiTemplates|match: task to process
#	    createTemplate: generate single template from one or more images of the same person
#       createMultiTemplates : generate multiple templates of one or more people from a single image
#	        - enroll: generate enrollment templates
#	        - verif: generate verification templates
#	    match: perform matching of templates
#
#   configDir: configuration directory
#   outputDir: directory where output logs are written to
#   outputStem: the string to prefix the output filename(s) with
#   inputFile: input file containing images to process (required for enroll and verif template creation)
#   numForks: number of processes to fork
#   templatesDir: directory where templates are written to/read from
echo "------------------------------"
echo " Running 1:1 validation"
echo "------------------------------"
    
# Set number of child processes to fork()
numForks=1
inputFile=input/short_enroll.txt
outputStem=enroll

echo -n "Checking for hard-coded config directory "
tempConfigDir=otherConfig
chmod 775 $configDir; mv $configDir $tempConfigDir; chmod 550 $tempConfigDir
bin/validate11 createTemplate -x enroll -c $tempConfigDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retEnroll=$?
if [[ $retEnroll == 0 ]]; then
    echo "[SUCCESS]" 
    # Merge output files together
    merge $outputDir/$outputStem log
else
	chmod 775 $tempConfigDir
	mv $tempConfigDir $configDir
    echo "[ERROR] Detection of hard-coded config directory in your software.  Please fix!"
    exit $failure
fi
rm -rf $outputDir; mkdir -p $templatesDir
chmod 775 $tempConfigDir; mv $tempConfigDir $configDir; chmod 550 $configDir

echo -n "Creating Enrollment Templates (Single Process) "
# Start checking for threading
../common/scripts/count_threads.sh validate11 $outputDir/thread.log & pid=$!

inputFile=input/enroll.txt
bin/validate11 createTemplate -x enroll -c $configDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retEnroll=$?

# End checking for threading
kill -9 "$pid"
wait "$pid" 2>/dev/null

if [[ $retEnroll == 0 ]]; then
	echo "[SUCCESS]" 
	# Merge output files together
	merge $outputDir/$outputStem log
else
	echo "${bold}[ERROR] Enrollment template creation validation (single process) failed${normal}"
	exit $failure
fi

maxThreads=$(cat $outputDir/thread.log | sort -u -n | tail -n1)
# 1 process for testdriver, 1 process for child
if [ "$maxThreads" -gt "2" ]; then
	echo "${bold}[WARNING] We've detected that your software may be threading or using other multiprocessing techniques during template creation.  The number of threads detected was $maxThreads and it should be 2.  Per the API document, implementations must run single-threaded.  In the test environment, there is no advantage to threading, because NIST will distribute workload across multiple blades and multiple processes.  We highly recommend that you fix this issue prior to submission.${normal}"
fi
rm -rf $outputDir; mkdir -p $templatesDir

echo -n "Creating Enrollment Templates on Multiple Images per Subject (Single Process) "
inputFile=input/enroll_multiface.txt
bin/validate11 createTemplate -x enroll -c $configDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retEnroll=$?
if [ $retEnroll -eq 0 ]; then
    echo "[SUCCESS]"
    # Merge output files together
    merge $outputDir/$outputStem log
else
    echo "${bold}[ERROR] Enrollment validation (multiple images per subject) failed${normal}"
    exit $failure
fi
rm -rf $outputDir; mkdir -p $templatesDir

inputFile=input/enroll.txt
numForks=4
echo -n "Creating Enrollment Templates (Multiple Processes) "
bin/validate11 createTemplate -x enroll -c $configDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retEnroll=$?
if [[ $retEnroll == 0 ]]; then
	echo "[SUCCESS]"
	# Merge output files together
	merge $outputDir/$outputStem log
else
	echo "${bold}[ERROR] Enrollment template creation validation (multiple process) failed.  Please ensure your software is compatible with fork(2).${normal}"
	exit $failure
fi

echo -n "Creating Verification Templates (Multiple Processes) "
inputFile=input/verif.txt
outputStem=verif
bin/validate11 createTemplate -x verif -c $configDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retVerif=$?
if [[ $retVerif == 0 ]]; then
	echo "[SUCCESS]" 
	# Merge output files together
	merge $outputDir/$outputStem log
else
	echo "${bold}[ERROR] Verification template creation validation failed{$normal}"
	exit $failure
fi

echo -n "Matching Templates (Multiple Processes) "
inputFile=input/match.txt
outputStem=match
bin/validate11 match -c $configDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retMatch=$?
if [[ $retMatch == 0 ]]; then
	echo "[SUCCESS]"
	# Merge output files together
	merge $outputDir/$outputStem log
else
	echo "${bold}[ERROR] Match validation failed${normal}"
	exit $failure
fi

echo -n "Creating Verification Templates for Multiple Persons Detected in an Image (Single Process) "
outputStem=verif_multiperson
inputFile=input/$outputStem.txt
numForks=1
bin/validate11 createMultiTemplates -x verif -c $configDir -o $outputDir -h $outputStem -i $inputFile -t $numForks -j $templatesDir
retVerifMulti=$?
if [[ $retVerifMulti == 0 ]]; then
    echo "[SUCCESS]"
    # Merge output files together
    merge $outputDir/$outputStem log
else
    echo "${bold}[ERROR] Enrollment validation (multiple persons in image) failed${normal}"
    exit $failure
fi

echo -n "Matching Multi-person Templates (Single Process) "
inputFileMulti=input/match_multiperson.txt
rm -f $inputFileMulti
sed '1d' $inputFile | awk '{ print $1 }' | while read line
do
    grep "${line}_" $outputDir/$outputStem.log | awk -v enroll=$line '{ print enroll".template " $1".template" }' >> $inputFileMulti
done
outputStem=match_multiperson
bin/validate11 match -c $configDir -o $outputDir -h $outputStem -i $inputFileMulti -t $numForks -j $templatesDir
retMatch=$?
if [[ $retMatch == 0 ]]; then
	echo "[SUCCESS]"
	# Merge output files together
	merge $outputDir/$outputStem log
else
	echo "${bold}[ERROR] Match validation failed${normal}"
	exit $failure
fi
