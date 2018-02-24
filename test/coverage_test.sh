#!/bin/bash

platform=`uname`

if [[ "$platform" == "Linux" ]]; then
    COVERAGE_TOOL=coverage
    OPEN_TOOL=xdg-open
elif [[ "$platform" == "Darwin" ]]; then
    COVERAGE_TOOL=coverage
    OPEN_TOOL=open
else
    COVERAGE_TOOL=coverage
fi 


echo Running face detect script...
$COVERAGE_TOOL run pyvision_unittests.py

$COVERAGE_TOOL html `find ../src/pyvision -name "*.py"`
$OPEN_TOOL htmlcov/index.html
