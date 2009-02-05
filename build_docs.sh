#!/bin/bash

mkdir -p docs/html
mkdir -p docs/pdf

export PYTHONPATH=src:$PYTHONPATH

epydoc -v -o docs/html --html pyvision
# epydoc -v -o docs/pdf  --pdf  pyvision
