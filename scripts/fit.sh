#!/bin/bash

source activate
conda activate bayes_yplus

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1 $2"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit.py $1 $2 condor
rm -rf $tmpdir