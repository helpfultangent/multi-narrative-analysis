#!/bin/bash
set -xe

cd ${_tapisExecSystemInputDir}
python /code/main.py interview.json  --output-dir ${_tapisExecSystemOutputDir}/