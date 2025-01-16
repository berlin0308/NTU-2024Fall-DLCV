#!/bin/bash
export PYTHONPATH=./gaussian-splatting:$PYTHONPATH

python3 gaussian-splatting/render.py -s $1 --output_path $2 -m gaussian-splatting/output/  --skip_test --white_background

# bash hw4.sh hw4_data/dataset_wo_images output_test
