#!/bin/bash

python3 p1_inference.py --image_dir $1 --output_json $2
# python3 evaluate.py --pred_file $2