#!/bin/bash

python3 p2_inference.py --noise_folder $1 --output_folder $2 --unet_model_path $3
# python3 p2_evaluate.py --gt_image_folder hw2_data/face/GT --gen_image_folder $2 