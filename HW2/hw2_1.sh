#!/bin/bash

python3 p1_inference.py --output_image_dir $1 --model_path combined_ddpm.pth
# python3 digit_classifier.py --folder $1 --checkpoint Classifier.pth