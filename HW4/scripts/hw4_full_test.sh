#!/bin/bash

echo "=== Step 1: Downloading data ==="
echo "Start time: $(date)"
bash hw4_download.sh
echo "End time: $(date)"
echo "==============================="

echo "=== Step 2: Running hw4.sh ==="
echo "Start time: $(date)"
time bash hw4.sh /home/nas/polin/DLCV/dlcv-fall-2024-hw4-berlin0308/hw4_data/dataset_wo_images output_images
echo "End time: $(date)"
echo "==============================="

echo "=== Step 3: Evaluating predictions ==="
time python3 grade.py output_images /home/nas/polin/DLCV/dlcv-fall-2024-hw4-berlin0308/hw4_data/public_test/images
echo "==============================="

echo "=== Step 4: Running hw4.sh for private dataset ==="
echo "Start time: $(date)"
time bash hw4.sh /home/nas/polin/DLCV/dlcv-fall-2024-hw4-berlin0308/hw4_data/private_test output_images_private
echo "End time: $(date)"
echo "==============================="