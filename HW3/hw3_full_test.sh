#!/bin/bash

echo "=== Step 1: Downloading data ==="
echo "Start time: $(date)"
bash hw3_download.sh
echo "End time: $(date)"
echo "==============================="

echo "=== Step 2: Running hw3_1.sh ==="
echo "Start time: $(date)"
time bash hw3_1.sh /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p1_data/images/val p1_pred.json
echo "End time: $(date)"
echo "==============================="

echo "=== Step 3: Evaluating hw3_1 predictions ==="
time python3 evaluate.py --pred_file p1_pred.json --images_root /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p1_data/images/val --annotation_file /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p1_data/val.json
echo "==============================="

echo "=== Step 4: Running hw3_2.sh ==="
echo "Start time: $(date)"
time bash hw3_2.sh /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p2_data/images/val p2_pred.json /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p2_data/decoder_model.bin
echo "End time: $(date)"
echo "==============================="

echo "=== Step 5: Evaluating hw3_2 predictions ==="
time python3 evaluate.py --pred_file p2_pred.json --images_root /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p2_data/images/val --annotation_file /home/nas/polin/DLCV/dlcv-fall-2024-hw3-berlin0308/hw3_data/p2_data/val.json
echo "==============================="
