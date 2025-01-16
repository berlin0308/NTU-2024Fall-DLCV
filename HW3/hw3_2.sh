#!/bin/bash

python3 p2_inference.py --image_dir $1 --output_json $2 --decoder_path $3 --device cuda --checkpoint module_ep3_94_73.pt --search 1
# python3 p2/p2_evaluate.py --pred_file $2