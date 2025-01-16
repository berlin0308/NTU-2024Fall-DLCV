cd /home/nas/polin/DLCV/dlcv-fall-2024-hw4-berlin0308/gaussian-splatting
python3 render.py -m output/ -s ../hw4_data/public_test --skip_test --white_background

cd ..
python3 grade.py gaussian-splatting/output/train/ours_30000/renders gaussian-splatting/output/train/ours_30000/gt