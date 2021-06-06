#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

python3 run.py
python3 run.py --num_prev_chars 12
python3 run.py --num_prev_chars 20
