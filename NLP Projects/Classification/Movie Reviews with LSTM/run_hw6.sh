#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env

python3 run.py
python3 run.py --l2 1e-4 --dropout 0.5
python3 run.py --lstm
python3 run.py --lstm --l2 1e-4 --dropout 0.5
