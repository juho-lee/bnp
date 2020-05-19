#!/bin/sh

python run_bo.py --mode bo --model anp
python run_bo.py --mode bo --model banp
python run_bo.py --mode bo --model bnp
python run_bo.py --mode bo --model canp
python run_bo.py --mode bo --model cnp
python run_bo.py --mode bo --model np

