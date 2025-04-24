#!/bin/bash
#SBATCH -t 03:10:00

#SBATCH -J BNL-AI

#SBATCH -o %j.out
#SBATCH -e %j.err

#SBATCH -p gpua100

python3 train.py