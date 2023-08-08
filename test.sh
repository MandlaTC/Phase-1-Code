#!/bin/bash
#SBATCH --job-name=Test-Run-Phase1
#SBATCH --output=/home-mscluster/mchavarika/phase1/result.txt
#SBATCH --partition=stampede
/bin/hostname
cd phase1
source ~/.bashrc
conda activate tf2-gpu
cd Phase-1-Code/
pip install -r requirements.txt.
python phase1_training.py
