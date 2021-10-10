#!/bin/sh
#SBATCH -J GAMENet
#SBATCH -c 1
#SBATCH -n 12
#SBATCH -x dell-gpu-30
#SBATCH -o ./logs/GAMNet.log
#SBATCH -e ./logs/GAMNet.log


python train_GAMENet.py --batch_size=32 --learning_rate=0.001 --hidden_size=64 --epoch=50