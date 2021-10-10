#!/bin/sh
#SBATCH -J MERITS
#SBATCH -c 1
#SBATCH -n 4
#SBATCH -o ./logs/MERITS.log
#SBATCH --gres gpu:1
#SBATCH -p dell

# python -u ablation/train_MERITS_Lin.py --params_path ./default_params.json
# python -u train_MERITS.py --params_path ./default_params.json
# python -u baseline/train_GAMENet.py ./default_params.json
python -u test_MERITS.py --params_path ../default_params.json
python -u test_GAMENet.py --params_path ../default_params.json
python -u test_Retain.py --params_path ../default_params.json