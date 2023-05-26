#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=48
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=matteo.rapa@student.uva.nl

source /home/matteor/anaconda3/etc/profile.d/conda.sh
conda activate base

cd /home/matteor/msc-thesis-self-adaptive-neural-networks/src

python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0
python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0.05
python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0.10
python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0.15
# python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0.20
# python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0.25
# python main.py --model resnet50 --epoch 10 --out ../results/conv_only/ --prune 0.30