#!/bin/bash
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=matteo.rapa@student.uva.nl

source /home/matteor/anaconda3/etc/profile.d/conda.sh
conda activate base

cd $HOME/thesis

python main.py --model resnet50 --epoch 10 --out ./results/resnet50/run_1.pth
