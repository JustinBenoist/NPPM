#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=def-agruson
#SBATCH --mem=64000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

module load StdEnv/2020
module load cuda
module load python/3.10.2

cd /home/jbenoist/projects/def-agruson/jbenoist/photon_mapper
source .venv/bin/activate

sh scenes/swimming_pool/compute_gt.sh