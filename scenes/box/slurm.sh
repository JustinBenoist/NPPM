#!/bin/bash
#SBATCH --time=23:30:00
#SBATCH --account=def-agruson
#SBATCH --mem=64000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

module load StdEnv/2020
module load cuda/11.7
module load python/3.10.2

cd /home/jbenoist/projects/def-agruson/jbenoist/photon_mapper
source .venv/bin/activate

sh scenes/mitsuba_CPPM_scenes/box/compute_gt.sh