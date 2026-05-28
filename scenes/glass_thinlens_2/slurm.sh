#!/bin/bash
#SBATCH --time=23:30:00
#SBATCH --account=def-agruson
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=2

module load python/3.12
module load cuda

cd /home/jbenoist/links/projects/def-agruson/jbenoist/photon_mapper
source .venv/bin/activate

sh scenes/mitsuba_CPPM_scenes/glass_thinlens/compute_gt.sh