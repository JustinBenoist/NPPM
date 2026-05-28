#!/bin/bash
#SBATCH --time=23:30:00
#SBATCH --account=def-agruson
#SBATCH --mem=64000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=2

module load python/3.12
module load cuda

cd /home/jbenoist/links/projects/def-agruson/jbenoist/photon_mapper
source .venv/bin/activate

export OPTIX_CACHE_PATH=$SLURM_TMPDIR

sh scenes/scene_crab_thinlens/compute_gt.sh