#!/bin/bash
#SBATCH --array=[0-49]
#SBATCH --time=23:59:00
#SBATCH --account=def-agruson
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

module load StdEnv/2020
module load cuda/11.7
module load python/3.10.2

cd /home/jbenoist/projects/def-agruson/jbenoist/photon_mapper
source .venv/bin/activate

export OPTIX_CACHE_PATH=$SLURM_TMPDIR

sh scenes/better_dataset_2/compute_gt.sh $SLURM_ARRAY_TASK_ID