#!/bin/bash
#SBATCH --job-name=ecDNA_optimization
#SBATCH --output=ecDNA_optimization_%j.out
#SBATCH --error=ecDNA_optimization_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=12:00:00


source /nas/longleaf/home/behnamie/miniconda3/etc/profile.d/conda.sh
conda activate ecDNA_env

srun python Optimization.py