#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/ReactBench
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:l40s:1 
#SBATCH --mem=128GB
#SBATCH --job-name=reactbench 
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/project/aip-aspuru/aburger/ReactBench/outslurm/slurm-%j.txt 
#SBATCH --error=/project/aip-aspuru/aburger/ReactBench/outslurm/slurm-%j.txt

# activate venv
source ${PYTHONBIN}/activate

# get environment variables
source .env

#module load cuda/12.6
#module load gcc/12.3

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

# hand over all arguments to the script
echo "Submitting ${HOMEROOT}/$@"

${PYTHONBIN}/python ${HOMEROOT}/"$@"