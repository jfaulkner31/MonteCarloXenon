#!/bin/bash
#SBATCH --job-name=SIE_PWR
#SBATCH --time=140:00:00
#SBATCH --nodes=1
### #SBATCH --ntasks-per-node=384
#SBATCH --ntasks-per-node=384
#SBATCH --mem=0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --wckey=edu_res
#SBATCH --mail-user=jfaulkner31@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#module purge
#module load use.exp_ctl
#module load scale
#export DATA=/hpc-common/data/scale/6.3

# Create a clean host list for SCALE
#scontrol show hostnames "$SLURM_JOB_NODELIST" > nodelist.txt
#echo "Nodes allocated:"
#cat nodelist.txt
#echo "SLURM_NTASKS=$SLURM_NTASKS"
#cd "$SLURM_SUBMIT_DIR"

# Run PYTHON ---
conda activate schemes
export PYTHONPATH=/home/fauljona/MonteCarloXenon:$PYTHONPATH

# OpenMC setup
module load openmc
export OPENMC_CROSS_SECTIONS=/home/fauljona/ondemand/data/endfb-vii.1-hdf5/cross_sections.xml

# Removal of garbage before we start running
rm -fr results
rm -fr depl_results
rm -fr LOGGING.LOG
rm -fr *xml
rm -fr *h5

python sie_job.py >> LOGGING.LOG
