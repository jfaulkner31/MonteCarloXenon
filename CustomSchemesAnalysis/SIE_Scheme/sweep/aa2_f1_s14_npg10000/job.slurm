#!/bin/bash
#SBATCH --job-name=aa2_f1_s14_npg10000
#SBATCH --time=120:00:00
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
# Python pytriton
#export PYTHONPATH=/home/fauljona/pytriton:$PYTHONPATH
# Create a clean host list for SCALE
#scontrol show hostnames "$SLURM_JOB_NODELIST" > nodelist.txt
#echo "Nodes allocated:"
#cat nodelist.txt
#echo "SLURM_NTASKS=$SLURM_NTASKS"
#cd "$SLURM_SUBMIT_DIR"
# Run PYTHON ---
conda activate schemes
export PYTHONPATH=/home/fauljona/MonteCarloXenon:$PYTHONPATH
#python -c "from pytriton.depleter.celiYaml import runCELIFromYAML as run; run(input='$1')"
module load openmc
export OPENMC_CROSS_SECTIONS=/home/fauljona/ondemand/data/endfb-vii.1-hdf5/cross_sections.xml
rm -fr LOGGING.LOG
rm -fr *xml
rm -fr *h5
mkdir results
python aa_singlestep_SWEEP.py >> LOGGING.LOG
