#!/bin/bash -x

#SBATCH --output=yfcc_intlabels_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc_intlabels
#SBATCH --dependency=afterany:22820629
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_single_3965496.csv" --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-singleclass-ep1-32-alt/checkpoints/epoch_32.pt" --csv-separator "," --integer-labels  --csv-caption-key="in1k_subset" --save-frequency 1 --warmup 2000 --batch-size=128 --precision=fp32 --epochs=64 --workers=8 --model=RN50-in1k --local-loss --gather-with-grad'