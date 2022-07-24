#!/bin/bash -x

#SBATCH --output=laion4m_in1k_coca_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=laion4m_in1k_coca
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash "/scratch/bf996/open_clip/src/script/run-singularity.bash" \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src"; python -u /scratch/bf996/open_clip/src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01200}.tar" --train-num-samples 12000000 --zeroshot-frequency=8 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --ds-filter="imagenet_classnames" --save-frequency 1 --report-to wandb --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=coca'