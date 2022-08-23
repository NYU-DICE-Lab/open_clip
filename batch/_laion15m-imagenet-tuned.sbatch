#!/bin/bash -x

#SBATCH --output=laion15m-imagenet-tuned-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:mi50:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=laion15m-imagenet-tuned
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
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --imagenet-tune-freq=2 --imagenet-train="/imagenet/train" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --model=RN50 --local-loss --gather-with-grad'