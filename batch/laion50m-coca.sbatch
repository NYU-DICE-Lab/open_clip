#!/bin/bash -x

#SBATCH --output=laion50m_coca_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=34:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=laion50m_coca
#SBATCH --dependency=afterany:22739126
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --save-frequency 1 --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{07500..22500}.tar" --train-num-samples 50000000 --zeroshot-frequency=8 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --warmup 2000 --batch-size=32 --resume "/scratch/bf996/open_clip/logs/laion50m-swin-coca-ep4/checkpoints/epoch_4.pt" --epochs=32 --workers=4 --model=coca --norm_gradient_clip=5e4 --local-loss --gather-with-grad'