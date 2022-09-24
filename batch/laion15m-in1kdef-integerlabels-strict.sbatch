#!/bin/bash -x

#SBATCH --output=laion-in1kdef-int-strict-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=laion-in1kdef-int-strict
#SBATCH --dependency=afterany:24533106
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
export NCCL_P2P_LEVEL=NVL

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --dataset-type webdataset --integer-labels --ds-filter="imagenet_def_classnames" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 4 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/laion15m-in1k-default-integerlabels-strict-ep1-12/checkpoints/epoch_12.pt" --norm_gradient_clip=5e4 --gather-with-grad --local-loss'