#!/bin/bash -x

#SBATCH --output=litswin_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_swin_laion400m_4node.sbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --dependency=afterany:22282747

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
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --save-frequency 1 --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{10000..41400}.tar" --resume "/scratch/bf996/open_clip/logs/litswin-imagenetsubset50m-ep1-8/checkpoints/epoch_8.pt" --train-num-samples 100000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --ds-filter="imagenet_classnames" --zeroshot-frequency=4 --warmup 2000 --batch-size=256 --wd=0.1 --epochs=16 --workers=8 --model=timm-swin_base_patch4_window7_224 --pretrained-image --lock-image --local-loss --gather-with-grad'
