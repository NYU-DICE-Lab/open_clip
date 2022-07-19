#!/bin/bash -x

#SBATCH --output=litvithuge_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_vit_huge_laion.sbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --dependency=afterany:22344813

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
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --save-frequency 1 --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{05000..41400}.tar" --resume "/scratch/bf996/open_clip/logs/2022_07_16-09_36_56-model_timm-vit_huge_patch14_224_in21k-lr_0.0005-b_256-j_8-p_amp/checkpoints/epoch_4.pt" --train-num-samples 50000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --ds-filter="imagenet_classnames" --zeroshot-frequency=4 --warmup 2000 --batch-size=256 --wd=0.1 --epochs=32 --workers=8 --model=timm-vit_huge_patch14_224_in21k --pretrained-image --lock-image --local-loss --gather-with-grad'
