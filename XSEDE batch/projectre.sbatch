#!/bin/bash -x

#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=2
#SBATCH -p GPU
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=16
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=finish_train_network
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ajn313@nyu.edu

module purge;
set -x

#debug flags
echo $SLURM_JOB_NAME

#env vars
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#comments

#trained on iNaturalist with captions

#run command
srun --accel-bind=v \
    /bin/bash /ocean/projects/cis220033p/ajn313/src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python /ocean/projects/cis220033p/ajn313/src/training/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data "/ocean/projects/cis220033p/ajn313/data/datasets/inat_captions/{000000..000362}.tar" \
    --train-num-samples 2600000 \
    --inat2021 "/ocean/projects/cis220033p/ajn313/data/datasets/inat_val/" \
    --zeroshot-frequency=1 \
    --warmup 2000 \
    --batch-size=128 \
    --wd=0.1 \
    --epochs=24 \
    --workers=4 \
    --model=timm-vit_large_patch16_384_pretrained \
    --pretrained-image \
    --lock-image \
    --seed 0 \
    --resume "/ocean/projects/cis220033p/ajn313/logs/2022_07_10-10_31_02-model_timm-vit_large_patch16_384_pretrained-lr_0.0005-b_256-j_4-p_amp/checkpoints/epoch_18.pt" \
    --local-loss \
    --gather-with-grad'
