#!/bin/bash -x

#SBATCH --output=yfcc_coca_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=yfcc_coca
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME
export NCCL_P2P_LEVEL=NVL; 
export NCCL_BLOCKING_WAIT=1; 
export NCCL_DEBUG=INFO; 
export TORCH_CPP_LOG_LEVEL=INFO; 
export TORCH_DISTRIBUTED_DEBUG=INFO; 
export HCC_SERIALIZE_KERNEL=0x3; 
export HCC_SERIALIZE_COPY=0x3; 
export HIP_TRACE_API=0x2; 
export MIOPEN_ENABLE_LOGGING_CMD=1; 
export TORCH_SHOW_CPP_STACKTRACES=1; 
export NCCL_DEBUG_SUBSYS=ALL; 
export PYTHONFAULTHANDLER=1;

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
    'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src"; python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_simplecaption_nounadj.csv" --csv-separator "," --zeroshot-frequency=8 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --clamp=1e5 --save-frequency 1 --report-to wandb --warmup 2000 --batch-size=64 --epochs=32 --workers=8 --model=coca'