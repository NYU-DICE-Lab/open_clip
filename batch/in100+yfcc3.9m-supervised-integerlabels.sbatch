#!/bin/bash -x

#SBATCH --output=in100yfcc39m-sup-int-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=in100yfcc39m-sup-int
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash "/scratch/bf996/open_clip/src/script/run-singularity-rocm.bash" \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/in100_yfcc39m_4091404.csv" --csv-img-key path --csv-caption-key idx --csv-separator "," --integer-labels --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --caption-subset=True --sim-clr-trans=True --warmup 2000 --batch-size=64 --norm_gradient_clip=5e4 --precision=fp32 --epochs=64 --workers=4 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/in100+yfcc39m-supervised-integerlabels-ep1-28/checkpoints/epoch_29.pt" --local-loss --gather-with-grad'