#!/bin/bash -x

#SBATCH --output=in100yfcc22m-titletagdescr-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=in100yfcc22m-titletagdescr
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
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/in100_yfcc22m_2396660.csv" --csv-img-key path --csv-caption-key caption --csv-separator "," --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 4 --caption-subset=True --sim-clr-trans=True --warmup 2000 --batch-size=128 --epochs=64 --workers=4 --model=RN50 --resume "/scratch/bf996/open_clip/logs/in100+yfcc22m-titletagdescr-ep1-52/checkpoints/epoch_52.pt" --local-loss --gather-with-grad'