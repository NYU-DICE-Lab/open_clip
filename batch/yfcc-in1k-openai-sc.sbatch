#!/bin/bash -x

#SBATCH --output=yfcc-oai-intlabel-sc-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=yfcc-oai-intlabel-sc
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
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_openai_mc_2532271.csv" --zeroshot-frequency=8 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --csv-separator "," --csv-caption-key="title" --save-frequency 4 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=RN50 --local-loss --gather-with-grad'