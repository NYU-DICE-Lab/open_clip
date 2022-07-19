### Environment

#### Local

First navigate to the open_clip directory. Then run --

```bash
conda activate open_clip; export PYTHONPATH="$PYTHONPATH:$PWD/src"; 
#sometimes you also need
pip install -r requirements-training.txt; 
```

#### SLURM

```bash
cd /scratch/bf996/open_clip

singularity exec --nv \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
  --overlay /scratch/bf996/datasets/imagenetv2-matched-frequency-format-val.sqf:ro \
  --overlay /scratch/bf996/datasets/flowers-102.sqf:ro \
  --overlay /scratch/bf996/datasets/stanford_cars.sqf:ro \
  --overlay /scratch/bf996/datasets/food-101.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  --overlay /scratch/bf996/datasets/fgvc-aircraft-2013b.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash

source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src"; export PYTHONPATH="$PYTHONPATH:/home/bf996/.local/bin";
```

### Commands

#### 1 Node, 1 GPU

python -u src/training/main.py

#### 1 Node, Multi GPU

torchrun --nproc_per_node 2 -m training.main *ARGS

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main *ARGS

### Flags

#### Cosmetic

--name "ViT-B-32-Vanilla-l50m-03"

--report-to wandb,tensorboard

#### Hyperparameters

--save-frequency 1

--batch-size=4096

--epochs=3  

--workers=8

--workers=$OMP_NUM_THREADS

--lr=1e-3     

--wd=0.1

--warmup=1000

--precision=fp32

--precision='amp'

#### LiT Tuning

--lock-image

#### Gradient Caching

--batch-size=2048  --gc=True  --gpumaxbatch=128

#### Declip

--model=timm-efficientnetv2_rw_s  --mlm=True

#### Eval

--zeroshot-frequency=1

--food "/" 

--air "/"

--stanfordcars "/"

--flowers "/"

--imagenet-val "/imagenet/val/"

--imagenet-a "/imagenet-a" 

--imagenet-r "/imagenet-r"

--imagenet-v2 "/scratch/bf996/datasets"

--imagenet-s "/imagenet-sketch"

--inat2021 "/scratch/bf996/datasets/"

#### Dataset Type

--train-data="~/Train_GCC-training_output.csv"       --csv-img-key filepath     --csv-caption-key title

--train-data '/media/benfeuer/laion/laion400m/{00000..10000}.tar' --train-num-samples 50000000 --dataset-type webdataset

#### Dataset Modification

--ds-filter="imagenet_classnames"

--csv-cleaned=True

--csv-scrambled=True

--zeroshot-scramble=True

--ds-cipher=True

--simplecaptions=True

#### Model Selection

--model=RN50  --pretrained=openai

--resume "/scratch/bf996/open_clip/logs/yfcc_cars_01/checkpoints/epoch_32.pt"

--model=ViT-L-14 --pretrained=laion400m_e32

--model=ViT-B-32 --pretrained=laion2b_e16

--model=ViT-B-32 --pretrained=laion400m_e32

--model timm-vit_large_patch16_384_pretrained

--model timm-resnet50d

### pretrained model list

[('RN50', 'openai'),
 ('RN50', 'yfcc15m'),
 ('RN50', 'cc12m'),
 ('RN50-quickgelu', 'openai'),
 ('RN50-quickgelu', 'yfcc15m'),
 ('RN50-quickgelu', 'cc12m'),
 ('RN101', 'openai'),
 ('RN101', 'yfcc15m'),
 ('RN101-quickgelu', 'openai'),
 ('RN101-quickgelu', 'yfcc15m'),
 ('RN50x4', 'openai'),
 ('RN50x16', 'openai'),
 ('ViT-B-32', 'openai'),
 ('ViT-B-32', 'laion400m_e31'),
 ('ViT-B-32', 'laion400m_e32'),
 ('ViT-B-32', 'laion400m_avg'),
 ('ViT-B-32-quickgelu', 'openai'),
 ('ViT-B-32-quickgelu', 'laion400m_e31'),
 ('ViT-B-32-quickgelu', 'laion400m_e32'),
 ('ViT-B-32-quickgelu', 'laion400m_avg'),
 ('ViT-B-16', 'openai'),
 ('ViT-L-14', 'openai')]

### Example Commands

#### INFERENCE ON PRETRAINED MODEL CHECKPOINT

python src/training/main.py --batch-size=32 --workers=4 --imagenet-a "/Volumes/Temp/imagenet-a" --stanfordcars "/Volumes/Temp" --zeroshot-frequency=1  --model=RN50  --pretrained=openai  --precision=fp32

python src/training/main.py --batch-size=32 --workers=4 --imagenet-a "/Volumes/Temp/imagenet-a" --zeroshot-frequency=1  --model=RN50  --pretrained=openai  --precision=fp32

python src/training/main.py --batch-size=32 --workers=4 --imagenet-v2 "/Volumes/Temp" --imagenet-a "/Volumes/Temp/imagenet-a" --imagenet-s "/Volumes/Temp/sketch" --zeroshot-frequency=1  --model=RN50  --pretrained=yfcc15m  --precision=fp32

python src/training/main.py --batch-size=32 --workers=4 --inat2021="/scratch/bf996/datasets/" --zeroshot-frequency=1  --model=RN50  --pretrained=yfcc15m 

python -u /scratch/bf996/open_clip/src/training/main.py --stanfordcars "/scratch/bf996/datasets/stanfordcars" --flowers "/scratch/bf996/datasets/flowers102" --zeroshot-frequency=1 --workers=16 --model=RN50 --pretrained=yfcc15m

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-scramble=True --zeroshot-frequency=1 --workers=16 --model=RN50 --pretrained=openai

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/"  --imagenet-a "/imagenet-a"  --imagenet-r "/imagenet-r" --zs-upper=True --model=ViT-B-32 --pretrained=laion400m_e32

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/"  --imagenet-a "/imagenet-a"  --imagenet-r "/imagenet-r" --model=ViT-B-32 --pretrained=laion2b_e16

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-v2 "/scratch/bf996/datasets"  --imagenet-s "/imagenet-sketch" --model=RN50  --pretrained=yfcc15m > inf.txt

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --air "/" --stanfordcars "/" --food "/" --model=ViT-L-14 --pretrained=laion400m_e32


#### INFERENCE ON TRAINED MODEL CHECKPOINT

python src/training/main.py --batch-size=32 --workers=16 --resume "/scratch/bf996/open_clip/logs/swin-ep3,4/checkpoints/epoch_4.pt" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --zeroshot-frequency=1 --model=RN50

python src/training/main.py --batch-size=32 --workers=16 --resume "/scratch/bf996/open_clip/logs/swin-ep3,4/checkpoints/epoch_4.pt" --imagenet-val "/imagenet/val/" --zeroshot-frequency=1 --model=timm-swin_base_patch4_window7_224 --pretrained-image --lock-image

--imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" 

python src/training/main.py --batch-size=32 --workers=16 --resume "/scratch/bf996/open_clip/logs/declip-ViT-B-32-ep1-8/checkpoints/epoch_7.pt" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1 --model="vit_base_patch32_224" --mlm=True

python src/training/main.py --batch-size=32 --workers=16 --resume "/scratch/bf996/open_clip/logs/coca-ep4-7/checkpoints/epoch_7.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --zeroshot-frequency=1 --model="coca"

#### LINEAR PROBE

python src/training/main.py --batch-size=32 --workers=16 --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model="tf_efficientnet_b0" --zeroshot-frequency=1 --linear-probe=True --image-size=224

#### SINGLE NODE TRAINING

python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --csv-separator "," --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --simplecaptions=True --csv-cleaned=True --zeroshot-frequency=2 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=16 --workers=16 --debug --model=RN50

#### WDS TRAINING WITH FILTERING

python -u /scratch/bf996/open_clip/src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..10000}.tar" --train-num-samples 15000000 --imagenet-val "/imagenet/val/" --ds-filter="imagenet_classnames" --gc=True  --gpumaxbatch=128 --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=1024 --epochs=16 --workers=8 --model=ViT-B-32

#### WDS TRAINING WITHOUT FILTERING

python src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..10000}.tar" --train-num-samples 500000 --imagenet-val "/imagenet/val/" --zeroshot-frequency=2 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=128 --epochs=16 --workers=4 --debug --model=RN50

#### DISTRIBUTED TRAINING

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --report-to wandb --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --csv-separator "," --imagenet-val "/imagenet/val/" --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=384 --epochs=32 --workers=4 --model=ViT-B-32 --local-loss --gather-with-grad

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..00100}.tar" --train-num-samples 50000 --imagenet-val "/imagenet/val/" --ds-filter="imagenet_classnames" --csv-cleaned=True --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=384 --epochs=32  --model=ViT-B-32 --local-loss --gather-with-grad