#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ "$(hostname -s)" =~ ^g[r,v,a] ]]; then nv="--nv"; fi

singularity \
  exec $nv \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /scratch/bf996/datasets/fgvc-aircraft-2013b.sqf:ro \
  --overlay /scratch/bf996/datasets/flowers-102.sqf:ro \
  --overlay /scratch/bf996/datasets/stanford_cars.sqf:ro \
  --overlay /scratch/bf996/datasets/food-101.sqf:ro \
  --overlay /scratch/bf996/datasets/in100.sqf:ro \
  --overlay /scratch/bf996/datasets/laion100.sqf:ro \
  --overlay /scratch/bf996/datasets/openimages100.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"
