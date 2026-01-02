#!/bin/bash

scene_path="$1"
train_mode="splannequin"
[ "$2" = "--baseline" ] && train_mode="baseline"

if [ -z "$scene_path" ]; then
    echo "Error: Scene path required. (Usage: bash $0 <scene_path> [--baseline])"
    exit 1
fi
scene_name=$(basename "$scene_path")

setup_gpu() {
   echo "▶ Server: $(hostname)"
   read GPU_ID gpu_usage gpu_memory gpu_total _ < <(nvidia-smi \
       --query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free \
       --format=csv,noheader,nounits | sort -t',' -k5 -nr | head -1 | tr ',' ' ')
   echo "▶ GPU $GPU_ID : Utilization ${gpu_usage}%, Memory. $((gpu_memory * 100 / gpu_total))% (${gpu_memory}MB/${gpu_total}MB)"
   export CUDA_VISIBLE_DEVICES=$GPU_ID
   export PYTORCH_ALLOC_CONF="max_split_size_mb:512"
}

setup_gpu
echo "▶ Time:       $(date "+%Y-%m-%d %H:%M")"
echo "▶ Scene:      $scene_path"
echo "▶ Train mode: $train_mode"

python ./train.py \
   -s "${scene_path}" \
   -m "./exp/${train_mode}/${scene_name}" \
   --port 6017 \
   --eval \
   --train_mode "$train_mode" \
   --resolution 1 \
   --expname "${scene_name}_${train_mode}" \
   --configs ./arguments/${train_mode}.py

echo "▶ Execution completed at $(date "+%Y-%m-%d %H:%M")"
