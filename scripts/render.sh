#!/bin/bash

DIR_PATH="$1"
time_index="${2:-}"
train_mode="splannequin"

if [ -z "$DIR_PATH" ]; then
    echo "Error: Directory path required. (Usage: bash $0 <directory_path> [time_index])"
    exit 1
fi

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
echo "▶ Time:          $(date "+%Y-%m-%d %H:%M")"
echo "▶ Directory:     $DIR_PATH"
echo "▶ Time Index:    ${time_index:-all (rendering all time indices)}"

python ./render.py \
   --model_path "$DIR_PATH" \
   --time_index "$time_index" \
   --render_modes "normal" "fixed-time-view" \
   --configs ./arguments/${train_mode}.py

echo "▶ Execution completed at $(date "+%Y-%m-%d %H:%M")"
