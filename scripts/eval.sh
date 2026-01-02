#!/bin/bash

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Error: Directory path required. (Usage: bash $0 <directory_path>)"
    exit 1
fi

DIR_PATH="$1"

setup_gpu() {
   echo "▶ Server: $(hostname)"
   read GPU_ID gpu_usage gpu_memory gpu_total _ < <(nvidia-smi \
       --query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free \
       --format=csv,noheader,nounits | sort -t',' -k5 -nr | head -1 | tr ',' ' ')
   echo "▶ GPU $GPU_ID : Utilization ${gpu_usage}%, Memory. $((gpu_memory * 100 / gpu_total))% (${gpu_memory}MB/${gpu_total}MB)"
   export CUDA_VISIBLE_DEVICES=$GPU_ID
   export PYTORCH_ALLOC_CONF="max_split_size_mb:512"
}

model_types=("aesthetic" "musiq" "topiq_nr" "hyperiqa" "clipiqa")
setup_gpu
for m_type in "${model_types[@]}"; do
    python ./eval.py \
        "$DIR_PATH" \
        --out_dir ./results \
        --render_mode "fixed-time-view" \
        --model_type $m_type \
        --save_summary \
        --save_details \
        --save_plots
done
echo "All directories processed."
