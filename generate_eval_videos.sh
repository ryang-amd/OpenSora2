#!/bin/bash

PROMPT_DIR="/home/EvalCrafter/prompts"
VIDEO_DIR="/home/EvalCrafter/opensora_videos"

# Ensure the video directory exists
mkdir -p $VIDEO_DIR

count=0
# Loop through all .txt files in the prompt directory
for file in $PROMPT_DIR/*.txt; do
  ((count++))
    # Read the content of the prompt file
    prompt=$(cat "$file")
    
    # Prepare the output filename by replacing .txt with .mp4
    sub_dir="$VIDEO_DIR/$(basename "${file%.*}")"

    # Command to generate the video
    # python generate.py --task t2v-1.3B --size 832*480 --sample_steps 5 --ckpt_dir ./Wan2.1-T2V-1.3B --prompt "$prompt" --save_file "$output_file"
    torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_256px.py --save-dir $VIDEO_DIR --prompt "$prompt"
    
    # Stop the loop after processing 100 files
    if [ $count -gt 10 ]; then
        break
    fi
done