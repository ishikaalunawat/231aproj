#!/bin/bash

# Array of folder names
folders=("scene_7" "scene_8" "scene_9")

# Loop through the folders
for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    # Add your commands here to process each folder
    # For example:
    CUDA_VISIBLE_DEVICES=2 python knowledge-distillation/training.py --weights_path naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --dataset_path datasets --scene_type $folder --model_type conv_pretrained
done