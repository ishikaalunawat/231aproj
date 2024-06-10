#!/bin/bash

# Array of folder names
folders=("scene_4" "scene_5" "scene_6")

# Loop through the folders
for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    # Add your commands here to process each folder
    # For example:
    CUDA_VISIBLE_DEVICES=1 python knowledge-distillation/training.py --weights_path naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --dataset_path datasets --scene_type $folder --model_type conv_pretrained
done