## Environment Setup
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
pip install -r requirements_optional.txt
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime. HIGHLY RECOMMENDED.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

## Dataset Download
Download the dataset and create the train/test split using this command:
`cd datasets && python3 setup_12scenes.py`

## Model download
I downloaded the pretrained model using following command. You can also use models uploaded on huggingface.
```bash
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P naver/
```

## Knowledge Distillation Training
Start knowledge distillation training using the following command:
`python3 knowledge-distillation/training.py --weights_path naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --dataset_path datasets --scene_type 12scenes_apt1_kitchen >> logfile.log`