## Download
Download the dataset and create the train/test split using this command:
`cd datasets && python3 setup_12scenes.py`

## Knowledge Distillation Training
Start knowledge distillation training using the following command:
`python3 knowledge-distillation/training.py --weights_path naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --dataset_path datasets --scene_type 12scenes_apt1_kitchen >> logfile.log`