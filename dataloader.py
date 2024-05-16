import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class RGBDDataset(Dataset):
    def __init__(self, root_dir, transform=False):
        self.root_dir = root_dir
        if transform:

            self.transform_rgb = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                ])

            self.transform_depth = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
        self.frames = self._load_frames()

    def _load_frames(self):
        frames = []
        for scene in os.listdir(self.root_dir):
            scene_dir = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_dir):
                for file in os.listdir(scene_dir):
                    if file.endswith(".color.jpg"):
                        frame_id = file.split(".")[0]
                        frames.append({
                            'color': os.path.join(scene_dir, f"{frame_id}.color.jpg"),
                            'depth': os.path.join(scene_dir, f"{frame_id}.depth.png"),
                            'pose': os.path.join(scene_dir, f"{frame_id}.pose.txt")
                        })
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_info = self.frames[idx]

        # load RGB image
        rgb_image = Image.open(frame_info['color'])
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)

        # load depth image
        depth_image = Image.open(frame_info['depth'])
        depth_image = np.array(depth_image).astype(np.float32)
        depth_image[depth_image == 0] = np.nan 
        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)
        depth_image = torch.tensor(depth_image).unsqueeze(0) 

        # load camera pose
        pose = np.loadtxt(frame_info['pose'])
        pose = torch.tensor(pose, dtype=torch.float32)

        return rgb_image, depth_image, pose

def get_dataloader(root_dir, batch_size=32, transform=False, shuffle=True):
    dataset = RGBDDataset(root_dir, transform=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader