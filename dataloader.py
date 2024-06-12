import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from dust3r.utils.image import load_images, rgb

class RGBDDataset(Dataset):
    def __init__(self, scene_dir, transform=False):
        self.scene_dir = scene_dir
        self.frames = self._load_frames()

    def _load_frames(self):
        frames = []
        if os.path.isdir(self.scene_dir):
            for f in os.listdir(os.path.join(self.scene_dir, "rgb")):
                if f.endswith(".color.jpg"):
                    frame_id = f.split(".")[0]
                    frames.append({
                        'color': os.path.join(self.scene_dir, "rgb", f"{frame_id}.color.jpg"),
                        'pts3d': os.path.join(self.scene_dir, "pts3d", f"{frame_id}.pt"),
                    })

        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_info = self.frames[idx]

        # load RGB image
        rgb_image = load_images([frame_info['color']], size=512)[0]['img'].squeeze().to(device="cuda")

        # load scene coordinate image
        scene_coordinate_image = torch.load(frame_info['pts3d']).to(device="cuda")

        return rgb_image, scene_coordinate_image

def get_dataloader(root_dir, batch_size=32, transform=False, shuffle=True):
    dataset = RGBDDataset(root_dir, transform=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader