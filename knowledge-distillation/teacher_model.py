import torch

import torch.nn as nn
from dust3r.dust3r.model import AsymmetricCroCo3DStereo
from dust3r.dust3r.utils.image import load_images, rgb
from dust3r.demo import get_reconstructed_scene

class TeacherModel(AsymmetricCroCo3DStereo):
    def __init__(self):
        super(TeacherModel, self).__init__()

    def teach(self, batch):
        view1, view2 = batch
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        # Run DUST3R inference
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            pred1, pred2 = model(view1, view2)

        # Do global optimization of the output
        