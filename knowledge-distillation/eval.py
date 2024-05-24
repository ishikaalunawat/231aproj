import sys
sys.path.append("../dust3r/")

from student_model import StudentModel
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dataloader import get_dataloader

from training import teacher_inference
import torch.nn.functional as F

from enum import Enum
import copy
import argparse
import os
import torch
import gc

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--dataset_path", type=str, default='cuda', help="pytorch device")
    return parser

class InferenceParams():
    IMAGE_SIZE = 512
    SCENEGRAPH_TYPE = "complete"
    DEVICE = "mps"
    BATCH_SIZE = 8
    GLOBAL_ALIGNMENT_NITER = 300
    SCHEDULE = "linear"

def eval():
    parser = get_args_parser()
    args = parser.parse_args()

    ## Teacher inference 
    pts3D = teacher_inference(args)
    
    # Load images
    filelist = [os.path.join(args.dataset_path, f) for f in os.listdir(args.dataset_path)]
    print(InferenceParams.IMAGE_SIZE)
    imgs = load_images(filelist, size=InferenceParams.IMAGE_SIZE)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    ## Student inference
    student = StudentModel().to(InferenceParams.DEVICE)
    student.load_state_dict(torch.load('student_model.pth'))
    pred = student(torch.cat([im['img'] for im in imgs[8:]], dim=0).to(InferenceParams.DEVICE))
    b, c, _, _ = pred.shape
    pred = torch.transpose(pred.reshape(b, c, -1), 1, 2)
    l2_error = F.mse_loss(pred, pts3D[8:])
    print(f"Average L2 error (mean squared error): {l2_error}")

if __name__ == "__main__":
    eval()