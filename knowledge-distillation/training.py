import sys
sys.path.append("dust3r/")

from student_model import StudentModel
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dataloader import get_dataloader

import torch.nn.functional as F

from enum import Enum
import copy
import argparse
import os
import torch
import gc

import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(levelname)s - %(message)s')

class InferenceParams():
    IMAGE_SIZE = 512
    SCENEGRAPH_TYPE = "complete"
    DEVICE = "cuda"
    BATCH_SIZE = 8
    GLOBAL_ALIGNMENT_NITER = 300
    SCHEDULE = "linear"

# Initialize teacher and student models
# teacher = TeacherModel()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--dataset_path", type=str, default="datasets")
    parser.add_argument("--scene_type", type=str, default="12scenes_office2_5a_trial", help="Scene type from 12Scenes dataset")
    parser.add_argument("--get_gts", action="store_true", help="Get ground truth 3D points")
    return parser

def teacher_inference(args):

    # Load images from both train and test
    scene_dir_train = os.path.join(args.dataset_path, args.scene_type, "train", "rgb")
    scene_dir_test = os.path.join(args.dataset_path, args.scene_type, "test", "rgb")
    filelist = [os.path.join(scene_dir_train, f) for f in os.listdir(scene_dir_train)]
    filelist += [os.path.join(scene_dir_test, f) for f in os.listdir(scene_dir_test)]
    filelist = sorted(filelist, key=lambda x: os.path.basename(x).split('.')[0].split('-')[1])

    imgs = load_images(filelist, size=InferenceParams.IMAGE_SIZE)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    # Teacher model teaches...
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights_path).to(InferenceParams.DEVICE)
    pairs = make_pairs(imgs, scene_graph=InferenceParams.SCENEGRAPH_TYPE, prefilter=None, symmetrize=True)
    output = inference(pairs, model, InferenceParams.DEVICE, batch_size=InferenceParams.BATCH_SIZE, verbose=True)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=InferenceParams.DEVICE, mode=mode, verbose=True)
    lr = 0.0001

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init='mst', 
            niter=InferenceParams.GLOBAL_ALIGNMENT_NITER, 
            schedule=InferenceParams.SCHEDULE, 
            lr=lr,
        )
        print(loss)
        pts3D = scene.depth_to_pts3d()
        del model  # Remove the teacher model from GPU memory

        # Free up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        pts_dict = {}
        for i in range(len(filelist)):
            frame_id = filelist[i].split(".")[0]
            ind = int(frame_id.split('-')[1])
            pts_dict[ind] = pts3D[i]

        return pts_dict


def create_dataset_labels(pts3D, args):
    """
    Create labels of 3D points for all images. These 3D points are the output of the teacher model
    """
    pts3d_dir_train = os.path.join(args.dataset_path, args.scene_type, "train", "pts3d")
    pts3d_dir_test = os.path.join(args.dataset_path, args.scene_type, "test", "pts3d")
    
    rgb_dir_train = os.listdir(os.path.join(args.dataset_path, args.scene_type, "train", "rgb"))
    rgb_dir_test = os.listdir(os.path.join(args.dataset_path, args.scene_type, "test", "rgb"))

    # num_train, num_test = train_test_split(args)

    if not os.path.exists(pts3d_dir_train):
        os.mkdir(pts3d_dir_train)

    if not os.path.exists(pts3d_dir_test):
        os.mkdir(pts3d_dir_test)

    for f in rgb_dir_train:
        frame_id = f.split(".")[0]
        ind = int(frame_id.split('-')[1])
        torch.save(pts3D[ind], os.path.join(pts3d_dir_train, f"{frame_id}.pt"))

    for f in rgb_dir_test:
        frame_id = f.split(".")[0]
        ind = int(frame_id.split('-')[1])
        torch.save(pts3D[ind], os.path.join(pts3d_dir_test, f"{frame_id}.pt"))

def student_learn(student, dataloader, scene_type, epochs):
    # Use the predicted 3D points to start training
    # breakpoint()
    # student.learn(torch.cat([im['img'] for im in imgs], dim=0).to(InferenceParams.DEVICE), pts3D)
    for e in range(epochs):
        i = 0
        for image, label in dataloader:
            torch.cuda.empty_cache()
            i += 1
            loss = student.learn(image, label)
            log_message = f"Epoch: {e}, Iteration: {i}, Loss: {loss}"
            logging.info(log_message)

    torch.save(student.state_dict(), "student_models/scene_{}.pth".format(scene_type))

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    pts3D = teacher_inference(args)
    create_dataset_labels(pts3D, args)

    if args.get_gts:
        print("Completed generating DUst3r ground truths")
        exit()
    ## create dataset using 3D points predicted by above teacher model
    train_dataloader = get_dataloader(args.dataset_path, args.scene_type, "train", batch_size=4)
    student = StudentModel().to(InferenceParams.DEVICE)

    student_learn(student, train_dataloader, args.scene_type, epochs=10)

    ## Eval
    test_dataloader = get_dataloader(args.dataset_path, "test", batch_size=1)
    for image, label in test_dataloader:
        pred = student(image)
        b, c, _, _ = pred.shape
        pred = torch.transpose(pred.reshape(b, c, -1), 1, 2)        
        l2_error = F.mse_loss(pred, label)
        print(l2_error)