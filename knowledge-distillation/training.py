import sys
sys.path.append("../dust3r/")

from student_model import StudentModel, PatchExtractor, InputEmbedding, EncoderBlock
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from enum import Enum
import copy
import argparse
import os
import gc

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.hub import tqdm

from dataloader import get_dataloader

class InferenceParams():
    IMAGE_SIZE = 512
    SCENEGRAPH_TYPE = "complete"
    DEVICE = "mps"
    BATCH_SIZE = 8
    GLOBAL_ALIGNMENT_NITER = 300
    SCHEDULE = "linear"

# Initialize teacher and student models
# teacher = TeacherModel()
# student = StudentModel()

# Define your training loop here

""" 
Training Function: Trains the student model and saves the model with lowest validation loss

Arguments:
    model - The Vision Transformer Student Model

    train_dataloader - The dataloader of the training dataset

    val_dataloder - The dataloader of the validation dataset

    args - Other required arguments: [epochs, lr, device, weight_decay, dry_run]
"""
def train(model, train_dataloader, val_dataloader, args):
    
    #Parse the args
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    device = args.device
    dry_run = args.dry_run

    #Define Loss and Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Define Train Function for each epoch
    def train_fn(current_epoch):
        model.train()
        total_loss = 0.0
        tk = tqdm(train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch + 1) + "/" + str(epochs))

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if dry_run:
                break

        return total_loss / len(train_dataloader)

    # Define Validation Function for each epoch
    def eval_fn(current_epoch):
        model.eval()
        total_loss = 0.0
        tk = tqdm(val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(epochs))

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if dry_run:
                break

        return total_loss / len(val_dataloader)

    best_valid_loss = np.inf
    best_train_loss = np.inf
    for i in range(epochs):
        train_loss = train_fn(i)
        val_loss = eval_fn(i)

        # Can Modify this
        if val_loss < best_valid_loss:
            torch.save(model.state_dict(), "best-weights.pt")
            print("Saved Best Weights")
            best_valid_loss = val_loss
            best_train_loss = train_loss

    print(f"Training Loss : {best_train_loss}")
    print(f"Valid Loss : {best_valid_loss}")

def get_args_parser():
    parser = argparse.ArgumentParser(description='Student Model Training Args')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='patch size for images (default : 16)')
    parser.add_argument('--latent-size', type=int, default=768,
                        help='latent size (default : 768)')
    parser.add_argument('--n-channels', type=int, default=3,
                        help='number of channels in images (default : 3 for RGB)')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='(default : 12)')
    parser.add_argument('--num-encoders', type=int, default=12,
                        help='number of encoders (default : 12)')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout value (default : 0.1)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='image size to be reshaped to (default : 224')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes in dataset (default : 10 for CIFAR10)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs (default : 10)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='base learning rate (default : 0.01)')
    parser.add_argument('--weight-decay', type=int, default=3e-2,
                        help='weight decay value (default : 0.03)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size (default : 4)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    
    ### Previous args ###
    parser.add_argument("--weights_path", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--dataset_path", type=str, default='cuda', help="pytorch device")

    return parser

def teacher_inference(args):

    # Load images from both train and test
    scene_dir_train = os.path.join(args.dataset_path, args.scene_type, "train", "rgb")
    scene_dir_test = os.path.join(args.dataset_path, args.scene_type, "test", "rgb")
    num_train = args.num_train//2
    filelist = [os.path.join(scene_dir_train, f) for f in os.listdir(scene_dir_train)][:num_train]
    filelist += [os.path.join(scene_dir_test, f) for f in os.listdir(scene_dir_test)][:args.num_train-num_train]
    filelist = sorted(filelist, key=lambda x: os.path.basename(x).split('.')[0].split('-')[1])

    imgs = load_images(filelist, size=InferenceParams.IMAGE_SIZE, num_samples=args.num_train)
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
    
    num_train = args.num_train//2
    rgb_dir_train = os.listdir(os.path.join(args.dataset_path, args.scene_type, "train", "rgb"))[:num_train]
    rgb_dir_test = os.listdir(os.path.join(args.dataset_path, args.scene_type, "test", "rgb"))[:args.num_train - num_train]

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

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    pts3D = teacher_inference(args)
    create_dataset_labels(pts3D, args)

    ## create dataset using 3D points predicted by above teacher model
    train_dataloader = get_dataloader("datasets/12scenes_apt1_kitchen/train/", num_samples = args.num_train, batch_size=4)
    test_dataloader = get_dataloader("datasets/12scenes_apt1_kitchen/test/", batch_size=1)
    # Start training
    student = StudentModel(args).to(args.device)

    ### Call train function on student model ###
    ### Need Train and Val DataLoaders ###
    train(model = student, train_dataloader=train_dataloader, test_dataloader=test_dataloader, args = args)

if __name__ == "__main__":
    main()