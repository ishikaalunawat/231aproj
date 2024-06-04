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
def train(model, train_dataloader, args):
    
    #Parse the args
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    device = args.device
    dry_run = args.dry_run

    #Define Loss and Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Placeholder for 3D points labels; replace with actual labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')


    ## Saving model
    torch.save(model.state_dict(), "student_models/scene_{}.pth".format(args.scene_type))

def test(model, test_dataloader, args):
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_dataloader)}')



def get_args_parser():
    parser = argparse.ArgumentParser(description='Student Model Training Args')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--patch-size', type=int, default=16, help='patch size for images (default : 16)')
    parser.add_argument('--latent-size', type=int, default=256, help='latent size (default : 256)')
    parser.add_argument('--n-channels', type=int, default=3, help='number of channels in images (default : 3 for RGB)')
    parser.add_argument('--num-heads', type=int, default=12, help='(default : 12)')
    parser.add_argument('--num-encoders', type=int, default=12, help='number of encoders (default : 12)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout value (default : 0.1)')
    parser.add_argument('--img-size', type=int, default=224, help='image size to be reshaped to (default : 224)')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size (default : 4)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs (default : 10)')
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate (default : 0.01)')
    parser.add_argument('--weight-decay', type=float, default=3e-2, help='weight decay value (default : 0.03)')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    
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
    small_dataset_path = "datasets/12scenes_apt1_kitchen/"

    train_dataloader = get_dataloader(small_dataset_path + "train/", num_samples = args.num_train, batch_size=4)
    test_dataloader = get_dataloader(small_dataset_path + "test/", batch_size=1)
    # Start training
    student = StudentModel(args).to(args.device)

    ### Call train function on student model ###
    ### Need Train and Val DataLoaders ###
    train(model = student, train_dataloader=train_dataloader, args = args)

    test(model=student, test_dataloader=test_dataloader, args=args)




if __name__ == "__main__":
    main()