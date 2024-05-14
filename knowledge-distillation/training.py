import sys
sys.path.append("../dust3r/")

from student_model import StudentModel
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from enum import Enum
import copy
import argparse
import os

class InferenceParams():
    IMAGE_SIZE = 512
    SCENEGRAPH_TYPE = "complete"
    DEVICE = "mps"
    BATCH_SIZE = 8
    GLOBAL_ALIGNMENT_NITER = 300
    SCHEDULE = "linear"

# Initialize teacher and student models
# teacher = TeacherModel()
student = StudentModel()

# Define your training loop here
def train():
    # TODO: Implement your training loop
    pass


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--dataset_path", type=str, default='cuda', help="pytorch device")
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Load images
    filelist = [os.path.join(args.dataset_path, f) for f in os.listdir(args.dataset_path)]
    print(InferenceParams.IMAGE_SIZE)
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
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init='mst', 
            niter=InferenceParams.GLOBAL_ALIGNMENT_NITER, 
            schedule=InferenceParams.SCHEDULE, 
            lr=lr,
        )
    breakpoint()

    # Start training
    train()

if __name__ == "__main__":
    main()