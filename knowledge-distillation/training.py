from student_model import StudentModel
from dust3r.dust3r.model import AsymmetricCroCo3DStereo
from dust3r.dust3r.inference import inference

import os

# Initialize teacher and student models
teacher = TeacherModel()
student = StudentModel()

# Define your training loop here
def train():
    # TODO: Implement your training loop
    pass

# Load images
filelist = os.listdir(img_folder)
image_size = (256, 256)
imgs = load_images(filelist, size=image_size)

# Start training
train()