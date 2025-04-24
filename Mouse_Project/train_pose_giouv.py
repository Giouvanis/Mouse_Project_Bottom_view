import os
import yaml
import torch
from train_pose import train_pose  # assuming train_pose.py is in same dir or accessible path
import hrnet  # should be in PYTHONPATH or same folder as train_pose

def bottom_view():
    # Paths for your specific setup
    image_train_folder = r'C:\Users\giouv\bnl-ai\Mouse_Project\train'
    image_test_folder  = r'C:\Users\giouv\bnl-ai\Mouse_Project\test'
    annotation_path    = r'C:\Users\giouv\bnl-ai\Mouse_Project\annotations.csv'
    config_path        = r'C:\Users\giouv\bnl-ai\config\hrnet_w32_256_192.yaml'

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg['MODEL']['NUM_JOINTS'] = 12  # Adjust if your bottom view has a different number
    model = hrnet.get_pose_net(cfg, is_train=True)

    train_pose(model, 
               image_train_folder=image_train_folder, 
               image_test_folder=image_test_folder, 
               annotation_path=annotation_path, 
               input_size=cfg['MODEL']['IMAGE_SIZE'],
               output_size=cfg['MODEL']['HEATMAP_SIZE'],
               n_joints=cfg['MODEL']['NUM_JOINTS'],
               train_rotate=True)  # Set to True if you want rotation augmentation

if __name__ == "__main__":
    bottom_view()
