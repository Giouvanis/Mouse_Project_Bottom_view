import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PoseDataset import PoseDataset
import hrnet
import yaml
import os
from datetime import datetime

def train_pose(model, image_train_folder, image_test_folder, 
               annotation_path, input_size, output_size, 
               n_joints=None, train_rotate=False):

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}'
    os.makedirs(output_folder, exist_ok=True)

    learning_rate = 0.001
    epochs = 1000
    patience = 25
    lowest_test_loss = float('inf')

    train_dataset = PoseDataset(image_folder=image_train_folder, 
                                label_file=annotation_path, 
                                resize_to=input_size,
                                heatmap_size=output_size,
                                rotate=train_rotate)
    test_dataset = PoseDataset(image_folder=image_test_folder,
                               label_file=annotation_path,
                               resize_to=input_size,
                               heatmap_size=output_size,
                               rotate=False)

    train_dataloader = DataLoader(train_dataset, batch_size=10, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    count_patience = 0
    for epoch in range(1, epochs + 1):
        if count_patience >= patience:
            print(f'Early stopping at epoch {epoch}...')
            break
        count_patience += 1

        model.train()
        train_loss = 0.0
        start_time = time.time()
        num_batches = 0
        for images, _, gt_hms in train_dataloader:
            images, gt_hms = images.to(device), gt_hms.to(device)
            num_batches += 1
            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, gt_hms)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= num_batches

        model.eval()
        test_loss = 0.0
        num_batches = 0
        for images, _, gt_hms in test_dataloader:
            images, gt_hms = images.to(device), gt_hms.to(device)
            num_batches += 1
            prediction = model(images)
            loss = criterion(prediction, gt_hms)
            test_loss += loss.item()
        test_loss /= num_batches
        overall_time = time.time() - start_time

        print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {overall_time:.2f}")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(output_folder, f'snapshot_{epoch}.pth'))

        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            count_patience = 0
            torch.save(model.state_dict(), os.path.join(output_folder, 'snapshot_best.pth'))

def bottom():
    image_train_folder = r'C:\Users\giouv\bnl-ai\data\train'
    image_test_folder  = r'C:\Users\giouv\bnl-ai\data\test'
    annotation_path    = r'C:\Users\giouv\bnl-ai\annotations.csv'

    yaml_path = r'config\hrnet_bottom.yaml'
    if not os.path.exists(yaml_path):
        print(f"Error: Config file not found at {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        cfg_bottom = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_bottom['MODEL']['NUM_JOINTS'] = 14  
        
        # FIXED: Added missing `is_train=True`
        model = hrnet.get_pose_net(cfg_bottom, is_train=True)

        train_pose(model, image_train_folder, 
                   image_test_folder, annotation_path, 
                   input_size=cfg_bottom['MODEL']['IMAGE_SIZE'],
                   output_size=cfg_bottom['MODEL']['HEATMAP_SIZE'],
                   n_joints=cfg_bottom['MODEL']['NUM_JOINTS'],
                   train_rotate=False)

if __name__ == '__main__':
    bottom()