import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from PoseDataset import PoseDataset
import hrnet
import resnet
import yaml
import os
from datetime import datetime

def train_pose(model, image_train_folder, image_test_folder, 
               annotation_path, input_size, output_size, 
               n_joints=None, train_rotate=False):

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    learning_rate = 0.001
    epochs = 1000
    patience = 25
    lowest_test_loss = float('inf')

    train_dataset = PoseDataset(image_folder=image_train_folder, 
                                label_file=annotation_path, 
                                resize_to=input_size,
                                heatmap_size=output_size,
                                rotate=train_rotate)
    test_dataset  = PoseDataset(image_folder=image_test_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size,
                                rotate=False)

    train_batch_size = 10
    test_batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=test_batch_size,  num_workers=2, shuffle=False)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # write all the information about this training
    info_file = os.path.join(output_folder, 'info.txt')
    with open(info_file, 'w') as file:
        file.write(f"model name: {model. __class__. __name__}\n")
        file.write(f"number parameters: {sum(p.numel() for p in model.parameters())}\n")
        file.write(f"image_train_folder: {image_train_folder}\n")
        file.write(f"image_test_folder: {image_test_folder}\n")
        file.write(f"annotation_path: {annotation_path}\n")
        file.write(f"input_size: {input_size}\n")
        file.write(f"output_size: {output_size}\n")
        file.write(f"epochs: {epochs}\n")
        file.write(f"optimizer: {optimizer.__class__.__name__}\n")
        file.write(f"learning_rate: {learning_rate}\n")
        file.write(f"train_batch_size: {train_batch_size}\n")
        file.write(f"test_batch_size: {test_batch_size}\n")
        file.write(f"bodypart-keypoint indices: {train_dataset.bp_hm_index}\n")
        file.write(f"Number of joints: {n_joints}\n")

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
        for batch_idx, (images, gt_kps, gt_hms) in enumerate(train_dataloader):
            images, gt_hms = images.to(device), gt_hms.to(device)
            num_batches += 1
            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, gt_hms)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print(f'[{(batch_idx + 1) * train_batch_size} / {len(train_dataset)}]: loss: {loss.item()}')
        train_loss /= num_batches
        
        model.eval()
        test_loss = 0.0
        num_batches = 0
        for batch_idx, (images, gt_kps, gt_hms) in enumerate(test_dataloader):
            images, gt_hms = images.to(device), gt_hms.to(device)
            num_batches += 1
            prediction = model(images)
            loss = criterion(prediction, gt_hms)
            test_loss += loss.item()
            # print(f'{batch_idx}: loss: {loss.item()}')
        test_loss /= num_batches
        overall_time = time.time() - start_time

        print(f"Epoch [{epoch}/{epochs}], Average Train Loss: {train_loss}, Average Test Loss: {test_loss}" \
            f", Time: {overall_time:.2f}")

        if epoch % 100 == 0:
            print(f'Saving snapshot at epoch {epoch}...')
            torch.save(model.state_dict(), os.path.join(output_folder, f'snapshot_{epoch}.pth'))
        
        with open(os.path.join(output_folder, 'loss.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:  # Check if the file is empty
                writer.writerow(['epoch', 'average_train_loss', 'average_test_loss', 'overall_time'])
            writer.writerow([epoch, train_loss, test_loss, overall_time])

        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            count_patience = 0
            print(f'Saving best model...')
            torch.save(model.state_dict(), os.path.join(output_folder, 'snapshot_best.pth'))

def top():
    image_train_folder = r'datasets\topV12\train'
    image_test_folder  = r'datasets\topV12\test'
    annotation_path    = r'datasets\topV12\annotations.csv'

    with open(r'config\hrnet_w32_256_192.yaml', 'r') as f:
        cfg_w32_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 10
        model = hrnet.get_pose_net(cfg_w32_256_192)
        train_pose(model, image_train_folder, 
                   image_test_folder, annotation_path, 
                   input_size=cfg_w32_256_192['MODEL']['IMAGE_SIZE'],
                   output_size=cfg_w32_256_192['MODEL']['HEATMAP_SIZE'],
                   n_joints=cfg_w32_256_192['MODEL']['NUM_JOINTS'],
                   train_rotate=True)

    # with open(r'config\hrnet_w32_384_288.yaml', 'r') as f:
    #     cfg_w32_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_w32_384_288['MODEL']['NUM_JOINTS'] = 14
    #     model = hrnet.get_pose_net(cfg_w32_384_288)
    #     train_pose(model, image_train_folder, 
    #                image_test_folder, annotation_path, 
    #                input_size=cfg_w32_384_288['MODEL']['IMAGE_SIZE'],
    #                output_size=cfg_w32_384_288['MODEL']['HEATMAP_SIZE'],
    #                n_joints=cfg_w32_384_288['MODEL']['NUM_JOINTS'],
    #                train_rotate=True)

    # with open(r'config\hrnet_w48_256_192.yaml', 'r') as f:
    #     cfg_w48_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_w48_256_192['MODEL']['NUM_JOINTS'] = 14
    #     model = hrnet.get_pose_net(cfg_w48_256_192)
    #     train_pose(model, image_train_folder, 
    #                image_test_folder, annotation_path, 
    #                input_size=cfg_w48_256_192['MODEL']['IMAGE_SIZE'],
    #                output_size=cfg_w48_256_192['MODEL']['HEATMAP_SIZE'],
    #                n_joints=cfg_w48_256_192['MODEL']['NUM_JOINTS'],
    #                train_rotate=True)

    # with open(r'config\hrnet_w48_384_288.yaml', 'r') as f:
    #     cfg_w48_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_w48_384_288['MODEL']['NUM_JOINTS'] = 14
    #     model = hrnet.get_pose_net(cfg_w48_384_288)
    #     train_pose(model, image_train_folder, 
    #                image_test_folder, annotation_path, 
    #                input_size=cfg_w48_384_288['MODEL']['IMAGE_SIZE'],
    #                output_size=cfg_w48_384_288['MODEL']['HEATMAP_SIZE'],
    #                n_joints=cfg_w48_384_288['MODEL']['NUM_JOINTS'],
    #                train_rotate=True)
    
    # with open(r'config\res50_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f:
    #     cfg_res50_256x192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_res50_256x192['MODEL']['NUM_JOINTS'] = 14
    #     model = resnet.get_pose_net(cfg_res50_256x192, is_train=True)
    #     train_pose(model, image_train_folder, 
    #                image_test_folder, annotation_path, 
    #                input_size=cfg_res50_256x192['MODEL']['IMAGE_SIZE'],
    #                output_size=cfg_res50_256x192['MODEL']['HEATMAP_SIZE'],
    #                n_joints=cfg_res50_256x192['MODEL']['NUM_JOINTS'],
    #                train_rotate=True)
        
    # with open(r'config\res152_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f:
    #     cfg_res152_256x192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_res152_256x192['MODEL']['NUM_JOINTS'] = 14
    #     model = resnet.get_pose_net(cfg_res152_256x192, is_train=True)
    #     train_pose(model, image_train_folder, 
    #                image_test_folder, annotation_path, 
    #                input_size=cfg_res152_256x192['MODEL']['IMAGE_SIZE'],
    #                output_size=cfg_res152_256x192['MODEL']['HEATMAP_SIZE'],
    #                n_joints=cfg_res152_256x192['MODEL']['NUM_JOINTS'],
    #                train_rotate=True)

def side():
    image_train_folder = r'C:\Users\Lund University\git\bnl-ai\datasets\side_374280\train'
    image_test_folder  = r'C:\Users\Lund University\git\bnl-ai\datasets\side_374280\test'
    annotation_path    = r'C:\Users\Lund University\git\bnl-ai\datasets\side_374280\annotations.csv'

    with open(r'config\hrnet_w48_384_288.yaml', 'r') as f:
        cfg_w48_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_w48_384_288['MODEL']['NUM_JOINTS'] = 26
        model = hrnet.get_pose_net(cfg_w48_384_288)
        train_pose(model, image_train_folder, 
                   image_test_folder, annotation_path, 
                   input_size=cfg_w48_384_288['MODEL']['IMAGE_SIZE'],
                   output_size=cfg_w48_384_288['MODEL']['HEATMAP_SIZE'],
                   n_joints=cfg_w48_384_288['MODEL']['NUM_JOINTS'],
                   train_rotate=False)

if __name__ == '__main__':
    top()
    # side()