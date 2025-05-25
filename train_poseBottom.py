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
import resnet
import SHG 
import os
import sys

# Add parent of current script (Mouse_Project) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
print(f"🔍 Adding to sys.path: {parent_dir}")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Check if tools/plot_losses.py actually exists
tools_path = os.path.join(parent_dir, 'tools', 'plot_losses.py')
print(f"🛠️ plot_losses.py exists: {os.path.exists(tools_path)}")
print(f"📦 sys.path: {sys.path}")

# Try importing
from tools.plot_losses import plot_losses
print("✅ Successfully imported plot_losses!")
from alive_progress import alive_bar
from alive_progress.animations import bar_factory
from tools.plot_losses import plot_losses
from test_pose import test_pose
import yaml
from datetime import datetime

def train_pose(model, image_train_folder, image_val_folder, 
               annotation_path, input_size, output_size, device='cpu',
               n_joints=None, train_batch_size=25, patience=10, 
               start_lr=0.001, min_lr=0.00001, epochs=10000,
               threshold = 1e-3, augmentation=False, output_folder=None):

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    if model.name:
        model_name = model.name
        print(f'Model name: {model_name} - {input_size}')
    else:
        model_name = model.__class__.__name__
        print(f'Model name: {model_name} - {input_size}')

    if not output_folder:
        output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    lowest_val_loss = float('inf')

    train_dataset = PoseDataset(image_folder=image_train_folder, 
                                label_file=annotation_path, 
                                resize_to=input_size,
                                heatmap_size=output_size,
                                augmentation=augmentation)
    val_dataset   = PoseDataset(image_folder=image_val_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size,
                                augmentation=False)

    val_batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=val_batch_size,   num_workers=0, shuffle=False)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    lr_factor = 0.1
    last_lr = min_lr * lr_factor
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        threshold_mode = 'abs', #does not consider a diff of 0 as improvement
        threshold = threshold,
        mode='min', 
        factor=lr_factor, 
        patience=patience, 
        min_lr=last_lr
    )

    prev_lr = -1
    total_time = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        num_batches = 0

        bar = bar_factory('.', tip='🚀', background=' ', borders=('🌒','🌌'))
        with alive_bar(len(train_dataloader.dataset), title=f"Epoch [{epoch}/{epochs}]", bar=bar, spinner='dots') as bar:
            for batch_idx, (images, gt_kps, gt_hms, _) in enumerate(train_dataloader):
                images, gt_hms = images.to(device), gt_hms.to(device)
                num_batches += 1
                optimizer.zero_grad()
                prediction = model(images)
                loss = criterion(prediction, gt_hms)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                # print(f'[{(batch_idx + 1) * train_batch_size} / {len(train_dataset)}]: loss: {loss.item()}')
                bar.text = f"Train Loss: {loss.item():.10f}"
                if (batch_idx + 1) * train_batch_size > len(train_dataset):
                    bar(images.shape[0])
                else:
                    bar(train_batch_size)
            train_loss /= num_batches
        
        model.eval()
        val_loss = 0.0
        num_batches = 0
        for batch_idx, (images, gt_kps, gt_hms, _) in enumerate(val_dataloader):
            images, gt_hms = images.to(device), gt_hms.to(device)
            num_batches += 1
            prediction = model(images)
            loss = criterion(prediction, gt_hms)
            val_loss += loss.item()
            # print(f'{batch_idx}: loss: {loss.item()}')
        val_loss /= num_batches
        overall_time = time.time() - start_time
        total_time += overall_time

        # erase progress bar
        sys.stdout.write('\033[A\033[K')

        # Step scheduler
        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        if prev_lr == -1:
            prev_lr = current_lr
        elif prev_lr != current_lr:
            prev_lr = current_lr
            print(f"Learning rate changed to {prev_lr}")

        print(f"Epoch [{epoch}/{epochs}] | Avg. Train Loss: {train_loss:.5e} | Avg. val Loss: {val_loss:.5e}" f" | Overall time: {overall_time:.2f}")
        
        loss_csv_path = os.path.join(output_folder, f'loss_{model_name}_{"x".join(str(n) for n in input_size)}.csv')
        with open(loss_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:  # Check if the file is empty
                writer.writerow(['epoch', 'average_train_loss', 'average_val_loss', 'overall_time'])
            writer.writerow([epoch, train_loss, val_loss, overall_time])

        if val_loss < lowest_val_loss - threshold:
            lowest_val_loss = val_loss
            print(f'Saving as best model...')
            torch.save(model.state_dict(), os.path.join(output_folder, f'snapshot_{model_name}_{"x".join(str(n) for n in input_size)}.pth'))

        if current_lr <= last_lr:
            print(f"Stopping training: learning rate has reached the minimum threshold ({min_lr})")
            break

    plot_losses(loss_csv_path)

    return epoch, lowest_val_loss, total_time
    ...

if __name__ == '__main__':
    # ✅ BOTTOM-VIEW paths
    image_train_folder = r'C:/Users/giouv\bnl-ai/Mouse_Project/new_scripts/trainN'
    image_val_folder   = r'C:/Users/giouv\bnl-ai/Mouse_Project/new_scripts/validation'
    image_test_folder  = r'C:/Users/giouv\bnl-ai/Mouse_Project/new_scripts/testN'
    annotation_path    = r'C:/Users/giouv/bnl-ai/Mouse_Project/New_Export/MouseProjectN/annotations.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    n_joints  = 12
    patience  = 10
    epochs    = 1000
    start_lr  = 1e-3
    min_lr    = 1e-5
    threshold = 1e-5
    train_batch_size = 50
    augmentation = False

    output_folder = f'out/train_pose_bottom-{datetime.now().strftime("%y%m%d_%H%M%S")}'
    os.makedirs(output_folder, exist_ok=True)

    info_file = os.path.join(output_folder, 'info.txt')
    with open(info_file, 'w') as file:
        file.write(f"Date and time: {datetime.now()}\n")
        file.write(f"image_train_folder: {image_train_folder}\n")
        file.write(f"image_val_folder: {image_val_folder}\n")
        file.write(f"image_test_folder: {image_test_folder}\n")
        file.write(f"annotation_path: {annotation_path}\n")
        file.write(f"epochs: {epochs}\n")
        file.write(f"Threshold lr for improvement: {threshold}\n")
        file.write(f"starting_learning_rate: {start_lr}\n")
        file.write(f"minimum_learning_rate: {min_lr}\n")
        file.write(f"patience: {patience}\n")
        file.write(f"train_batch_size: {train_batch_size}\n")
        file.write(f"Number of joints: {n_joints}\n")
        file.write(f"Augmentation: {augmentation}\n")

    # 👇 HRNet W48 Config (256x192)
    with open(r'C:/Users/giouv/bnl-ai/config/hrnet_w32_384_288.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        cfg['MODEL']['NUM_JOINTS'] = n_joints
        input_size = cfg['MODEL']['IMAGE_SIZE']
        output_size = cfg['MODEL']['HEATMAP_SIZE']
        model = hrnet.get_pose_net(cfg, is_train=True)

        epoch, val_loss, total_time = train_pose(model, image_train_folder, image_val_folder, annotation_path, 
                                                 input_size=input_size, output_size=output_size,
                                                 n_joints=n_joints, device=device,
                                                 train_batch_size=train_batch_size, patience=patience,
                                                 epochs=epochs, start_lr=start_lr, min_lr=min_lr,
                                                 threshold=threshold, output_folder=output_folder,
                                                 augmentation=augmentation)

        model.load_state_dict(torch.load(os.path.join(output_folder, f'snapshot_{model.name}_{"x".join(str(n) for n in input_size)}.pth'), 
                                         weights_only=True, map_location=device))

        RMSE = test_pose(model, image_test_folder, annotation_path, input_size=input_size,
                         output_size=output_size, device=device, output_folder=output_folder)

        with open(os.path.join(output_folder, 'results.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(['model', 'input_size', 'epochs', 'best_val_loss (MSE)', 'test_RMSE', 'training_time'])
            writer.writerow([model.name, "x".join(str(n) for n in input_size), epoch, val_loss, RMSE, total_time])

    # ✅ You can now repeat this block with ResNet or SHG configs if desired
