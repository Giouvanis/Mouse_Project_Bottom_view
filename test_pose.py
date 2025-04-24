<<<<<<< HEAD
import csv
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PoseDataset import PoseDataset
import hrnet
import resnet
from torchvision.transforms.functional import resize
import os
from datetime import datetime

def test_pose(model, image_test_folder, annotation_path, input_size, output_size):

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    model.eval()

    output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}'
    image_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        pass
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        pass
    

    test_dataset  = PoseDataset(image_folder=image_test_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size,
                                rotate=False)
    test_batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=2, shuffle=False)

    info_file = os.path.join(output_folder, 'info.txt')
    with open(info_file, 'w') as file:
        file.write(f"model name: {model. __class__. __name__}\n")
        file.write(f"number parameters: {sum(p.numel() for p in model.parameters())}\n")
        file.write(f"image_test_folder: {image_test_folder}\n")
        file.write(f"annotation_path: {annotation_path}\n")
        file.write(f"input_size: {input_size}\n")
        file.write(f"output_size: {output_size}\n")
        file.write(f"test_batch_size: {test_batch_size}\n")
        file.write(f"bodypart-keypoint indices: {test_dataset.bp_hm_index}\n")

    def extract_keypoints_with_confidence(heatmaps):
        """
        Extracts keypoints and their confidence from heatmaps.
        Args:
            heatmaps: Tensor of shape (num_keypoints, h, w) containing heatmaps for each keypoint.
        Returns:
            keypoints_with_confidence: List of ((x, y), confidence) for each keypoint.
        """
        keypoints_with_confidence = []
        for heatmap in heatmaps:
            heatmap = heatmap.squeeze(0)
            # Get the maximum value and its index
            max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
            y, x = divmod(max_idx.item(), heatmap.size(1))  # Convert linear index to x, y
            
            # Confidence as the maximum value
            confidence = max_val.item()
            keypoints_with_confidence.append(((x, y), confidence))
        return keypoints_with_confidence

    confidence = 0
    with torch.no_grad():
        test_loss = 0.0
        num_batches = 0
        pck = []
        for batch_idx, (images, gt_keypoints, gt_hms) in enumerate(dataloader):
            num_batches += 1
            images = images.to(device)
            start = time.time()
            predictions = model(images)
            end = time.time()
            print(f"Batch {batch_idx} took {end - start:.4f} seconds")
            image = images.squeeze(0)
            gt_keypoints = gt_keypoints.squeeze(0)
            gt_hms = gt_hms.squeeze(0)
            predictions = predictions.squeeze(0)
            original_size = image.shape[1:]  # Assume CHW format
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size, ) for hm in predictions])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            kps = torch.tensor([t[0] for t in keypoints if t[1] > confidence], dtype=torch.float32)
            # image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC

            # Denormalize the image
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
            denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
            denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range

            # Convert from (C, H, W) to (H, W, C) for visualization
            denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

            # Plot the image
            plt.figure(figsize=(6, 6))
            plt.imshow(denormalized_image)
            # plt.imshow(image.permute(1, 2, 0))
            plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            plt.scatter(kps[:, 0], kps[:, 1], c='red', marker='x', label="Predicted")  # Predictions in red

            plt.legend()
            plt.axis("off")

            # Ensure equal aspect ratio so the image is not stretched
            plt.gca().set_aspect('equal', adjustable='box')

            # Save the image
            figure_path = os.path.join(image_folder, str(batch_idx))
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.show()

            print(f"Image saved at {figure_path}")

            # distance between nose and ears_midpoints (which are at the index 3 and 5 respectively)
            temp = []
            for factor in [0.20, 0.5, 0.75]:
                kps = torch.tensor([t[0] for t in keypoints], dtype=torch.float32)
                dist_gt = torch.dist(gt_keypoints[0], gt_keypoints[7]) * factor
                dist_pred = torch.sqrt(torch.sum((kps - gt_keypoints) ** 2, dim=1))
                count = int(((dist_pred < dist_gt) & (dist_pred > 0)).sum().item())
                temp.append(count)
            pck.append(temp)
        pck = np.array(pck)
        tot_kp = 14 * dataloader.__len__()
        pck = pck.sum(axis=0) / tot_kp
        print(pck)

if __name__ == '__main__':
    image_test_folder  = r'datasets\topV12\test'
    annotation_path    = r'datasets\topV12\annotations.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    with open(r'config\hrnet_w32_256_192.yaml', 'r') as f:
            cfg_w32_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 14
            model = hrnet.get_pose_net(cfg_w32_256_192, is_train=False)
            model = model.to(device)
            model.load_state_dict(torch.load(r'out\train-250311_140202\snapshot_best.pth', weights_only=True, map_location=device))
            test_pose(model, image_test_folder, annotation_path, 
                      input_size=cfg_w32_256_192['MODEL']['IMAGE_SIZE'],
                      output_size=cfg_w32_256_192['MODEL']['HEATMAP_SIZE'])

    # with open(r'config\hrnet_w48_384_288.yaml', 'r') as f:
    #         cfg_w48_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
    #         cfg_w48_384_288['MODEL']['NUM_JOINTS'] = 14
    #         model = hrnet.get_pose_net(cfg_w48_384_288)
    #         model.load_state_dict(torch.load(r'out\train-250307_154813\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))
    #         test_pose(model, image_test_folder, annotation_path, 
    #                   input_size=cfg_w48_384_288['MODEL']['IMAGE_SIZE'],
    #                   output_size=cfg_w48_384_288['MODEL']['HEATMAP_SIZE'])
    
    # with open(r'config\res50_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f:
    #     cfg_res50_256x192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_res50_256x192['MODEL']['NUM_JOINTS'] = 14
    #     model = resnet.get_pose_net(cfg_res50_256x192, is_train=False)
    #     model.load_state_dict(torch.load(r'out\train-250311_125712\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))
    #     test_pose(model, image_test_folder, annotation_path, 
    #                 input_size=cfg_res50_256x192['MODEL']['IMAGE_SIZE'],
    #                 output_size=cfg_res50_256x192['MODEL']['HEATMAP_SIZE'])
    
    # with open(r'config\res152_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f:
    #     cfg_res152_256x192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_res152_256x192['MODEL']['NUM_JOINTS'] = 14
    #     model = resnet.get_pose_net(cfg_res152_256x192, is_train=False)
    #     model.load_state_dict(torch.load(r'out\train-250311_134138\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))
    #     test_pose(model, image_test_folder, annotation_path, 
    #                 input_size=cfg_res152_256x192['MODEL']['IMAGE_SIZE'],
=======
import csv
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PoseDataset import PoseDataset
import hrnet
import resnet
from torchvision.transforms.functional import resize
import os
from datetime import datetime

def test_pose(model, image_test_folder, annotation_path, input_size, output_size):

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    model.eval()

    output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}'
    image_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        pass
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        pass
    

    test_dataset  = PoseDataset(image_folder=image_test_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size,
                                rotate=False)
    test_batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=2, shuffle=False)

    info_file = os.path.join(output_folder, 'info.txt')
    with open(info_file, 'w') as file:
        file.write(f"model name: {model. __class__. __name__}\n")
        file.write(f"number parameters: {sum(p.numel() for p in model.parameters())}\n")
        file.write(f"image_test_folder: {image_test_folder}\n")
        file.write(f"annotation_path: {annotation_path}\n")
        file.write(f"input_size: {input_size}\n")
        file.write(f"output_size: {output_size}\n")
        file.write(f"test_batch_size: {test_batch_size}\n")
        file.write(f"bodypart-keypoint indices: {test_dataset.bp_hm_index}\n")

    def extract_keypoints_with_confidence(heatmaps):
        """
        Extracts keypoints and their confidence from heatmaps.
        Args:
            heatmaps: Tensor of shape (num_keypoints, h, w) containing heatmaps for each keypoint.
        Returns:
            keypoints_with_confidence: List of ((x, y), confidence) for each keypoint.
        """
        keypoints_with_confidence = []
        for heatmap in heatmaps:
            heatmap = heatmap.squeeze(0)
            # Get the maximum value and its index
            max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
            y, x = divmod(max_idx.item(), heatmap.size(1))  # Convert linear index to x, y
            
            # Confidence as the maximum value
            confidence = max_val.item()
            keypoints_with_confidence.append(((x, y), confidence))
        return keypoints_with_confidence

    confidence = 0
    with torch.no_grad():
        test_loss = 0.0
        num_batches = 0
        pck = []
        for batch_idx, (images, gt_keypoints, gt_hms) in enumerate(dataloader):
            num_batches += 1
            images = images.to(device)
            start = time.time()
            predictions = model(images)
            end = time.time()
            print(f"Batch {batch_idx} took {end - start:.4f} seconds")
            image = images.squeeze(0)
            gt_keypoints = gt_keypoints.squeeze(0)
            gt_hms = gt_hms.squeeze(0)
            predictions = predictions.squeeze(0)
            original_size = image.shape[1:]  # Assume CHW format
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size, ) for hm in predictions])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            kps = torch.tensor([t[0] for t in keypoints if t[1] > confidence], dtype=torch.float32)
            # image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC

            # Denormalize the image
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
            denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
            denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range

            # Convert from (C, H, W) to (H, W, C) for visualization
            denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

            # Plot the image
            plt.figure(figsize=(6, 6))
            plt.imshow(denormalized_image)
            # plt.imshow(image.permute(1, 2, 0))
            plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            plt.scatter(kps[:, 0], kps[:, 1], c='red', marker='x', label="Predicted")  # Predictions in red

            plt.legend()
            plt.axis("off")

            # Ensure equal aspect ratio so the image is not stretched
            plt.gca().set_aspect('equal', adjustable='box')

            # Save the image
            figure_path = os.path.join(image_folder, str(batch_idx))
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.show()

            print(f"Image saved at {figure_path}")

            # distance between nose and ears_midpoints (which are at the index 3 and 5 respectively)
            temp = []
            for factor in [0.20, 0.5, 0.75]:
                kps = torch.tensor([t[0] for t in keypoints], dtype=torch.float32)
                dist_gt = torch.dist(gt_keypoints[0], gt_keypoints[7]) * factor
                dist_pred = torch.sqrt(torch.sum((kps - gt_keypoints) ** 2, dim=1))
                count = int(((dist_pred < dist_gt) & (dist_pred > 0)).sum().item())
                temp.append(count)
            pck.append(temp)
        pck = np.array(pck)
        tot_kp = 14 * dataloader.__len__()
        pck = pck.sum(axis=0) / tot_kp
        print(pck)

if __name__ == '__main__':
    image_test_folder  = r'datasets\topV12\test'
    annotation_path    = r'datasets\topV12\annotations.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    with open(r'config\hrnet_w32_256_192.yaml', 'r') as f:
            cfg_w32_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 14
            model = hrnet.get_pose_net(cfg_w32_256_192, is_train=False)
            model = model.to(device)
            model.load_state_dict(torch.load(r'out\train-250311_140202\snapshot_best.pth', weights_only=True, map_location=device))
            test_pose(model, image_test_folder, annotation_path, 
                      input_size=cfg_w32_256_192['MODEL']['IMAGE_SIZE'],
                      output_size=cfg_w32_256_192['MODEL']['HEATMAP_SIZE'])

    # with open(r'config\hrnet_w48_384_288.yaml', 'r') as f:
    #         cfg_w48_384_288 = yaml.load(f, Loader=yaml.SafeLoader)
    #         cfg_w48_384_288['MODEL']['NUM_JOINTS'] = 14
    #         model = hrnet.get_pose_net(cfg_w48_384_288)
    #         model.load_state_dict(torch.load(r'out\train-250307_154813\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))
    #         test_pose(model, image_test_folder, annotation_path, 
    #                   input_size=cfg_w48_384_288['MODEL']['IMAGE_SIZE'],
    #                   output_size=cfg_w48_384_288['MODEL']['HEATMAP_SIZE'])
    
    # with open(r'config\res50_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f:
    #     cfg_res50_256x192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_res50_256x192['MODEL']['NUM_JOINTS'] = 14
    #     model = resnet.get_pose_net(cfg_res50_256x192, is_train=False)
    #     model.load_state_dict(torch.load(r'out\train-250311_125712\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))
    #     test_pose(model, image_test_folder, annotation_path, 
    #                 input_size=cfg_res50_256x192['MODEL']['IMAGE_SIZE'],
    #                 output_size=cfg_res50_256x192['MODEL']['HEATMAP_SIZE'])
    
    # with open(r'config\res152_256x192_d256x3_adam_lr1e-3.yaml', 'r') as f:
    #     cfg_res152_256x192 = yaml.load(f, Loader=yaml.SafeLoader)
    #     cfg_res152_256x192['MODEL']['NUM_JOINTS'] = 14
    #     model = resnet.get_pose_net(cfg_res152_256x192, is_train=False)
    #     model.load_state_dict(torch.load(r'out\train-250311_134138\snapshot_best.pth', weights_only=True, map_location=torch.device('cpu')))
    #     test_pose(model, image_test_folder, annotation_path, 
    #                 input_size=cfg_res152_256x192['MODEL']['IMAGE_SIZE'],
>>>>>>> 3db77323602c47cfab201843326ae61a35a24cbe
    #                 output_size=cfg_res152_256x192['MODEL']['HEATMAP_SIZE'])