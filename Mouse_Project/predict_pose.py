from torchvision.transforms import Pad
from torchvision.transforms import functional as F
from torchvision.transforms.functional import resize
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import yaml
import hrnet
import cv2
import os
import glob

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

def calculate_padding(original_width, original_height, target_width, target_height):
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        # Pad height
        new_height = original_width / target_aspect_ratio
        padding_height = (new_height - original_height) / 2
        padding_width = 0
    else:
        # Pad width
        new_width = original_height * target_aspect_ratio
        padding_width = (new_width - original_width) / 2
        padding_height = 0

    return int(padding_width), int(padding_height)

def get_image_files(input_path):
    filenames = []
    
    if os.path.isdir(input_path):
        # Get all image files in the directory (common image formats)
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
        for ext in image_extensions:
            filenames.extend(glob.glob(os.path.join(input_path, ext)))
    elif os.path.isfile(input_path):
        # If it's a single file, check if it's an image
        if input_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
            filenames.append(input_path)
    else:
        # If it's a list of files, check and add valid image files
        potential_files = input_path.split(',')
        for file in potential_files:
            file = file.strip()
            if os.path.isfile(file) and file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
                filenames.append(file)
    
    return filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image files from a folder, list, or single file.")
    parser.add_argument("-i", "--input", required=True, help="Input folder, list of image files (comma-separated), or single image file")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    filenames = get_image_files(args.input)

    model_detection = YOLO(r"runs\detect\first\weights\best.pt")  # pretrained YOLO11n model

    with open(r'config\hrnet_w32_256_192.yaml', 'r') as f:
        cfg_w32_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 14
        model_pose = hrnet.get_pose_net(cfg_w32_256_192, is_train=False)
        model_pose = model_pose.to(device)
        model_pose.load_state_dict(torch.load(r'out\train-250311_140202\snapshot_best.pth', weights_only=True, map_location=device))

        for filename in filenames:
            bbox = model_detection(filename)[0].boxes.xyxy[0].to('cpu')
            print("Bounding box:", bbox)
            image = Image.open(filename).convert("RGB")
            image = image.crop(bbox.numpy())
            padding_width, padding_height = calculate_padding(*image.size, *cfg_w32_256_192['MODEL']['IMAGE_SIZE'])
            image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(image)
            scale_x = cfg_w32_256_192['MODEL']['IMAGE_SIZE'][1] / image.size[0]
            scale_y = cfg_w32_256_192['MODEL']['IMAGE_SIZE'][0] / image.size[1]
            image = F.resize(image, cfg_w32_256_192['MODEL']['IMAGE_SIZE'])
            image = F.to_tensor(image)
            image = F.normalize(image, mean=[0.5] * 3, std=[0.5] * 3)
            image = image.unsqueeze(0).to(device)
            print(image.shape)
            output = model_pose(image)
            print(output.shape)
            predictions = output.squeeze(0)
            image = image.squeeze(0)
            original_size = image.shape[1:]  # Assume CHW format
            confidence = 0.2
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size) for hm in predictions])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            kps = torch.tensor([t[0] for t in keypoints if t[1] > confidence], dtype=torch.float32)
            # image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC

            # Convert keypoints to NumPy and add bounding box offsets
            keypoints_np = (kps.numpy() / np.array([scale_x, scale_y])) + \
                            np.array([bbox[0], bbox[1]]) - \
                            np.array([padding_width, padding_height])

            image = cv2.imread(filename)
            for x, y in keypoints_np.astype(int):
                cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            # Display image
            cv2.imshow("Image with Keypoints", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()