import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import Pad
import matplotlib.pyplot as plt
import torchvision


def calculate_padding(original_width, original_height, target_width, target_height):
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        new_height = original_width / target_aspect_ratio
        padding_height = (new_height - original_height) / 2
        padding_width = 0
    else:
        new_width = original_height * target_aspect_ratio
        padding_width = (new_width - original_width) / 2
        padding_height = 0

    return int(padding_width), int(padding_height)


def get_left_right_swap_indices(keypoint_dict):
    swap_indices = []
    names = list(keypoint_dict.keys())

    for i, name in enumerate(names):
        if name.startswith('left_'):
            right_name = name.replace('left_', 'right_')
            if right_name in keypoint_dict:
                j = keypoint_dict[right_name]
                swap_indices.append((i, j))
    return swap_indices


def generate_heatmap(image, keypoint, padding_width, padding_height, heatmap_size=(64, 48)):
    _, img_h, img_w = image.shape
    heatmap_h, heatmap_w = heatmap_size

    if torch.isnan(keypoint).any():
        return torch.zeros(heatmap_h, heatmap_w, dtype=torch.float32)

    x, y = keypoint
    scale_x = heatmap_w / img_w
    scale_y = heatmap_h / img_h
    keypoint_hm = torch.tensor([x * scale_x, y * scale_y])

    def sigma(H, W, base_H=64, base_W=48, base_sigma=1.1):
        base_diag = math.sqrt(base_H ** 2 + base_W ** 2)
        diag = math.sqrt(H ** 2 + W ** 2)
        return base_sigma * (diag / base_diag)

    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    center_x, center_y = int(keypoint_hm[0]), int(keypoint_hm[1])

    for i in range(heatmap_h):
        for j in range(heatmap_w):
            heatmap[i, j] = np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma(heatmap_h, heatmap_w) ** 2))

    if padding_height:
        heatmap[:padding_height, :] = 0
        heatmap[-padding_height:, :] = 0
    if padding_width:
        heatmap[:, :padding_width] = 0
        heatmap[:, -padding_width:] = 0

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return torch.tensor(heatmap, dtype=torch.float32)


class PoseDataset(Dataset):
    def __init__(self, image_folder, resize_to, heatmap_size, label_file, augmentation=False):
        self.image_folder = image_folder
        self.resize_to = resize_to
        self.heatmap_size = heatmap_size
        self.augmentation = augmentation
        self.labels = pd.read_csv(label_file)
        self.image_names = sorted(os.listdir(image_folder))

        self.images = [torchvision.io.read_image(os.path.join(image_folder, name)) for name in self.image_names]

        # BBox
        self.bbox = self.labels[[col for col in self.labels.columns if col.startswith("bbox_")]]
        df = self.labels.drop(columns=[col for col in self.labels.columns if col.startswith("bbox_")])

        keypoints = sorted(set(col.rsplit('-', 1)[0] for col in df.columns if '-' in col))
        ordered_columns = ['filename'] + [col for pair in keypoints for col in (f"{pair}-x", f"{pair}-y") if col in df.columns]
        self.keypoints = df[ordered_columns]

        df_cp = df.drop(columns=['filename'])
        self.keypoint_names = {}
        for idx, col in enumerate(df_cp.columns):
            base_name = col.rstrip('-xy')
            if base_name not in self.keypoint_names:
                self.keypoint_names[base_name] = idx
        self.swap_keypoints = get_left_right_swap_indices(self.keypoint_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = self.images[idx].clone()
        img_name = self.image_names[idx]
        idx_label = self.labels[self.labels['filename'] == img_name].index[0]
        bbox = self.bbox.iloc[idx_label].to_numpy()
        keypoints = self.keypoints[self.keypoints.iloc[:, 0] == img_name].iloc[:, 1:].values.astype('float32')
        keypoints = torch.tensor(keypoints).view(-1, 2)

        valid_keypoints = keypoints[~torch.isnan(keypoints).any(dim=1)]
        if len(valid_keypoints) == 0:
            raise ValueError("All keypoints are NaN.")

        image = F.crop(image, int(bbox[1]), int(bbox[0]), int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0]))
        keypoints[:, 0] -= bbox[0]
        keypoints[:, 1] -= bbox[1]

        if self.augmentation:
            scale = random.uniform(0.9, 1.1)
            keypoints *= scale
            image = image.unsqueeze(0)
            image = torch.nn.functional.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)

        if self.augmentation:
            angle = random.choice([0, 90, 180, 270])
            crop_w, crop_h = image.shape[2], image.shape[1]
            angle_rad = math.radians(angle)
            image = F.rotate(image, angle, expand=True)
            center_x, center_y = image.shape[2] / 2, image.shape[1] / 2
            rotation_matrix = torch.tensor([
                [math.cos(-angle_rad), -math.sin(-angle_rad)],
                [math.sin(-angle_rad), math.cos(-angle_rad)]
            ])
            keypoints += torch.tensor([(image.shape[2] - crop_w) / 2, (image.shape[1] - crop_h) / 2])
            keypoints -= torch.tensor([center_x, center_y])
            keypoints = torch.mm(keypoints, rotation_matrix.T) + torch.tensor([center_x, center_y])

        pad_w, pad_h = calculate_padding(image.shape[2], image.shape[1], *self.resize_to)
        image = Pad((pad_w, pad_h), fill=0)(image)
        keypoints += torch.tensor([pad_w, pad_h])

        scale_x = self.resize_to[1] / image.shape[2]
        scale_y = self.resize_to[0] / image.shape[1]
        image = F.resize(image, self.resize_to)
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        image = image.float() / 255.0

        if self.augmentation:
            _, H, W = image.shape
            if random.choice([True, False]):
                image = torch.flip(image, dims=[2])
                keypoints[:, 0] = W - keypoints[:, 0]
                for i, j in self.swap_keypoints:
                    keypoints[[i, j]] = keypoints[[j, i]]

            image = T.Compose([
                T.ColorJitter(0.5, 0.5, 0.5, 0.5),
                T.RandomApply([T.GaussianBlur(3)], p=0.5),
                T.RandomGrayscale(p=0.2)
            ])(image)

        image = F.normalize(image, mean=[0.5]*3, std=[0.5]*3)

        heatmaps = []
        for keypoint in keypoints:
            heatmaps.append(generate_heatmap(image, keypoint, int(pad_w * scale_x), int(pad_h * scale_y), self.heatmap_size))
        heatmaps = torch.stack(heatmaps)

        return image, keypoints, heatmaps, torch.tensor([scale_x, scale_y])


if __name__ == "__main__":
    image_folder = "PATH_TO_IMAGES"
    label_file = "PATH_TO_ANNOTATIONS_CSV"
    rows, cols = 1, 5

    fig, ax = plt.subplots(rows, cols)
    fig.set_size_inches(12, 4)

    dataset = PoseDataset(image_folder=image_folder,
                          label_file=label_file,
                          resize_to=(192, 256),
                          heatmap_size=(48, 64),
                          augmentation=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (img, kps, hms, _) in enumerate(data_loader):
        if idx >= cols: break
        img = img[0]
        kps = kps[0]
        img = (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

        ax[idx].imshow(img)
        ax[idx].scatter(kps[:, 0], kps[:, 1], c='red', s=20)
        ax[idx].axis('off')

    plt.tight_layout()
    plt.show()
