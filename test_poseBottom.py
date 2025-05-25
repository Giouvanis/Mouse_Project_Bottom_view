# test_pose_bottom.py

import os
import csv
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import platform
import multiprocessing

from datetime import datetime
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader
from PoseDataset import PoseDataset
import hrnet

bottom_labels = {
    0: 'back_left_paw',
    1: 'back_left_wrist',
    2: 'back_right_paw',
    3: 'back_right_wrist',
    4: 'front_left_paw',
    5: 'front_right_paw',
    6: 'nose',
    7: 'tail_base',
    8: 'tail_end',
    9: 'tail_lower_midpoint',
    10: 'tail_midpoint',
    11: 'tail_upper_midpoint'
}

def extract_keypoints_with_confidence(heatmaps):
    keypoints_with_confidence = []
    for heatmap in heatmaps:
        heatmap = heatmap.squeeze(0)
        max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
        y, x = divmod(max_idx.item(), heatmap.size(1))
        confidence = max_val.item()
        keypoints_with_confidence.append(((x, y), confidence))
    return keypoints_with_confidence

def test_pose(model, image_test_folder, annotation_path, input_size, output_size, device, output_folder=None, confidence=0.0):
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    model.eval()
    model_name = model.__class__.__name__
    print(f"{model_name}-{input_size}, confidence = {confidence}")

    if output_folder is None:
        output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
        os.makedirs(output_folder, exist_ok=True)

    image_folder = os.path.join(output_folder, f'test_images_{model_name}_{"x".join(map(str, input_size))}_{confidence}')
    os.makedirs(image_folder, exist_ok=True)

    test_dataset = PoseDataset(
        image_folder=image_test_folder,
        label_file=annotation_path,
        resize_to=input_size,
        heatmap_size=output_size,
        augmentation=False
    )
    dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    tot_kps = 0
    count_valid_keypoints = None
    count_covered_keypoints = None
    list_SE, list_not_nans = [], []

    with torch.no_grad():
        for idx, (images, gt_keypoints, gt_hms, scaler_kps) in enumerate(dataloader):
            images = images.to(device)
            preds = model(images).squeeze(0)
            image = images.squeeze(0)
            gt_keypoints = gt_keypoints.squeeze(0)

            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), image.shape[1:]) for hm in preds])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            keypoints_tensor = torch.tensor([kp for kp, _ in keypoints], dtype=torch.float32)
            confidences = torch.tensor([conf for _, conf in keypoints], dtype=torch.float32)

            tot_kps += gt_keypoints.shape[0]

            if count_valid_keypoints is None:
                count_valid_keypoints = torch.zeros_like(confidences)
            if count_covered_keypoints is None:
                count_covered_keypoints = torch.zeros_like(confidences)

            count_valid_keypoints += ~torch.isnan(gt_keypoints[:, 0])
            count_covered_keypoints += (confidences >= confidence)

            confident_kps = torch.tensor(
                [kp if conf >= confidence else (float('nan'), float('nan')) for kp, conf in keypoints],
                dtype=torch.float32
            )
            list_SE.append(torch.sum((confident_kps/scaler_kps - gt_keypoints/scaler_kps)**2, dim=1))
            list_not_nans.append(~torch.isnan(confident_kps[:, 0]) & ~torch.isnan(gt_keypoints[:, 0]))

            # Visualization
            denorm = (image * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
            denorm = np.transpose((denorm * 255).astype(np.uint8), (1, 2, 0))

            plt.figure(figsize=(7, 7))
            plt.imshow(denorm)
            plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', label='GT')
            plt.scatter(confident_kps[:, 0], confident_kps[:, 1], c='red', label='Pred')
            for i, (x, y) in enumerate(confident_kps):
                if not (np.isnan(x) or np.isnan(y)):
                    plt.text(x + 2, y - 2, bottom_labels[i], fontsize=8, color='red')
            plt.legend()
            plt.axis('off')
            plt.savefig(os.path.join(image_folder, f"{idx}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

    # Compute RMSE and Coverage
    SE = torch.stack(list_SE)
    valid_mask = torch.stack(list_not_nans)
    n_valid = torch.count_nonzero(valid_mask, dim=0)
    RMSE_per_keypoint = torch.sqrt(torch.nansum(SE, dim=0) / n_valid)
    coverage_per_keypoint = count_covered_keypoints / len(dataloader.dataset)

    RMSE = torch.sqrt(torch.nanmean(SE)).item()
    coverage = (torch.sum(count_covered_keypoints) / tot_kps).item()
    return RMSE, RMSE_per_keypoint, coverage, coverage_per_keypoint

if __name__ == '__main__':
    annotation_path = r'C:\Users\giouv\bnl-ai\Mouse_Project\New_Export\MouseProjectN\annotations.csv'
    image_test_folder = r'C:\Users\giouv\bnl-ai\Mouse_Project\New_Export\MouseProjectN\images\default'
    config_path = r'C:\Users\giouv\bnl-ai\Mouse_Project\new_scripts\hrnet_w32_256_192.yaml'
    model_checkpoint = r'C:\Users\giouv\bnl-ai\Mouse_Project\new_scripts\snapshot_PoseHRNet-W32_192x256.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['NUM_JOINTS'] = 12
    model = hrnet.get_pose_net(cfg, is_train=False)
    model = model.to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)

    output_folder = os.path.join("out", f'test_pose_bottom_{datetime.now().strftime("%y%m%d_%H%M%S")}')
    confidence_range = np.arange(0.0, 0.05 + 0.001, 0.05)

    table = pd.DataFrame(columns=["confidence", "RMSE", "coverage", "rmse_per_keypoint", "coverage_per_keypoint"])
    for confidence in confidence_range:
        rmse, rmse_kp, cov, cov_kp = test_pose(
            model=model,
            image_test_folder=image_test_folder,
            annotation_path=annotation_path,
            input_size=cfg['MODEL']['IMAGE_SIZE'],
            output_size=cfg['MODEL']['HEATMAP_SIZE'],
            device=device,
            output_folder=output_folder,
            confidence=confidence
        )
        table = pd.concat([table, pd.DataFrame([{
            "confidence": confidence,
            "RMSE": rmse,
            "coverage": cov,
            "rmse_per_keypoint": rmse_kp.tolist(),
            "coverage_per_keypoint": cov_kp.tolist()
        }])], ignore_index=True)

    table.to_csv(os.path.join(output_folder, 'results.csv'), index=False)
    print("Done.")
