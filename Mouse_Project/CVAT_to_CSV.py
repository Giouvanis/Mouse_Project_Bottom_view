"""
CVAT_to_CSV.py

This script converts CVAT XML annotations into a CSV file containing
keypoint coordinates and bounding box information for each labeled image.

Usage:
    python CVAT_to_CSV.py <cvat_annotation.xml>

Arguments:
    <cvat_annotation.xml>: Path to the CVAT XML annotation file.

Output:
    - A CSV file with the same basename as the input XML file.
    - If the output CSV already exists, an incremental number is added to the filename.

CSV Format:
    filename, nose-x, nose-y, ears_midpoint-x, ears_midpoint-y, ..., bbox_tl-x, bbox_tl-y, bbox_br-x, bbox_br-y

Notes:
    - Only images with labels (keypoints or bounding boxes) are considered.
    - Keypoint labels are dynamically extracted from the XML file.
    - Keypoints with the attribute outside="1" will have empty coordinates in the CSV.
"""

import xml.etree.ElementTree as ET
import csv
import os
import sys

def parse_cvat_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    keypoints = set()
    
    for image in root.findall(".//image"):
        filename = image.get("name")
        annotation = {"filename": filename}
        has_labels = False
        
        for skeleton in image.findall(".//skeleton"):
            for point in skeleton.findall(".//points"):
                label = point.get("label")
                if point.get("outside") == "1":
                    annotation[f"{label}-x"] = ""
                    annotation[f"{label}-y"] = ""
                else:
                    coords = point.get("points").split(",")
                    if len(coords) == 2:
                        annotation[f"{label}-x"] = float(coords[0])
                        annotation[f"{label}-y"] = float(coords[1])
                        keypoints.add(label)
                        has_labels = True
        
        for box in image.findall(".//box"):
            annotation["bbox_tl-x"] = float(box.get("xtl"))
            annotation["bbox_tl-y"] = float(box.get("ytl"))
            annotation["bbox_br-x"] = float(box.get("xbr"))
            annotation["bbox_br-y"] = float(box.get("ybr"))
            has_labels = True
        
        if has_labels:
            annotations.append(annotation)
    
    return annotations, sorted(keypoints)

def save_to_csv(xml_file, annotations, keypoints):
    csv_basename = os.path.splitext(xml_file)[0] + ".csv"
    counter = 1
    output_csv = csv_basename
    while os.path.exists(output_csv):
        output_csv = f"{os.path.splitext(xml_file)[0]}_{counter}.csv"
        counter += 1
    
    headers = ["filename"] + [f"{kp}-x" for kp in keypoints] + [f"{kp}-y" for kp in keypoints] + ["bbox_tl-x", "bbox_tl-y", "bbox_br-x", "bbox_br-y"]
    
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for ann in annotations:
            writer.writerow(ann)
    
    print(f"CSV file saved as: {output_csv}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python CVAT_to_CSV.py <cvat_annotation.xml>")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    annotations, keypoints = parse_cvat_xml(xml_file)
    save_to_csv(xml_file, annotations, keypoints)

if __name__ == "__main__":
    main()