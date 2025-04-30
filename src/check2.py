
import torch
import open3d as o3d
import numpy as np
from src.dataset.datasets import load_one_scene
from src.eval.eval_nr3d import parse

import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# === Setup ===
scan_id = "scene0025_00"
sentence = "i need a monitor"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Ensure NLTK downloads work in venv ===
nltk.data.path.append('./nltk_data')
for resource in ['punkt', 'averaged_perceptron_tagger', 'wordnet']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir='./nltk_data')

# === Load scene data ===
_, scene = load_one_scene(scan_id, label_type="pred")
room_points = scene["room_points"]
obj_locs = scene["pred_locs"]

# === Load raw .pth file for color info ===
pth_path = f"data/referit3d/scan_data/pcd_with_global_alignment/{scan_id}.pth"
print(f"[INFO] Loading color data from {pth_path}...")
points_raw, colors_raw, _, _ = torch.load(pth_path, map_location="cpu", weights_only=False)

assert np.allclose(points_raw, room_points), "Point mismatch! Verify coordinate alignment."

if colors_raw.max() > 1.0:
    colors_raw = colors_raw / 255.0

# === Load feature map for this scan ===
features_path = "output/nr3d_features_per_scene_pred_label.pth"
all_features = torch.load(features_path, map_location="cpu")
features_this_scene = all_features[scan_id]

# Move feature tensors to DEVICE
features_this_scene = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in features_this_scene.items()}

valid_ids = torch.arange(len(obj_locs), device=DEVICE)

# === Dynamically extract symbolic object category from the sentence ===
words = word_tokenize(sentence.lower())
tags = pos_tag(words)
nouns = [word for word, pos in tags if pos.startswith('NN')]
category_guess = nouns[-1] if nouns else "object"
json_obj = {"category": category_guess}

# === Run EaSe parse to get prediction scores ===
final_concept = parse(scan_id, json_obj, features_this_scene, valid_ids)
predicted_obj_id = torch.argmax(final_concept).item()

# === Build point cloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(room_points)
pcd.colors = o3d.utility.Vector3dVector(colors_raw)

# === Draw bounding boxes ===
bbox_list = []
for i, bbox in enumerate(obj_locs):
    center = bbox[:3]
    size = bbox[3:]
    obb = o3d.geometry.OrientedBoundingBox(center, np.eye(3), size)
    obb.color = (0, 1, 0) if i == predicted_obj_id else (1, 0, 0)
    bbox_list.append(obb)

print(f"[INFO] Predicted object ID: {predicted_obj_id} for sentence: '{sentence}'")
print(f"[INFO] Visualizing {scan_id} with predicted object and bounding boxes...")
o3d.visualization.draw_geometries([pcd, *bbox_list])