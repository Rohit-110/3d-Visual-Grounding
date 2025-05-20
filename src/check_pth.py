import torch
features_path = "data/scanrefer_features_per_scene.pth" # Or your actual path
scan_id = "scene0025_00"
num_objects = 38 # From your script output

all_features = torch.load(features_path, map_location="cpu", weights_only=False) # Set weights_only based on file origin
if scan_id in all_features:
    features_this_scene = all_features[scan_id]
    print(f"Keys found for {scan_id}: {list(features_this_scene.keys())}")
    for k, v in features_this_scene.items():
        if torch.is_tensor(v):
            print(f"  - Key: '{k}', Shape: {v.shape}, Dtype: {v.dtype}")
            # Check if dimensions match num_objects
            is_valid_dim = False; shape = v.shape
            if len(shape) > 0 and shape[0] == num_objects:
                 if len(shape) == 1: is_valid_dim = True
                 elif len(shape) >= 2 and shape[1] == num_objects:
                    if len(shape) == 2: is_valid_dim = True
                    elif len(shape) >= 3 and shape[2] == num_objects: is_valid_dim = True
            if is_valid_dim:
                 print(f"      --> Dimension matches {num_objects} objects!")
            else:
                 print(f"      --> Dimension DOES NOT match {num_objects} objects!")
        else:
             print(f"  - Key: '{k}', Type: {type(v)}")
else:
    print(f"Scan ID {scan_id} not found in the features file.")