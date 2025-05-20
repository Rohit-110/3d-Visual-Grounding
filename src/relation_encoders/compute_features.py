# --- START OF REPLACEMENT compute_features.py ---

import argparse
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
import sys

# --- Use the CONSISTENT scene loader ---
try:
    from src.dataset.datasets import load_one_scene
    print("[Compute Features] Imported load_one_scene successfully.")
except ImportError:
    print("[Compute Features] ERROR: Cannot import load_one_scene from src.dataset.datasets. Ensure it's available.")
    sys.exit(1)

# --- Feature Encoder Imports ---
try:
    from src.relation_encoders.base import ScanReferCategoryConcept, CategoryConcept, Near, Far
    from src.relation_encoders.unary import AgainstTheWall, AtTheCorner, Tall, Low, OnTheFloor, Small, Large
    from src.relation_encoders.vertical import Above, Below
    from src.relation_encoders.ternary import Between as MiddleConcept
    from src.relation_encoders.view import Left, Right, Front, Behind
except ImportError as e:
    print(f"[Compute Features] ERROR: Cannot import feature encoder classes: {e}")
    sys.exit(1)

# --- Util Imports ---
try:
    from src.util.eval_helper import get_all_categories, get_all_relations
except ImportError as e:
    print(f"[Compute Features] ERROR: Cannot import helper functions: {e}")
    sys.exit(1)

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Compute Features] Using device: {DEVICE}")

# Relation Class Mapping (same as before)
concepts_classes = {
    "beside": Near,"near": Near,'next to': Near,"within": Near,"in": Near,
    "inside": Near,"close": Near,"closer": Near,"closest": Near,"far": Far,
    "farthest": Far,"opposite": Far,"furthest": Far,"corner": AtTheCorner,
    "against wall": AgainstTheWall,"above": Above,"on top": Above,'on the top': Above,
    "on": Above,"below": Below,"under": Below,"higher": Tall,"taller": Tall,
    "lower": Low,'beneath': Below,"on the floor": OnTheFloor,"smaller": Small,
    "shorter": Small,"larger": Large,"bigger": Large,"middle": MiddleConcept,
    "center": MiddleConcept,"between": MiddleConcept,'surrounded by': Near,
    'around': Near,"left": Left,"right": Right,"cross": Far,"across": Far,
    "front": Front,"back": Behind,"behind": Behind,"facing": Near,"with": Near,
    "attached": Near,'largest': Large,'highest': Tall,'upper': Tall,'longer': Large,
}

# Relation Arity Mapping (same as before)
concept_num = {
    Near: 2, Far: 2, AgainstTheWall: 1, Above: 2, Below: 2, Tall: 1, Low: 1,
    Small: 1, Large: 1, MiddleConcept: 3, Left: 2, Right: 2, Front: 2,
    AtTheCorner: 1, OnTheFloor: 1, Behind: 2,
}
rel_num = {k: concept_num[v] - 1 for k, v in concepts_classes.items()}
ALL_VALID_RELATIONS = set(concepts_classes.keys()) # Use set for faster lookups


# --- Modified Feature Computation Function ---
def get_all_features_from_data(scan_id, concept_names: set, scene_data: dict, dataset_type, label_type):
    """
    Computes features using object/scene data passed directly as a dictionary.
    Ensures consistency with the data loaded by load_one_scene.
    """
    all_concepts = {}

    # --- 1. Get Object Locations from provided scene_data ---
    if "pred_locs" not in scene_data or not scene_data["pred_locs"]:
        print(f"WARN: No 'pred_locs' found in scene_data for {scan_id}. Skipping feature computation.")
        return {} # Cannot compute without objects
    try:
        # Ensure it's a list of arrays/lists before vstack
        if isinstance(scene_data["pred_locs"], list) and len(scene_data["pred_locs"]) > 0:
             object_locations_np = np.vstack(scene_data["pred_locs"])
        elif isinstance(scene_data["pred_locs"], np.ndarray):
             object_locations_np = scene_data["pred_locs"]
        else: # Handle empty list case or unexpected type
             print(f"WARN: 'pred_locs' for {scan_id} is empty or has unexpected type {type(scene_data['pred_locs'])}. Skipping.")
             return {}
        object_locations = torch.from_numpy(object_locations_np).float().to(DEVICE)
    except Exception as e:
         print(f"ERROR processing 'pred_locs' for {scan_id}: {e}")
         return {}

    num_objects = object_locations.shape[0]
    if num_objects == 0:
        print(f"WARN: Zero objects found after processing 'pred_locs' for {scan_id}. Skipping.")
        return {}
    # print(f"[Debug] Computing features for {scan_id} with {num_objects} objects.") # Optional debug

    # --- 2. Initialize Category Concept Handler ---
    # IMPORTANT: Check if ScanReferCategoryConcept needs more than just scene_id from scan_data.
    # It uses scan_data['pred_labels'] in its _init_params.
    # Ensure load_one_scene provides 'pred_labels' in its returned dict,
    # OR modify ScanReferCategoryConcept to handle its absence / get labels differently.
    if dataset_type == "scanrefer":
        if 'pred_labels' not in scene_data:
             print(f"WARN: 'pred_labels' missing in scene_data for {scan_id}. ScanReferCategoryConcept might fail or use fallback.")
             # Depending on ScanReferCategoryConcept's implementation, this might be an error or just use defaults.
        try:
            category_concept = ScanReferCategoryConcept(scene_data, label_type=label_type) # Pass the whole dict
        except Exception as e:
             print(f"ERROR initializing ScanReferCategoryConcept for {scan_id}: {e}. Skipping feature computation.")
             return {}
    # Add Nr3D handling back if needed
    # elif dataset_type == "nr3d":
    #     category_concept = CategoryConcept(scene_data, label_type=label_type)
    else:
         print(f"ERROR: Unsupported dataset_type '{dataset_type}' in get_all_features_from_data.")
         return {}


    # --- 3. Prepare Room Points (Load only if needed) ---
    room_points = None
    needs_room_points = any(c in ["corner", "against wall", "on the floor"] for c in concept_names)
    if needs_room_points:
        if "room_points" in scene_data and scene_data["room_points"] is not None:
            try:
                room_points = torch.from_numpy(scene_data["room_points"]).float().to(DEVICE)
            except Exception as e:
                print(f"WARN: Failed to process 'room_points' for {scan_id}: {e}")
        else:
            print(f"WARN: room_points needed for concept in {scan_id} but not found in scene_data.")
            # Some features might fail to compute below if room_points is None

    # --- 4. Compute Each Concept ---
    # print(f"[Debug] Concepts to compute for {scan_id}: {concept_names}") # Optional
    for concept_name in concept_names:
        try:
            # --- Category Features ---
            if concept_name not in ALL_VALID_RELATIONS:
                concept_tensor = category_concept.forward(concept_name)
                # Basic Validation (ensure output has dimension N)
                if not torch.is_tensor(concept_tensor) or concept_tensor.dim() == 0 or concept_tensor.shape[0] != num_objects:
                    print(f"WARN: Category '{concept_name}' tensor shape {concept_tensor.shape if torch.is_tensor(concept_tensor) else 'N/A'} != num_objects {num_objects}. Skipping.")
                    continue
                all_concepts[concept_name] = concept_tensor.float() # Ensure float
                continue

            # --- Relational Features ---
            if concept_name not in concepts_classes:
                 print(f"WARN: Unknown relation '{concept_name}' found in concept_names. Skipping.")
                 continue

            relation_class = concepts_classes[concept_name]
            arity = concept_num[relation_class] # Arity 1, 2, or 3

            # Instantiate and compute based on arity
            if arity == 1: # Unary (e.g., Tall, AtCorner, OnTheFloor, AgainstWall)
                if concept_name in ["corner", "against wall", "on the floor"]:
                    if room_points is None: print(f"WARN: Skipping unary '{concept_name}' - required room_points not available."); continue
                    concept_instance = relation_class(object_locations, room_points)
                else:
                    # Assume other unary relations take (obj_locs, scene_points=None)
                    concept_instance = relation_class(object_locations, None) # Pass None if room_points not needed/available
                concept_tensor = concept_instance.forward().float()
                # Validation (Shape N,)
                if concept_tensor.shape != (num_objects,): print(f"WARN: Unary '{concept_name}' shape {concept_tensor.shape} != ({num_objects},). Skipping."); continue
                all_concepts[concept_name] = concept_tensor

            elif arity == 2: # Binary (e.g., Near, Above, Left)
                concept_instance = relation_class(object_locations)
                concept_tensor = concept_instance.forward().float()
                # Validation (Shape N,N)
                if concept_tensor.shape != (num_objects, num_objects): print(f"WARN: Binary '{concept_name}' shape {concept_tensor.shape} != ({num_objects},{num_objects}). Skipping."); continue
                all_concepts[concept_name] = concept_tensor

            elif arity == 3: # Ternary (Between)
                concept_instance = relation_class(object_locations)
                concept_tensor = concept_instance.forward().float()
                # Validation (Shape N,N,N)
                if concept_tensor.shape != (num_objects, num_objects, num_objects): print(f"WARN: Ternary '{concept_name}' shape {concept_tensor.shape} != ({num_objects},{num_objects},{num_objects}). Skipping."); continue
                all_concepts[concept_name] = concept_tensor

        except KeyError as e:
            print(f"WARN: KeyError processing concept '{concept_name}' for {scan_id}. Is it defined correctly? Error: {e}")
        except Exception as e:
            print(f"ERROR computing feature '{concept_name}' for {scan_id}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging errors within encoder classes

    # print(f"[Debug] Computed concepts for {scan_id}: {list(all_concepts.keys())}") # Optional
    return all_concepts


# --- Modified Main Function ---
def compute_all_features(args):
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # --- ScanRefer Specific Logic ---
    if args.dataset == "scanrefer":
        if args.label == "gt":
            raise NotImplementedError("GT labels not supported via this flow currently.")

        # Load ScanRefer JSON to get scan_ids and concepts needed
        scanrefer_json_path = "data/symbolic_exp/scanrefer.json" # Adjust if needed
        if not Path(scanrefer_json_path).exists():
             print(f"ERROR: Cannot find {scanrefer_json_path}")
             return

        print(f"[INFO] Loading ScanRefer data from {scanrefer_json_path}...")
        with open(scanrefer_json_path, 'r') as f:
            all_scanrefer_data = json.load(f)

        # Collect all concepts needed per scan_id
        concept_name_per_scene = defaultdict(set)
        scan_ids_to_process = set()
        print("[INFO] Collecting required concepts per scene...")
        for data in tqdm(all_scanrefer_data):
            scan_id = data["scan_id"]
            scan_ids_to_process.add(scan_id)
            json_obj = data["json_obj"]
            all_relations = get_all_relations(json_obj)
            all_categories = get_all_categories(json_obj)
            concept_name_per_scene[scan_id].update(all_categories)
            concept_name_per_scene[scan_id].update(all_relations)

        # Cache for scene data loaded via load_one_scene
        scan_data_cache = {}
        concepts_per_scene = defaultdict(dict) # Final results

        print("[INFO] Computing features per scene using load_one_scene...")
        # Process scans one by one
        for scan_id in tqdm(list(scan_ids_to_process)):
            # Load scene data using the consistent loader
            if scan_id not in scan_data_cache:
                # print(f"[Debug] Loading scene {scan_id} via load_one_scene...") # Verbose
                _, scene_data = load_one_scene(scan_id, label_type=args.label)
                # Basic validation of loaded data
                if not scene_data or "pred_locs" not in scene_data or not isinstance(scene_data["pred_locs"], (list, np.ndarray)):
                    print(f"WARN: load_one_scene failed or returned invalid data for {scan_id}. Skipping.")
                    continue
                if 'pred_labels' not in scene_data:
                     print(f"WARN: 'pred_labels' missing from load_one_scene output for {scan_id}. Category features might be affected.")
                     # Provide dummy if needed by ScanReferCategoryConcept, or modify it
                     # scene_data['pred_labels'] = ['object'] * len(scene_data['pred_locs'])
                scan_data_cache[scan_id] = scene_data
            else: # Should not happen if iterating scan_ids_to_process
                 scene_data = scan_data_cache[scan_id]

            # Get concepts needed for this specific scan
            concept_names = concept_name_per_scene[scan_id]

            # Compute features using the loaded, consistent scene data
            computed_concepts = get_all_features_from_data(
                scan_id,
                concept_names,
                scene_data, # Pass the dictionary
                args.dataset, # 'scanrefer'
                args.label # 'pred'
            )
            if computed_concepts: # Only add if computation didn't fail
                 concepts_per_scene[scan_id] = computed_concepts

        # --- Save the new, consistent features ---
        output_filename = f"scanrefer_features_per_scene_{args.label}_consistent.pth"
        save_path = output_path / output_filename
        print(f"[INFO] Saving consistently computed features to {save_path}")
        torch.save(dict(concepts_per_scene), save_path) # Convert back to dict for saving

    # --- Nr3D Logic (keep original if needed, or adapt similarly) ---
    elif args.dataset == "nr3d":
        # ... (Keep original Nr3D logic or adapt it like ScanRefer) ...
        print("Nr3D feature computation using this modified script requires similar adaptation.")
        raise NotImplementedError("Nr3D feature computation needs adaptation in this script.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute features consistently using load_one_scene.")
    parser.add_argument("--dataset", type=str, choices=["scanrefer", "nr3d"], default="scanrefer", help="Dataset to process")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save the features file")
    parser.add_argument("--label", type=str, choices=["pred", "gt"], default="pred", help="Label type to use (currently only 'pred' for scanrefer)")
    # Add other args if needed
    args = parser.parse_args()

    # Basic validation
    if args.dataset == "scanrefer" and args.label == "gt":
         print("ERROR: GT label type not supported for ScanRefer in this script.")
    else:
         compute_all_features(args)
    print("[INFO] Feature computation finished.")

# --- END OF REPLACEMENT compute_features.py ---