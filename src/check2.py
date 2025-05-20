# --- START OF FILE check2_final.py ---

import torch
import open3d as o3d
import numpy as np
import json
import argparse
import sys
import os
import re
from copy import deepcopy
import time
import torch.nn.functional as F # Needed by internal parse

# --- NLU and Synonym Imports ---
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import wordnet as wn
import google.generativeai as genai
from dotenv import load_dotenv

# --- Project Specific Imports ---
try: from src.dataset.datasets import load_one_scene; print("[INFO] Imported load_one_scene")
except ImportError: print("ERROR: Could not import 'load_one_scene'."); sys.exit(1)
try: from src.relation_encoders.compute_features import ALL_VALID_RELATIONS, rel_num; print("[INFO] Imported relation defs.")
except ImportError: print("ERROR: Could not import from compute_features."); sys.exit(1)

# === Setup ===
CHECK2_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARSER_DEVICE = CHECK2_DEVICE # Internal parse uses same device
print(f"[INFO] Using device: {CHECK2_DEVICE}")

# --- Configuration ---
# *** UPDATED DEFAULT PATH to the consistently generated features ***
DEFAULT_FEATURES_PATH = "output/scanrefer_features_per_scene_pred_consistent.pth"
RAW_PCD_PATH_TEMPLATE = "data/referit3d/scan_data/pcd_with_global_alignment/{scan_id}.pth"
SCANREFER_JSON_PATH = "data/symbolic_exp/scanrefer.json" # For target ID lookup

# --- NLTK Setup ---
# (Include download_nltk_resource function)
nltk.data.path.append('./nltk_data')
def download_nltk_resource(resource, download_name=None, type_path=None):
    if download_name is None: download_name = resource
    lookup_path = resource
    if type_path == 'tokenizers': lookup_path = f'tokenizers/{resource}'
    elif type_path == 'taggers': lookup_path = f'taggers/{resource}'
    elif type_path == 'chunkers': lookup_path = f'chunkers/{resource}'
    elif type_path == 'corpora': lookup_path = f'corpora/{resource}'
    try: nltk.data.find(lookup_path)
    except LookupError:
        print(f"Downloading NLTK resource: {download_name} to ./nltk_data")
        try: nltk.download(download_name, download_dir='./nltk_data')
        except Exception as e: print(f"ERROR downloading {download_name}: {e}")
download_nltk_resource('punkt', type_path='tokenizers')
download_nltk_resource('averaged_perceptron_tagger', type_path='taggers')
download_nltk_resource('maxent_ne_chunker', download_name='maxent_ne_chunker', type_path='chunkers')
download_nltk_resource('words', type_path='corpora')
download_nltk_resource('wordnet', type_path='corpora')

# --- Gemini API Setup & Synonym Function ---
# (Include get_synonyms_gemini function)
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Load from environment
gemini_model = None
if GEMINI_API_KEY:
    try: genai.configure(api_key=GEMINI_API_KEY); gemini_model = genai.GenerativeModel('gemini-1.5-flash'); print("[INFO] Gemini API configured.")
    except Exception as e: print(f"ERROR configuring Gemini API: {e}")
else: print("WARN: GOOGLE_API_KEY not found. Synonym lookup disabled.")

SYNONYM_TIMEOUT_SECONDS = 15
def get_synonyms_gemini(noun):
    if not hasattr(get_synonyms_gemini, "cache"): get_synonyms_gemini.cache = {}
    noun_key = noun.lower();
    if noun_key in get_synonyms_gemini.cache: return get_synonyms_gemini.cache[noun_key]
    if gemini_model is None: return []
    prompt = f"List exactly 5 single-word synonyms for the noun '{noun}'. Output only a comma-separated list, nothing else. Example: car,automobile,vehicle,motorcar,auto"
    synonyms = []
    try:
        print(f"[INFO] Querying Gemini for synonyms of '{noun}'...")
        start_time = time.time(); response = gemini_model.generate_content(prompt); duration = time.time() - start_time
        if duration > SYNONYM_TIMEOUT_SECONDS: print(f"WARN: Gemini query took {duration:.1f}s.")
        synonyms = [s.strip().lower() for s in response.text.split(',') if s.strip() and ' ' not in s.strip()]
        print(f"[INFO] Found synonyms: {synonyms[:5]}"); get_synonyms_gemini.cache[noun_key] = synonyms[:5]
    except Exception as e: print(f"ERROR: Gemini API call failed for '{noun}': {e}")
    return synonyms[:5]

# --- NLU Parser (Refined Heuristic) ---
# (Include parse_sentence_to_json_obj function)
def parse_sentence_to_json_obj(sentence, known_relations):
    # (Using the refined heuristic NLU parser from previous responses)
    print(f"[NLU] Parsing sentence: '{sentence}'")
    words = word_tokenize(sentence.lower()); tagged_words = pos_tag(words)
    grammar = r"NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}"; cp = RegexpParser(grammar); tree = cp.parse(tagged_words)
    noun_phrases = []
    np_subtrees = list(tree.subtrees(filter=lambda t: t.label() == 'NP'))
    word_indices = {word: i for i, (word, tag) in enumerate(tagged_words)}
    for subtree in np_subtrees:
        np_words = [w for w, p in subtree.leaves()]
        last_noun = next((w for w, p in reversed(subtree.leaves()) if p.startswith('NN')), None)
        if last_noun: start_index = word_indices.get(np_words[0], -1); noun_phrases.append({'phrase': " ".join(np_words), 'head': last_noun, 'subtree': subtree, 'start_index': start_index})
    noun_phrases.sort(key=lambda x: x['start_index'] if x['start_index'] != -1 else float('inf'))
    if not noun_phrases:
        print("[NLU] No NP found. Fallback."); nouns = [w for w, p in tagged_words if p.startswith('NN')]
        if not nouns: print("[NLU] No nouns found."); return None
        target_category = nouns[0]; anchor_candidates_list = [{'head': h, 'subtree': None, 'start_index': word_indices.get(h, float('inf'))} for h in nouns[1:]]
        print(f"[NLU-Fallback] Target: '{target_category}', Anchors: {[a['head'] for a in anchor_candidates_list]}")
    else:
        target_category = noun_phrases[0]['head']; anchor_candidates_list = noun_phrases[1:]
        print(f"[NLU] Target NP Head: '{target_category}'"); print(f"[NLU] Anchor Candidates: {[a['phrase'] for a in anchor_candidates_list]}")

    json_obj = {"category": target_category, "relations": []}; relations_found = []; used_anchor_indices = set(); last_relation_index = -1
    for i, (word, tag) in enumerate(tagged_words):
        potential_rel, rel_word = None, word
        # Relation keyword mapping
        if word in known_relations: potential_rel = word
        elif word in ['on', 'above', 'top']: potential_rel = 'above'
        elif word in ['under', 'below', 'beneath']: potential_rel = 'below'
        elif word in ['near', 'beside', 'next', 'close', 'closest', 'by']: potential_rel = 'near'
        elif word in ['between']: potential_rel = 'between'
        elif word in ['behind', 'back']: potential_rel = 'behind'
        elif word == 'front':
             if i > 0 and tagged_words[i-1][0] == 'in': potential_rel = 'front'; rel_word = "in front"
        elif word in ['left']: potential_rel = 'left'
        elif word in ['right']: potential_rel = 'right'
        elif word == 'corner': potential_rel = 'corner'
        elif word == 'to' and i > 0 and tagged_words[i-1][0] == 'left': potential_rel = 'left'; rel_word = "left to"
        elif word == 'to' and i > 0 and tagged_words[i-1][0] == 'right': potential_rel = 'right'; rel_word = "right to"
        if potential_rel and potential_rel not in known_relations: potential_rel = None
        if potential_rel:
            last_relation_index = i; relation_struct = {"relation_name": potential_rel, "objects": []}
            anchor_found_head = None; first_anchor_subtree = None; best_anchor_cand_idx = -1; min_start_index_diff = float('inf')
            for cand_idx, anchor_cand in enumerate(anchor_candidates_list):
                 if cand_idx in used_anchor_indices: continue
                 anchor_start_idx = anchor_cand['start_index']
                 if anchor_start_idx > i:
                     dist = anchor_start_idx - i
                     if dist < min_start_index_diff: min_start_index_diff = dist; anchor_found_head = anchor_cand['head']; first_anchor_subtree = anchor_cand['subtree']; best_anchor_cand_idx = cand_idx
            if anchor_found_head:
                relation_struct["objects"].append({"category": anchor_found_head, "relations": []})
                if best_anchor_cand_idx != -1 : used_anchor_indices.add(best_anchor_cand_idx)
                if potential_rel == 'between' and best_anchor_cand_idx != -1:
                    second_anchor_found_head = None; best_second_cand_idx = -1; min_start_index_diff2 = float('inf'); first_anchor_start = anchor_candidates_list[best_anchor_cand_idx]['start_index']
                    for cand2_idx, anchor2_cand in enumerate(anchor_candidates_list):
                         if cand2_idx in used_anchor_indices: continue
                         anchor2_start_idx = anchor2_cand['start_index']
                         if anchor2_start_idx > first_anchor_start:
                            dist2 = anchor2_start_idx - first_anchor_start
                            if dist2 < min_start_index_diff2: min_start_index_diff2 = dist2; second_anchor_found_head = anchor2_cand['head']; best_second_cand_idx = cand2_idx
                    if second_anchor_found_head:
                         relation_struct["objects"].append({"category": second_anchor_found_head, "relations": []})
                         if best_second_cand_idx != -1: used_anchor_indices.add(best_second_cand_idx)
            if i > 0 and tagged_words[i-1][0] == 'not': relation_struct["negative"] = True
            is_unary = rel_num.get(potential_rel, -1) == 0
            if relation_struct["objects"] or is_unary: relations_found.append(relation_struct)
    json_obj["relations"] = relations_found
    print(f"[NLU] Parsed json_obj: {json.dumps(json_obj, indent=2)}")
    return json_obj

# +++ Internal Parse Function (Copied from user's provided check2.py - with added checks) +++
def parse(scan_id, json_obj, all_concepts, current_device):
    """
    Evaluate the symbolic parse tree to a score vector over objects.
    *** Uses INTERNAL logic - Requires Feature Dimension Match ***
    """
    category = json_obj.get("category", "object")
    if category not in all_concepts:
        print(f"ERROR in parse: Category '{category}' not found in prepared features for {scan_id}. Available: {list(all_concepts.keys())}")
        return None
    if not torch.is_tensor(all_concepts[category]):
         print(f"ERROR in parse: Category feature '{category}' is not a tensor.")
         return None

    appearance_concept = all_concepts[category]
    # *** Infer N safely ***
    if appearance_concept.dim() == 0:
         print(f"ERROR in parse: Category feature '{category}' is a 0-dim tensor. Cannot infer N.")
         return None
    num_objects = appearance_concept.shape[0]
    final_concept = torch.ones(num_objects, device=current_device)

    if category in ["corner", "middle", "room", "center"]:
        print(f"INFO in parse: Handling abstract category '{category}'.")
        return appearance_concept.clone()

    final_concept = torch.minimum(final_concept, appearance_concept)

    if "relations" in json_obj:
        for relation_item in json_obj["relations"]:
            # ... (rest of relation processing logic as before, with try/except) ...
            if "anchors" in relation_item: relation_item["objects"] = relation_item["anchors"]
            relation_name = relation_item.get("relation_name")
            if not relation_name or relation_name not in ALL_VALID_RELATIONS or relation_name not in all_concepts: print(f"WARN in parse: Relation '{relation_name}' invalid/missing. Skipping."); continue
            relation_concept = all_concepts[relation_name]
            if not torch.is_tensor(relation_concept): print(f"WARN in parse: Relation '{relation_name}' not tensor. Skipping."); continue

            num = rel_num.get(relation_name, -1)
            concept = torch.ones((num_objects,), device=current_device)

            try:
                if num == 0: # Unary
                    sub_objects = relation_item.get("objects", [])
                    expected_shape1 = (num_objects,)
                    expected_shape2 = (num_objects, num_objects)
                    if sub_objects: # Relative to anchor
                        sub_concept = parse(scan_id, sub_objects[0], all_concepts, current_device)
                        if sub_concept is None: print(f"WARN: Anchor parse failed for unary '{relation_name}'. Skipping."); continue
                        if relation_concept.shape == expected_shape1: concept = relation_concept * concept
                        elif relation_concept.shape == expected_shape2: concept = (relation_concept @ sub_concept)
                        else: print(f"WARN: Dim mismatch unary '{relation_name}' (Shape: {relation_concept.shape}). Skip."); continue
                    else: # Applies to self
                         if relation_concept.shape == expected_shape1: concept = relation_concept * concept
                         elif relation_concept.shape == expected_shape2: print(f"WARN: Unary '{relation_name}' NxN feature. Using diag."); concept = relation_concept.diag() * concept
                         else: print(f"WARN: Dim mismatch self-unary '{relation_name}' (Shape: {relation_concept.shape}). Skip."); continue

                elif num == 1: # Binary
                    sub_objects = relation_item.get("objects", [])
                    if not sub_objects: print(f"WARN: Binary '{relation_name}' needs anchor. Skip."); continue
                    sub_concept = parse(scan_id, sub_objects[0], all_concepts, current_device)
                    if sub_concept is None: print(f"WARN: Anchor parse failed for binary '{relation_name}'. Skipping."); continue
                    expected_shape = (num_objects, num_objects)
                    if relation_concept.shape == expected_shape: concept = torch.matmul(relation_concept, sub_concept)
                    else: print(f"WARN: Dim mismatch binary '{relation_name}' (Shape: {relation_concept.shape}). Skip."); continue

                elif num == 2: # Ternary
                    objs = relation_item.get("objects", [])
                    if len(objs) < 2: print(f"WARN: Ternary '{relation_name}' needs 2 anchors. Skip."); continue
                    sub1 = parse(scan_id, objs[0], all_concepts, current_device)
                    sub2 = parse(scan_id, objs[1], all_concepts, current_device)
                    if sub1 is None or sub2 is None: print(f"WARN: Anchor parse failed for ternary '{relation_name}'. Skipping."); continue
                    expected_shape = (num_objects, num_objects, num_objects)
                    if relation_concept.shape == expected_shape: concept = torch.einsum('ijk,j,k->i', relation_concept, sub1, sub2)
                    else: print(f"WARN: Dim mismatch ternary '{relation_name}' (Shape: {relation_concept.shape}). Skip."); continue
                else: print(f"WARN: Unknown arity {num} for '{relation_name}'. Skip."); continue

                # Combine concept
                if relation_item.get("negative", False):
                    if torch.isfinite(concept).all(): concept = concept.max() - concept
                    else: print(f"WARN: Cannot negate non-finite concept for '{relation_name}'."); concept = torch.zeros_like(concept)
                final_concept = final_concept * concept
            except RuntimeError as e: print(f"ERROR: Runtime error processing relation '{relation_name}': {e}. Skip."); continue

    # Final checks
    final_concept = torch.clamp(final_concept, min=0.0)
    if final_concept.sum() == 0: print("WARN: Final scores sum to zero. Returning uniform."); return torch.ones_like(final_concept) / final_concept.numel() if final_concept.numel() > 0 else None
    if not torch.isfinite(final_concept).all(): print("ERROR: Final concept NaN/Inf. Returning None."); return None
    return final_concept
# +++ End Internal Parse Function +++


# --- Helper Function Definitions ---
# (Include ALL helpers: prepare_features(strict), find_target_id_for_sentence, collect_concepts, process_prediction, build_point_cloud, build_bboxes, print_summary, visualize)
# (Definitions are unchanged from check2_final_attempt.py response)
# ... [Copy all helpers here] ...
def prepare_features(features_this_scene, num_objects, device):
    features_out = {}; feature_dim = -1; keys_kept = []
    if not isinstance(features_this_scene, dict): print(f"ERROR: features_this_scene not dict."); return {}, -1
    for k, v in features_this_scene.items():
        if torch.is_tensor(v):
            is_valid_dim = False; shape = v.shape
            if len(shape) > 0 and shape[0] == num_objects:
                if len(shape) == 1: is_valid_dim = True
                elif len(shape) >= 2 and shape[1] == num_objects:
                    if len(shape) == 2: is_valid_dim = True
                    elif len(shape) >= 3 and shape[2] == num_objects: is_valid_dim = True
            if is_valid_dim:
                if feature_dim == -1: feature_dim = shape[0]
                elif feature_dim != shape[0]: continue
                features_out[k] = v.to(device); keys_kept.append(k)
        else: features_out[k] = v; keys_kept.append(k)
    if not features_out or not any(torch.is_tensor(features_out[k]) for k in keys_kept if k in features_out): return {}, -1
    return features_out, feature_dim

def find_target_id_for_sentence(json_path, target_scan_id, target_sentence, obj_ids_in_scene):
    if not os.path.exists(json_path): return -1, -1
    target_sentence_norm = target_sentence.strip(); target_id, target_idx = -1, -1
    try:
        with open(json_path, 'r') as f:
            scanrefer_data = json.load(f)
            for data in scanrefer_data:
                if data.get("scan_id") == target_scan_id and data.get("caption", "").strip() == target_sentence_norm:
                    target_id = data.get("target_id", -1)
                    if target_id != -1:
                        try: target_idx = obj_ids_in_scene.index(target_id)
                        except ValueError: target_idx = -1
                    return target_id, target_idx # Return first match
    except Exception as e: print(f"WARN: Error reading {json_path} for target ID lookup: {e}")
    return -1, -1

def collect_concepts(obj, concepts_set, known_relations):
    if isinstance(obj, dict):
        if "category" in obj and obj["category"] not in ["corner", "middle", "room", "center", "you"]: concepts_set.add(obj["category"])
        if "relation_name" in obj and obj.get("relation_name") in known_relations: concepts_set.add(obj["relation_name"])
        for key, value in obj.items():
            if key in ["relations", "objects"]: collect_concepts(value, concepts_set, known_relations)
    elif isinstance(obj, list):
         for item in obj: collect_concepts(item, concepts_set, known_relations)

def process_prediction(final_concept, num_objects, obj_ids_in_scene):
    predicted_obj_idx, predicted_internal_id = -1, -1
    if final_concept is None or final_concept.numel() == 0: print(f"Error: Parse failed/invalid. No prediction."); return -1, -1
    if not torch.isfinite(final_concept).all(): print(f"Error: Final concept NaN/Inf. No prediction."); return -1,-1
    if final_concept.sum().item() == 0: print(f"Warning: All scores zero. No prediction."); return -1, -1
    if final_concept.shape[0] != num_objects: print(f"CRITICAL WARN: Score count ({final_concept.shape[0]}) != object count ({num_objects}). No prediction."); return -1, -1
    predicted_obj_idx = torch.argmax(final_concept).item()
    if 0 <= predicted_obj_idx < len(obj_ids_in_scene): predicted_internal_id = obj_ids_in_scene[predicted_obj_idx]
    return predicted_obj_idx, predicted_internal_id

def build_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud();
    if points is not None: pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        try: pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception as e: print(f"WARN: Failed to set colors: {e}."); pcd.paint_uniform_color([0.5, 0.5, 0.5])
    elif pcd.has_points(): pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd

def build_bboxes(obj_locs, highlight_index):
    bbox_list = [];
    for i, bbox in enumerate(obj_locs):
        center, size = bbox[:3], np.maximum(bbox[3:], 1e-6)
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)): print(f"Warn: Invalid bbox index {i}. Skipping."); continue
        try: obb = o3d.geometry.OrientedBoundingBox(center.tolist(), np.eye(3).tolist(), size.tolist()); obb.color = (0, 1, 0) if i == highlight_index else (1, 0, 0); bbox_list.append(obb)
        except Exception as e: print(f"Error creating OBB index {i}: {e}")
    return bbox_list

def print_summary(scan_id, caption, target_id, target_idx, pred_idx, pred_id):
    print("\n" + "="*30 + " RESULTS " + "="*30); print(f"Scan ID: {scan_id}"); print(f"Input Sentence: '{caption}'"); print("-" * 69)
    print(f"Target ID (lookup): {target_id if target_id != -1 else 'N/A'}"); print(f"Target Index (lookup): {target_idx if target_idx != -1 else 'N/A'}"); print("-" * 69)
    print(f"Predicted Index: {pred_idx if pred_idx != -1 else 'N/A'}"); print(f"Predicted Internal ID: {pred_id if pred_id != -1 else 'N/A'}"); print("-" * 69)
    if target_id != -1 and target_idx != -1 and pred_id != -1:
         if pred_id == target_id: print("--- PREDICTION MATCHES TARGET (based on lookup) ---")
         else: print("--- PREDICTION DOES NOT MATCH TARGET (based on lookup) ---")
    elif target_id != -1: print("--- CANNOT VERIFY MATCH ---")
    print("="* 69 + "\n")

def visualize(pcd, bbox_list, scan_id, dataset_idx, pred_id, target_id):
    print(f"Visualizing {scan_id}...")
    geometries = [];
    if pcd.has_points(): geometries.append(pcd)
    if bbox_list: geometries.extend(bbox_list)
    if geometries: title_idx = f" | Idx:{dataset_idx}" if dataset_idx != -1 else ""; window_title = f"{scan_id}{title_idx} | Pred:{pred_id if pred_id!=-1 else 'None'} | Target:{target_id if target_id!=-1 else '?'}" ; o3d.visualization.draw_geometries(geometries, window_name=window_title)
    else: print("Error: No points and no valid bounding boxes to visualize.")

def add_to_parser(parser):
    """Add the no_visualization flag to the argument parser"""
    parser.add_argument("--no_visualization", action="store_true", help="Disable visualization window (for headless environments)")
    return parser

# Modify the visualize function to accept a no_visualization parameter
def visualize_modified(pcd, bbox_list, scan_id, dataset_idx, pred_id, target_id, no_visualization=False):
    """Modified visualize function that can skip visualization"""
    print(f"Visualizing {scan_id}...")
    geometries = []
    if pcd.has_points():
        geometries.append(pcd)
    if bbox_list:
        geometries.extend(bbox_list)
    
    if geometries:
        title_idx = f" | Idx:{dataset_idx}" if dataset_idx != -1 else ""
        window_title = f"{scan_id}{title_idx} | Pred:{pred_id if pred_id!=-1 else 'None'} | Target:{target_id if target_id!=-1 else '?'}"
        
        if not no_visualization:
            import open3d as o3d
            o3d.visualization.draw_geometries(geometries, window_name=window_title)
        return True
    else:
        print("Error: No points and no valid bounding boxes to visualize.")
        return False


# --- Main Execution ---
def main(args):
    scan_id = args.scan_id; sentence = args.sentence; features_path = args.features_path
    json_path = args.scanrefer_json_path
    if not os.path.exists(json_path): print(f"WARN: ScanRefer JSON {json_path} not found.")
    # *** Ensure the specified features file exists ***
    if not os.path.exists(features_path): print(f"ERROR: Features file not found: {features_path}"); sys.exit(1)

    # --- NLU Parsing ---
    parsed_json_obj = parse_sentence_to_json_obj(sentence, ALL_VALID_RELATIONS)
    if parsed_json_obj is None: print("ERROR: NLU parsing failed."); sys.exit(1)

    # --- Load Scene Data ---
    print(f"[INFO] Loading scene data for {scan_id}...")
    _, scene = load_one_scene(scan_id, label_type="pred") # Use pred to match feature expectation
    if not scene or 'pred_locs' not in scene or not scene['pred_locs']: print(f"ERROR: Failed to load scene data for {scan_id}"); sys.exit(1)
    room_points, obj_locs = scene.get("room_points"), scene["pred_locs"]
    obj_ids_in_scene = scene.get('obj_ids', list(range(len(obj_locs))))
    num_objects = len(obj_locs)
    if num_objects == 0: print(f"ERROR: No predicted objects loaded for {scan_id}."); sys.exit(1)
    print(f"[INFO] Loaded {num_objects} predicted objects.")


    # --- Load Point Cloud ---
    pth_path = RAW_PCD_PATH_TEMPLATE.format(scan_id=scan_id)
    points_raw, colors_raw = None, None
    try:
        points_raw, colors_raw, _, _ = torch.load(pth_path, map_location="cpu", weights_only=False)
        if colors_raw is not None and colors_raw.max() > 1.0: colors_raw = colors_raw / 255.0
        if points_raw is not None and room_points is not None and not np.allclose(points_raw[:100], room_points[:100]): print("WARN: Raw points differ. Using raw points."); room_points = points_raw
    except Exception as e: print(f"WARN: Could not load raw point cloud {pth_path}: {e}"); points_raw = room_points; colors_raw = np.ones_like(room_points) * 0.5 if room_points is not None else None


    # --- Load & Prepare Features (Using STRICT Check) ---
    print(f"[INFO] Loading features from {features_path}...")
    all_features = torch.load(features_path, map_location="cpu", weights_only=False) # Use False if file needs it
    if scan_id not in all_features: print(f"ERROR: Scan ID {scan_id} not found in {features_path}"); sys.exit(1)
    features_this_scene = all_features[scan_id]

    # *** Use the STRICT feature preparer ***
    features_this_scene_for_parser, feature_dim = prepare_features(features_this_scene, num_objects, PARSER_DEVICE)

    # *** CRITICAL CHECK: Exit if no usable features found ***
    if not features_this_scene_for_parser or not any(torch.is_tensor(t) for t in features_this_scene_for_parser.values()):
         print(f"FATAL ERROR: No usable *tensor* features prepared for parsing. Cannot continue.")
         print(f"           Please ensure features were generated correctly with matching dimensions.")
         print(f"           Feature path: {features_path}")
         print(f"           Scan ID: {scan_id}, Expected Objects: {num_objects}")
         print(f"           Original keys found in file: {list(features_this_scene.keys())}")
         # Add detail about prepared keys vs original if helpful
         print(f"           Keys prepared (check shapes): {list(features_this_scene_for_parser.keys())}")
         sys.exit(1)

    if feature_dim != -1 and feature_dim != num_objects: print(f"CRITICAL WARN: Feature dimension ({feature_dim}) != object count ({num_objects})!")

    available_concepts = list(features_this_scene_for_parser.keys())
    print(f"[INFO] Available concepts prepared for parsing: {available_concepts}")

    # --- Synonym Handling ---
    main_category = parsed_json_obj['category']
    if main_category not in available_concepts:
        print(f"WARN: Parsed category '{main_category}' not in features. Trying synonyms...")
        synonyms = get_synonyms_gemini(main_category)
        found_synonym = False
        if synonyms:
            for syn in synonyms:
                if syn in available_concepts:
                    print(f"[INFO] Using synonym '{syn}' for '{main_category}'.")
                    parsed_json_obj['category'] = syn; found_synonym = True; break
        if not found_synonym: print(f"WARN: No valid synonyms found for '{main_category}'.")

    # --- Concept Validation ---
    required_concepts = set(); collect_concepts(parsed_json_obj, required_concepts, ALL_VALID_RELATIONS)
    print(f"[INFO] Concepts required by final json_obj: {required_concepts}")
    missing_concepts = required_concepts - set(available_concepts)
    final_concept = None
    if missing_concepts:
        print(f"ERROR: Required concepts MISSING from prepared features: {missing_concepts}")
        print("       Cannot run parse function.")
    else:
        print("[INFO] All required concepts found. Running INTERNAL parse function...")
        try:
            # *** Call the INTERNAL parse function defined in this file ***
            final_concept = parse(scan_id, parsed_json_obj, features_this_scene_for_parser, PARSER_DEVICE)
            if final_concept is not None: final_concept = final_concept.to(CHECK2_DEVICE)
            print(f"[INFO] Internal parse returned concept scores (shape {final_concept.shape if final_concept is not None else 'None'})")
        except KeyError as e: print(f"ERROR: KeyError during parse - missing concept '{e}'.")
        except RuntimeError as e: print(f"ERROR: RuntimeError during parse - LIKELY DIMENSION MISMATCH: {e}"); import traceback; traceback.print_exc()
        except Exception as e: print(f"ERROR: Unexpected error during parse: {e}"); import traceback; traceback.print_exc()

    # --- Process prediction ---
    predicted_obj_idx, predicted_internal_id = process_prediction(final_concept, num_objects, obj_ids_in_scene)

    # --- Find Target ID for reference (optional) ---
    target_id_from_example, target_idx_in_list = find_target_id_for_sentence(json_path, scan_id, sentence, obj_ids_in_scene)

    # --- Build point cloud & BBoxes ---
    pcd = build_point_cloud(points_raw if points_raw is not None else room_points, colors_raw)
    bbox_list = build_bboxes(obj_locs, predicted_obj_idx)

    # --- Final Output and Visualization ---
    print_summary(scan_id, sentence, target_id_from_example, target_idx_in_list, predicted_obj_idx, predicted_internal_id)
    visualize(pcd, bbox_list, scan_id, -1, predicted_internal_id, target_id_from_example)

# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sentence, handle synonyms, run INTERNAL parse logic, visualize.")
    parser.add_argument("--scan_id", type=str, required=True, help="Scan ID (e.g., scene0025_00)")
    parser.add_argument("--sentence", type=str, required=True, help="Natural language sentence describing the target object.")
    parser.add_argument("--features_path", type=str, default="output/scanrefer_features_per_scene_pred_consistent.pth", help=f"Path to *consistent* ScanRefer features file (.pth)")
    parser.add_argument("--scanrefer_json_path", type=str, default=SCANREFER_JSON_PATH, help=f"Path to scanrefer.json (for target ID lookup) [default: {SCANREFER_JSON_PATH}]")
    parser = add_to_parser(parser)  # Add the no_visualization flag
    
    args = parser.parse_args()
    if not args.scan_id or not args.sentence:
        parser.error("Both --scan_id and --sentence are required.")
    
    main(args)  # Call the original main function
# Example usage:
# python check2_final.py --scan_id scene0025_00 --sentence "the sofa left to the door" --features_path output/scanrefer_features_per_scene_pred_consistent.pth
# python check2_final.py --scan_id scene0025_00 --sentence "this is a small gray pillow that is crumpled up . it is in the corner of a black couch , near a wooden door ." --features_path output/scanrefer_features_per_scene_pred_consistent.pth