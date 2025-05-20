# --- START OF FILE check3.py (Modified for Activity Suggestions) ---

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
import torch.nn.functional as F

# --- NLU and Synonym Imports ---
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import wordnet as wn
import google.generativeai as genai
from dotenv import load_dotenv

# --- Project Specific Imports ---
try:
    from src.dataset.datasets import load_one_scene
except ImportError: print("ERROR: Could not import 'load_one_scene'."); sys.exit(1)
try:
    from src.relation_encoders.compute_features import ALL_VALID_RELATIONS, rel_num
except ImportError: print("ERROR: Could not import from compute_features."); sys.exit(1)

# --- Configuration ---
DEFAULT_FEATURES_PATH = "output/scanrefer_features_per_scene_pred_consistent.pth" # Assumes consistent features exist
RAW_PCD_PATH_TEMPLATE = "data/referit3d/scan_data/pcd_with_global_alignment/{scan_id}.pth"
SCANREFER_JSON_PATH = "data/symbolic_exp/scanrefer.json" # For target ID lookup
INSTANCE_ID_PATH_TEMPLATE = "data/referit3d/scan_data/instance_id_to_name/{scan_id}.json"

# --- Device Setup ---
CHECK2_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARSER_DEVICE = CHECK2_DEVICE # Internal parse uses same device

# --- NLTK Setup ---
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
# Download necessary resources on first run
def setup_nltk():
    print("[INFO] Setting up NLTK...")
    download_nltk_resource('punkt', type_path='tokenizers')
    download_nltk_resource('averaged_perceptron_tagger', type_path='taggers')
    download_nltk_resource('maxent_ne_chunker', download_name='maxent_ne_chunker', type_path='chunkers')
    download_nltk_resource('words', type_path='corpora')
    download_nltk_resource('wordnet', type_path='corpora')
    print("[INFO] NLTK setup complete.")

# --- Gemini API Setup & Synonym Function ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Load from environment
gemini_model = None
SYNONYM_TIMEOUT_SECONDS = 15
synonym_cache = {} # Simple in-memory cache for synonyms per run

def configure_gemini():
    global gemini_model
    if GEMINI_API_KEY and gemini_model is None: # Configure only once
        try:
            print("[INFO] Configuring Gemini API...")
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("[INFO] Gemini API configured.")
        except Exception as e:
            print(f"ERROR configuring Gemini API: {e}")
            gemini_model = None # Ensure it's None on failure
    elif not GEMINI_API_KEY:
         print("WARN: GOOGLE_API_KEY not found. Gemini API disabled.")

def get_synonyms_gemini(noun):
    global synonym_cache # Use simple in-memory cache
    noun_key = noun.lower();
    if noun_key in synonym_cache: return synonym_cache[noun_key]
    if gemini_model is None: return []
    # No rate limiting here, assumes single call from check2/dashboard use case
    prompt = f"List exactly 5 single-word synonyms for the noun '{noun}'. Output only a comma-separated list, nothing else. Example: car,automobile,vehicle,motorcar,auto"
    synonyms = []
    try:
        print(f"[INFO] Querying Gemini for synonyms of '{noun}'...")
        response = gemini_model.generate_content(prompt, request_options={'timeout': SYNONYM_TIMEOUT_SECONDS}) # Add timeout
        synonyms = [s.strip().lower() for s in response.text.split(',') if s.strip() and ' ' not in s.strip()][:5]
        print(f"[INFO] Found synonyms: {synonyms}"); synonym_cache[noun_key] = synonyms
    except Exception as e: print(f"ERROR: Gemini API call failed for '{noun}': {e}")
    return synonyms

# --- Activity Suggestion Function ---
def suggest_activity_with_object(prompt, objects_list):
    """
    Use Gemini to suggest an activity with an object from the list based on the prompt
    """
    if gemini_model is None:
        print("ERROR: Gemini API not configured. Cannot suggest activities.")
        return None
    
    if not objects_list:
        print("ERROR: No objects available in the scene.")
        return None
    
    # Format the objects list for the prompt
    objects_str = ", ".join(objects_list)
    
    # Create a prompt for Gemini
    suggestion_prompt = f"""
    A person said: "{prompt}"
    
    I have the following objects in a 3D scene: {objects_str}
    
    Please suggest ONE specific activity they could do with ONE of these objects to address their prompt.
    Format your response as: "You could [action] with the [object]" or "You could use the [object] to [action]".
    Keep it brief and direct (maximum 2 sentences).
    """
    
    try:
        print(f"[INFO] Querying Gemini for activity suggestion...")
        response = gemini_model.generate_content(
            suggestion_prompt, 
            request_options={'timeout': SYNONYM_TIMEOUT_SECONDS}
        )
        suggestion = response.text.strip()
        print(f"[INFO] Suggestion: {suggestion}")
        return suggestion
    except Exception as e:
        print(f"ERROR: Gemini API call failed for activity suggestion: {e}")
        return "I couldn't come up with a suggestion at the moment."

# --- Load Instance Names Function ---
def load_instance_names(scan_id):
    """
    Load the instance ID to name mapping for the given scan ID
    """
    instance_path = INSTANCE_ID_PATH_TEMPLATE.format(scan_id=scan_id)
    if not os.path.exists(instance_path):
        print(f"ERROR: Instance ID mapping file not found: {instance_path}")
        return {}
    
    try:
        with open(instance_path, 'r') as f:
            instance_mapping = json.load(f)
        return instance_mapping
    except Exception as e:
        print(f"ERROR loading instance mapping file {instance_path}: {e}")
        return {}

# --- Process Prompt Function ---
def process_scene(scan_id, prompt):
    """
    Process a user prompt for activity suggestions in a scene
    """
    print(f"\n--- Processing prompt for {scan_id}: '{prompt}' ---")
    
    # 1. Configure Gemini
    configure_gemini()
    if gemini_model is None:
        print("ERROR: Gemini API not configured. Cannot process prompt.")
        return None
    
    # 2. Load instance ID to name mapping
    print(scan_id)
    instance_mapping = load_instance_names(scan_id)
    print(instance_mapping)
    if not instance_mapping:
        print(f"ERROR: Could not load instance mapping for {scan_id}")
        return None
    
    # 3. Extract object names
    object_names = instance_mapping
    print(f"[INFO] Found {len(object_names)} unique objects in scene: {object_names}")
    
    # 4. Generate activity suggestion
    suggestion = suggest_activity_with_object(prompt, object_names)
    
    # 5. Load scene for visualization (optional)
    try:
        _, scene = load_one_scene(scan_id, label_type="pred")
        if not scene or 'pred_locs' not in scene or not scene['pred_locs']:
            print(f"WARN: Failed to load scene data for visualization.")
            scene = None
    except Exception as e:
        print(f"WARN: Error loading scene: {e}")
        scene = None
    
    # Return results
    results = {
        "scan_id": scan_id,
        "prompt": prompt,
        "suggestion": suggestion,
        "object_names": object_names,
        "scene": scene
    }
    return results

# --- Visualization for Activity Suggestion ---
def visualize_suggestion(scan_id, suggestion, scene=None):
    """Visualize the scene with highlighted relevant objects"""
    if scene is None:
        print("[INFO] No scene data available for visualization.")
        return
    
    # Extract the object name from the suggestion
    object_name_match = re.search(r"the\s+(\w+)", suggestion.lower())
    target_object = object_name_match.group(1) if object_name_match else None
    
    # Load point cloud
    pth_path = RAW_PCD_PATH_TEMPLATE.format(scan_id=scan_id)
    points_raw, colors_raw = None, None
    try:
        points_raw, colors_raw, _, _ = torch.load(pth_path, map_location="cpu", weights_only=False)
        if colors_raw is not None and colors_raw.max() > 1.0: colors_raw = colors_raw / 255.0
    except Exception as e:
        print(f"WARN: Could not load raw pcd {pth_path}: {e}")
        room_points = scene.get("room_points")
        points_raw = room_points
        colors_raw = np.ones_like(room_points) * 0.5 if room_points is not None else None
    
    # Build point cloud
    pcd = build_point_cloud(points_raw, colors_raw)
    
    # Load instance mapping
    instance_mapping = load_instance_names(scan_id)
    
    # Find objects matching the target
    highlighted_indices = []
    if target_object:
        for idx, name in enumerate(instance_mapping):
            if target_object.lower() in name.lower():
                try:
                    obj_idx = scene.get('obj_ids', []).index(int(idx))
                    highlighted_indices.append(obj_idx)
                except (ValueError, TypeError):
                    continue
    
    # Build bounding boxes
    obj_locs = scene.get("pred_locs", [])
    bbox_list = []
    for i, bbox in enumerate(obj_locs):
        center, size = bbox[:3], np.maximum(bbox[3:], 1e-6)
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)):
            continue
        try:
            obb = o3d.geometry.OrientedBoundingBox(center.tolist(), np.eye(3).tolist(), size.tolist())
            # Highlight objects that match the target
            obb.color = (0, 1, 0) if i in highlighted_indices else (1, 0, 0)
            bbox_list.append(obb)
        except Exception as e:
            print(f"Error creating OBB index {i}: {e}")
    
    # Visualize
    geometries = []
    if pcd is not None and pcd.has_points(): geometries.append(pcd)
    if bbox_list: geometries.extend(bbox_list)
    
    if geometries:
        window_title = f"{scan_id} - Suggestion: {suggestion}"
        o3d.visualization.draw_geometries(geometries, window_name=window_title)
    else:
        print("Error: No points and no valid bounding boxes to visualize.")

# --- Original Helper Functions (Unmodified) ---
def parse_sentence_to_json_obj(sentence, known_relations):
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

def parse(scan_id, json_obj, all_concepts, current_device):
    category = json_obj.get("category", "object")
    if category not in all_concepts: return None
    if not torch.is_tensor(all_concepts[category]): return None
    appearance_concept = all_concepts[category]
    if appearance_concept.dim() == 0 or appearance_concept.shape[0] == 0: return None
    num_objects = appearance_concept.shape[0]; final_concept = torch.ones(num_objects, device=current_device)
    if category in ["corner", "middle", "room", "center"]: return appearance_concept.clone()
    final_concept = torch.minimum(final_concept, appearance_concept)
    if "relations" in json_obj:
        for relation_item in json_obj["relations"]:
            if "anchors" in relation_item: relation_item["objects"] = relation_item["anchors"]
            relation_name = relation_item.get("relation_name")
            if not relation_name or relation_name not in ALL_VALID_RELATIONS or relation_name not in all_concepts: continue
            relation_concept = all_concepts[relation_name]
            if not torch.is_tensor(relation_concept): continue
            num = rel_num.get(relation_name, -1); concept = torch.ones((num_objects,), device=current_device)
            try:
                if num == 0: # Unary
                    sub_objects = relation_item.get("objects", [])
                    expected_shape1=(num_objects,); expected_shape2=(num_objects,num_objects)
                    if sub_objects:
                        sub_concept = parse(scan_id, sub_objects[0], all_concepts, current_device)
                        if sub_concept is None: continue
                        if relation_concept.shape == expected_shape1: concept = relation_concept * concept
                        elif relation_concept.shape == expected_shape2: concept = (relation_concept @ sub_concept)
                        else: continue
                    else:
                         if relation_concept.shape == expected_shape1: concept = relation_concept * concept
                         elif relation_concept.shape == expected_shape2: concept = relation_concept.diag() * concept
                         else: continue
                elif num == 1: # Binary
                    sub_objects = relation_item.get("objects", [])
                    if not sub_objects: continue
                    sub_concept = parse(scan_id, sub_objects[0], all_concepts, current_device)
                    if sub_concept is None: continue
                    expected_shape = (num_objects, num_objects)
                    if relation_concept.shape == expected_shape: concept = torch.matmul(relation_concept, sub_concept)
                    else: continue
                elif num == 2: # Ternary
                    objs = relation_item.get("objects", [])
                    if len(objs) < 2: continue
                    sub1 = parse(scan_id, objs[0], all_concepts, current_device); sub2 = parse(scan_id, objs[1], all_concepts, current_device)
                    if sub1 is None or sub2 is None: continue
                    expected_shape = (num_objects, num_objects, num_objects)
                    if relation_concept.shape == expected_shape: concept = torch.einsum('ijk,j,k->i', relation_concept, sub1, sub2)
                    else: continue
                else: continue
                if relation_item.get("negative", False):
                    if torch.isfinite(concept).all(): concept = concept.max() - concept
                    else: concept = torch.zeros_like(concept)
                final_concept = final_concept * concept
            except RuntimeError as e: print(f"ERROR runtime relation '{relation_name}': {e}. Skip."); continue
    final_concept = torch.clamp(final_concept, min=0.0)
    if final_concept.sum() == 0: return torch.ones_like(final_concept) / final_concept.numel() if final_concept.numel() > 0 else None
    if not torch.isfinite(final_concept).all(): return None
    return final_concept

def prepare_features(features_this_scene, num_objects, device):
    """Filters features based on STRICT dimension matching num_objects and moves to device."""
    features_out = {}; feature_dim = -1; keys_kept = []
    if not isinstance(features_this_scene, dict): print(f"ERROR: features_this_scene not dict."); return {}, -1
    for k, v in features_this_scene.items():
        if torch.is_tensor(v):
            is_valid_dim = False; shape = v.shape
            if len(shape) > 0 and shape[0] == num_objects: # Strict check on dim 0
                if len(shape) == 1: is_valid_dim = True
                elif len(shape) >= 2 and shape[1] == num_objects: # Strict check on dim 1
                    if len(shape) == 2: is_valid_dim = True
                    elif len(shape) >= 3 and shape[2] == num_objects: is_valid_dim = True # Strict check on dim 2
            if is_valid_dim:
                if feature_dim == -1: feature_dim = shape[0]
                elif feature_dim != shape[0]: continue
                features_out[k] = v.to(device); keys_kept.append(k)
        else: features_out[k] = v; keys_kept.append(k)
    if not features_out or not any(torch.is_tensor(features_out[k]) for k in keys_kept if k in features_out): return {}, -1
    return features_out, feature_dim

def build_point_cloud(points, colors):
    """Builds an Open3D point cloud object."""
    pcd = o3d.geometry.PointCloud();
    if points is not None: pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        try: pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception as e: print(f"WARN: Failed to set colors: {e}."); pcd.paint_uniform_color([0.5, 0.5, 0.5])
    elif pcd.has_points(): pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd

def build_bboxes(obj_locs, highlight_index):
    """Builds a list of Open3D bounding box objects."""
    bbox_list = [];
    for i, bbox in enumerate(obj_locs):
        center, size = bbox[:3], np.maximum(bbox[3:], 1e-6)
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)): print(f"Warn: Invalid bbox index {i}. Skipping."); continue
        try: obb = o3d.geometry.OrientedBoundingBox(center.tolist(), np.eye(3).tolist(), size.tolist()); obb.color = (0, 1, 0) if i == highlight_index else (1, 0, 0); bbox_list.append(obb)
        except Exception as e: print(f"Error creating OBB index {i}: {e}")
    return bbox_list


# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process natural language prompts for activity suggestions in 3D scenes.")
    parser.add_argument("--scan_id", type=str, required=True, help="Scan ID (e.g., scene0025_00)")
    parser.add_argument("--sentence", type=str, required=True, help="User prompt (e.g., 'I am getting bored, what should I do now?')")
    parser.add_argument("--no_visualization", action="store_true", help="Disable visualization window")

    args = parser.parse_args()
    if not args.scan_id or not args.sentence: parser.error("Both --scan_id and --prompt are required.")

    # Setup NLTK and Gemini
    setup_nltk()
    configure_gemini()

    # Process the prompt
    results = process_scene(args.scan_id, args.sentence)
    
    if results:
        # Print the suggestion
        print("\n" + "="*30 + " SUGGESTION " + "="*30)
        print(f"Scan ID: {results['scan_id']}")
        print(f"Prompt: '{results['prompt']}'")
        print("-" * 69)
        print(f"Suggestion: {results['suggestion']}")
        print("="* 69 + "\n")
        
        # Visualize if enabled
        if not args.no_visualization and results.get('scene') is not None:
            visualize_suggestion(
                results["scan_id"],
                results["suggestion"],
                results["scene"]
            )
    else:
        print("--- Processing failed ---")

# --- END OF FILE check3.py (Modified for Activity Suggestions) ---