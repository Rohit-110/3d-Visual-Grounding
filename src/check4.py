# --- START OF FILE check3.py (Combined Relational ID & Activity Suggestion) ---

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
# from nltk.corpus import wordnet as wn # Optional
import google.generativeai as genai # Needed for suggestions
from dotenv import load_dotenv # Needed for suggestions

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
SCANREFER_JSON_PATH = "data/symbolic_exp/scanrefer.json" # For target ID lookup (in identification mode)
INSTANCE_ID_PATH_TEMPLATE = "data/referit3d/scan_data/instance_id_to_name/{scan_id}.json" # For object names (in suggestion mode)

# --- Device Setup ---
CHECK2_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARSER_DEVICE = CHECK2_DEVICE # Internal parse uses same device

# --- NLTK Setup ---
nltk.data.path.append('./nltk_data')
def download_nltk_resource(resource, download_name=None, type_path=None):
    # ... (keep original function - unchanged)
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
def setup_nltk():
    # ... (keep original function - unchanged)
    print("[INFO] Setting up NLTK...")
    download_nltk_resource('punkt', type_path='tokenizers')
    download_nltk_resource('averaged_perceptron_tagger', type_path='taggers')
    download_nltk_resource('maxent_ne_chunker', download_name='maxent_ne_chunker', type_path='chunkers')
    download_nltk_resource('words', type_path='corpora')
    # download_nltk_resource('wordnet', type_path='corpora')
    print("[INFO] NLTK setup complete.")


# --- Gemini API Setup & Synonym Function ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Load from environment
gemini_model = None
SYNONYM_TIMEOUT_SECONDS = 15
synonym_cache = {} # Simple in-memory cache for synonyms per run

def configure_gemini():
    global gemini_model
    if not GEMINI_API_KEY:
         print("WARN: GOOGLE_API_KEY not found in environment variables. Gemini API disabled.")
         return False # Indicate failure
    if gemini_model is None: # Configure only once
        try:
            print("[INFO] Configuring Gemini API...")
            genai.configure(api_key=GEMINI_API_KEY)
            # Consider trying a different model if 'gemini-1.5-flash' causes issues, e.g., 'gemini-pro'
            gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro'
            # Test connection (optional but recommended)
            # gemini_model.generate_content("test", request_options={'timeout': 5})
            print("[INFO] Gemini API configured.")
            return True # Indicate success
        except Exception as e:
            print(f"ERROR configuring Gemini API: {e}")
            gemini_model = None # Ensure it's None on failure
            return False # Indicate failure
    return True # Already configured

# def get_synonyms_gemini(noun): # Keep if needed for future enhancements
#    # ... (keep original function)
#    pass

# --- Activity Suggestion Function ---
def suggest_activity_with_object(prompt, objects_list):
    """
    Use Gemini to suggest an activity with an object from the list based on the prompt.
    """
    if gemini_model is None:
        print("ERROR: Gemini API not configured or configuration failed. Cannot suggest activities.")
        return "Sorry, I can't provide suggestions right now." # Return user-friendly message

    if not objects_list:
        print("WARN: No object names provided for activity suggestion.")
        # Fallback: Maybe ask a generic question without objects?
        objects_str = "various items"
        # Or return an error message:
        # return "Sorry, I need to know what objects are present to make a suggestion."
    else:
        objects_str = ", ".join(objects_list)

    # Create a prompt for Gemini
    # Tuned prompt for better formatting and relevance:
    suggestion_prompt = f"""
    Context: A person is in a 3D scanned room containing these objects: {objects_str}.
    User Prompt: "{prompt}"

    Task: Based *only* on the User Prompt and the available objects, suggest ONE simple, plausible activity the person could do using ONE of the listed objects.

    Output Format: Start the response with "You could...". Be concise (1-2 sentences). Focus on the action and the specific object used.
    Example: If objects are "chair, table, lamp" and prompt is "I'm tired", a good response is "You could sit down on the chair."
    Example: If objects are "sofa, tv, remote" and prompt is "I'm bored", a good response is "You could use the remote to turn on the tv."

    Generate the suggestion now:
    """

    try:
        print(f"[INFO] Querying Gemini for activity suggestion...")
        response = gemini_model.generate_content(
            suggestion_prompt,
            request_options={'timeout': SYNONYM_TIMEOUT_SECONDS}
        )
        # Basic cleaning: remove potential markdown/formatting
        suggestion = response.text.strip().replace("*", "")
        print(f"[INFO] Gemini Suggestion: {suggestion}")
        return suggestion
    except Exception as e:
        print(f"ERROR: Gemini API call failed for activity suggestion: {e}")
        # Check for specific error types if possible (e.g., Blocked prompt)
        # from google.api_core import exceptions as google_exceptions
        # if isinstance(e, google_exceptions.PermissionDenied):
        #    print("ERROR: Gemini API permission denied. Check API key and permissions.")
        # elif isinstance(e, genai.types.generation_types.BlockedPromptException): # Might need specific import
        #    print("ERROR: Prompt was blocked by Gemini safety filters.")
        # Provide a generic failure message to the user
        return "I encountered an issue while thinking of a suggestion. Please try again."


# --- Load Instance Names Function ---
def load_instance_names(scan_id):
    """Load the instance ID to name mapping for the given scan ID."""
    instance_path = INSTANCE_ID_PATH_TEMPLATE.format(scan_id=scan_id)
    instance_mapping = {}
    if not os.path.exists(instance_path):
        print(f"WARN: Instance ID mapping file not found: {instance_path}")
        return {}, [] # Return empty dict and list

    try:
        with open(instance_path, 'r') as f:
            # Assuming the JSON is {"id": "name", ...}
            instance_mapping = json.load(f)
        object_names = list(set(instance_mapping.values())) # Get unique names
        print(f"[INFO] Loaded {len(instance_mapping)} instance mappings, {len(object_names)} unique names.")
        return instance_mapping, object_names # Return both map and unique names list
    except Exception as e:
        print(f"ERROR loading instance mapping file {instance_path}: {e}")
        return {}, []

# --- Helper to extract object names from scene_data if possible ---
def get_object_names_from_scene(scene_data, id_to_name_map):
    """Extract unique object names from scene_data using the id_to_name map."""
    if not scene_data or 'obj_ids' not in scene_data or not id_to_name_map:
        return []
    names = set()
    for obj_id in scene_data['obj_ids']:
        name = id_to_name_map.get(str(obj_id)) # Ensure ID is string for lookup
        if name:
            names.add(name)
    return list(names)

# --- NLU Parsing Function (Original - Unchanged) ---
def parse_sentence_to_json_obj(sentence, known_relations):
    # ... (keep original function - unchanged)
    """Parses a sentence into a structured JSON object with categories and relations."""
    print(f"[NLU] Parsing sentence: '{sentence}'")
    words = word_tokenize(sentence.lower()); tagged_words = pos_tag(words)
    # Grammar to find Noun Phrases (NP)
    grammar = r"NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}"
    cp = RegexpParser(grammar); tree = cp.parse(tagged_words)
    noun_phrases = []
    np_subtrees = list(tree.subtrees(filter=lambda t: t.label() == 'NP'))
    word_indices = {word: i for i, (word, tag) in enumerate(tagged_words)}

    # Extract noun phrases with their head noun and position
    for subtree in np_subtrees:
        np_words = [w for w, p in subtree.leaves()]
        # Find the last noun in the phrase to act as the head
        head_noun = next((w for w, p in reversed(subtree.leaves()) if p.startswith('NN')), None)
        if head_noun:
            start_index = word_indices.get(np_words[0], -1)
            noun_phrases.append({
                'phrase': " ".join(np_words),
                'head': head_noun,
                'subtree': subtree, # Keep subtree for potential complex parsing later
                'start_index': start_index
            })

    # Sort NPs by their appearance order in the sentence
    noun_phrases.sort(key=lambda x: x['start_index'] if x['start_index'] != -1 else float('inf'))

    if not noun_phrases:
        print("[NLU] No Noun Phrases found. Falling back to simple noun extraction.")
        nouns = [w for w, p in tagged_words if p.startswith('NN')]
        if not nouns:
            print("[NLU] No nouns found in the sentence. Cannot determine target.")
            return None # Indicate failure to parse for identification
        target_category = nouns[0]
        # Use subsequent nouns as potential anchors (less reliable than NPs)
        anchor_candidates_list = [{'head': h, 'subtree': None, 'start_index': word_indices.get(h, float('inf'))} for h in nouns[1:]]
        print(f"[NLU-Fallback] Target: '{target_category}', Anchors: {[a['head'] for a in anchor_candidates_list]}")
    else:
        # Assume the first NP is the target object
        target_category = noun_phrases[0]['head']
        # The rest are potential anchor objects for relations
        anchor_candidates_list = noun_phrases[1:]
        print(f"[NLU] Target NP Head: '{target_category}'")
        print(f"[NLU] Anchor Candidates: {[a['phrase'] for a in anchor_candidates_list]}")

    # --- Check if parsing is meaningful for identification ---
    # If the target category is too generic (e.g., 'thing', 'object') and there are no relations,
    # it's likely not a specific identification query.
    is_specific_enough = True
    if target_category in ['object', 'thing', 'item', 'stuff', 'area', 'place'] and not anchor_candidates_list:
         print(f"[NLU] Target category '{target_category}' is very generic and no anchors found. Treating as non-identifying prompt.")
         is_specific_enough = False
    # Add more checks if needed (e.g., sentence length, presence of question words)

    if not is_specific_enough:
        return None # Indicate that this should be treated as a suggestion prompt

    # --- Proceed with relation extraction if parsing was specific enough ---
    json_obj = {"category": target_category, "relations": []}
    relations_found = []
    used_anchor_indices = set() # Track which anchors have been assigned to a relation
    last_relation_index = -1

    # Iterate through words to find relation keywords
    for i, (word, tag) in enumerate(tagged_words):
        potential_rel, rel_word = None, word # rel_word stores the actual text matched (e.g., "left to")

        # --- Simple Relation Keyword Matching ---
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
        elif word == 'corner': potential_rel = 'corner' # Often unary or needs context
        elif word == 'to':
            if i > 0 and tagged_words[i-1][0] == 'left': potential_rel = 'left'; rel_word = "left to"
            elif i > 0 and tagged_words[i-1][0] == 'right': potential_rel = 'right'; rel_word = "right to"
        # --- End Relation Matching ---

        if potential_rel and potential_rel not in known_relations:
            print(f"[NLU] WARN: Potential relation '{potential_rel}' (from '{rel_word}') not in known_relations {list(known_relations)}. Ignoring.")
            potential_rel = None

        if potential_rel:
            print(f"[NLU] Found potential relation '{potential_rel}' (from '{rel_word}') at index {i}")
            last_relation_index = i
            relation_struct = {"relation_name": potential_rel, "objects": []}
            anchor_found_head = None; best_anchor_cand_idx = -1; min_start_index_diff = float('inf')

            for cand_idx, anchor_cand in enumerate(anchor_candidates_list):
                 if cand_idx in used_anchor_indices: continue
                 anchor_start_idx = anchor_cand['start_index']
                 if anchor_start_idx > i:
                     dist = anchor_start_idx - i
                     if dist < min_start_index_diff:
                         min_start_index_diff = dist; anchor_found_head = anchor_cand['head']
                         best_anchor_cand_idx = cand_idx

            if anchor_found_head:
                print(f"[NLU]   -> Linking with anchor: '{anchor_found_head}'")
                relation_struct["objects"].append({"category": anchor_found_head, "relations": []}) # Assume simple anchors
                if best_anchor_cand_idx != -1 : used_anchor_indices.add(best_anchor_cand_idx)

                if potential_rel == 'between' and best_anchor_cand_idx != -1:
                    # ... (logic for finding second anchor for 'between' - unchanged)
                    print(f"[NLU]   -> Relation '{potential_rel}' needs a second anchor.")
                    second_anchor_found_head = None; best_second_cand_idx = -1; min_start_index_diff2 = float('inf'); first_anchor_start = anchor_candidates_list[best_anchor_cand_idx]['start_index']
                    for cand2_idx, anchor2_cand in enumerate(anchor_candidates_list):
                         if cand2_idx in used_anchor_indices: continue
                         anchor2_start_idx = anchor2_cand['start_index']
                         if anchor2_start_idx > first_anchor_start: # Must be after the first anchor
                            dist2 = anchor2_start_idx - first_anchor_start
                            if dist2 < min_start_index_diff2: min_start_index_diff2 = dist2; second_anchor_found_head = anchor2_cand['head']; best_second_cand_idx = cand2_idx
                    if second_anchor_found_head:
                         print(f"[NLU]   -> Linking with second anchor: '{second_anchor_found_head}'")
                         relation_struct["objects"].append({"category": second_anchor_found_head, "relations": []})
                         if best_second_cand_idx != -1: used_anchor_indices.add(best_second_cand_idx)
                    else: print(f"[NLU] WARN: Could not find a second anchor for 'between'. Relation may be incomplete.")


            if i > 0 and tagged_words[i-1][0] == 'not':
                print(f"[NLU]   -> Detected negation for relation '{potential_rel}'")
                relation_struct["negative"] = True

            is_unary = rel_num.get(potential_rel, -1) == 0
            if relation_struct["objects"] or is_unary:
                relations_found.append(relation_struct)
            elif not is_unary:
                print(f"[NLU] WARN: Relation '{potential_rel}' requires anchor(s) but none found/linked. Ignoring relation.")


    json_obj["relations"] = relations_found
    # If no relations were found, but we had a specific target, it's still identification
    print(f"[NLU] Parsed json_obj for identification: {json.dumps(json_obj, indent=2)}")
    return json_obj

# --- Core Parsing Logic (Uses Features - Original, Unchanged) ---
def parse(scan_id, json_obj, all_concepts, current_device):
    # ... (keep original function - unchanged, including internal prints)
    """Recursively parses the JSON object and computes object scores based on features."""
    category = json_obj.get("category", "object") # Default to generic object if missing
    if category not in all_concepts:
        print(f"[Parse] ERROR: Category '{category}' not found in features.")
        # Consider synonym fallback here if desired
        # syns = get_synonyms_gemini(category) or get_synonyms_wordnet(category)
        # for syn in syns: if syn in all_concepts: category = syn; print(f"[Parse] Using synonym '{syn}'"); break
        # else: return None # If no synonym found either
        return None # Strict: category must exist
    if not torch.is_tensor(all_concepts[category]):
        print(f"[Parse] ERROR: Concept for '{category}' is not a tensor.")
        return None
    appearance_concept = all_concepts[category]

    if appearance_concept.dim() == 0 or appearance_concept.shape[0] == 0:
        print(f"[Parse] ERROR: Concept tensor for '{category}' is empty or scalar.")
        return None

    num_objects = appearance_concept.shape[0]
    if category in ["corner", "middle", "room", "center"] and category in all_concepts:
         print(f"[Parse] Using concept directly for spatial term: '{category}'")
         return all_concepts[category].clone().to(current_device) # Should be (num_objects,)

    final_concept = appearance_concept.clone().to(current_device).float()
    # print(f"[Parse] Initializing final_concept for '{category}' with shape {final_concept.shape}") # Verbose

    if "relations" in json_obj:
        for relation_item in json_obj["relations"]:
            if "anchors" in relation_item and "objects" not in relation_item: relation_item["objects"] = relation_item["anchors"]
            relation_name = relation_item.get("relation_name")
            if not relation_name or relation_name not in ALL_VALID_RELATIONS: continue
            if relation_name not in all_concepts: print(f"[Parse] WARN: Relation '{relation_name}' not found in features. Skipping."); continue
            relation_concept = all_concepts[relation_name]
            if not torch.is_tensor(relation_concept): print(f"[Parse] WARN: Relation concept '{relation_name}' not tensor. Skipping."); continue

            relation_concept = relation_concept.to(current_device).float()
            num = rel_num.get(relation_name, -1)
            concept_update = torch.ones((num_objects,), device=current_device).float()
            # print(f"[Parse] Processing relation: '{relation_name}' (Arity: {num})") # Verbose

            try:
                if num == 0: # Unary
                    expected_shape = (num_objects,)
                    if relation_concept.shape == expected_shape: concept_update = relation_concept
                    elif relation_concept.shape == (num_objects, num_objects): concept_update = relation_concept.diag()
                    elif relation_item.get("objects"): # Unary relation *with* an object (less common)
                         sub_obj = relation_item["objects"][0]
                         sub_concept = parse(scan_id, sub_obj, all_concepts, current_device)
                         if sub_concept is not None and relation_concept.shape == (num_objects, num_objects):
                            concept_update = torch.matmul(relation_concept, sub_concept)
                         else: continue # Skip if sub-parsing fails or shape mismatch
                    else: continue # Skip if shape mismatch

                elif num == 1: # Binary
                    sub_objects = relation_item.get("objects", [])
                    if not sub_objects: continue
                    anchor_obj_json = sub_objects[0]
                    anchor_concept = parse(scan_id, anchor_obj_json, all_concepts, current_device)
                    if anchor_concept is None: continue
                    expected_shape = (num_objects, num_objects)
                    if relation_concept.shape == expected_shape: concept_update = torch.matmul(relation_concept, anchor_concept)
                    else: continue

                elif num == 2: # Ternary
                    objs = relation_item.get("objects", [])
                    if len(objs) < 2: continue
                    anchor1_json, anchor2_json = objs[0], objs[1]
                    anchor1_concept = parse(scan_id, anchor1_json, all_concepts, current_device)
                    anchor2_concept = parse(scan_id, anchor2_json, all_concepts, current_device)
                    if anchor1_concept is None or anchor2_concept is None: continue
                    expected_shape = (num_objects, num_objects, num_objects)
                    if relation_concept.shape == expected_shape: concept_update = torch.einsum('ijk,j,k->i', relation_concept, anchor1_concept, anchor2_concept)
                    else: continue
                else: continue # Skip unknown arity

                if relation_item.get("negative", False):
                    if torch.isfinite(concept_update).all() and concept_update.numel() > 0:
                         max_val = concept_update.max()
                         if max_val > 0 : concept_update = max_val - concept_update
                         else: concept_update = torch.ones_like(concept_update) # Fallback if all scores were <= 0
                    else: concept_update = torch.zeros_like(concept_update)

                if final_concept.shape == concept_update.shape:
                    concept_update = torch.clamp(concept_update, min=0.0)
                    final_concept = final_concept * concept_update
                    # print(f"[Parse]   Updated final_concept using '{relation_name}'. Sum: {final_concept.sum():.4f}") # Verbose
                # else: print(f"[Parse] WARN: Shape mismatch final {final_concept.shape} vs update {concept_update.shape}") # Verbose


            except RuntimeError as e: print(f"ERROR: Runtime error processing relation '{relation_name}': {e}. Skipping relation."); continue

    final_concept = torch.clamp(final_concept, min=0.0)
    if final_concept.sum() == 0:
        print(f"[Parse] WARN: All object scores zero for '{category}'. Returning uniform.")
        return torch.ones_like(final_concept) / final_concept.numel() if final_concept.numel() > 0 else None
    if not torch.isfinite(final_concept).all():
        print(f"[Parse] ERROR: Non-finite values in final scores for '{category}'. Returning None.")
        return None # Indicate failure

    # Normalize scores
    if final_concept.sum() > 0: final_concept = final_concept / final_concept.sum()
    else: return torch.ones_like(final_concept) / final_concept.numel() if final_concept.numel() > 0 else None # Should be caught earlier

    # print(f"[Parse] Final scores calculated for '{category}'. Max score: {final_concept.max():.4f}") # Verbose
    return final_concept


# --- Feature Preparation (Original, Unchanged) ---
def prepare_features(features_this_scene, num_objects, device):
    # ... (keep original function - unchanged)
    """Filters features based on dimension matching num_objects and moves to device."""
    features_out = {}
    feature_dim = -1 # Store the expected dimension size (num_objects)
    keys_kept = []

    if not isinstance(features_this_scene, dict):
        print(f"ERROR: features_this_scene is not a dictionary.")
        return {}, -1

    # print(f"[Features] Preparing features for {num_objects} objects...") # Verbose
    for k, v in features_this_scene.items():
        if torch.is_tensor(v):
            shape = v.shape; is_valid_dim = False
            if len(shape) > 0:
                # Check dimensions based on relation arity or assume appearance vector
                arity = rel_num.get(k, -1) # -1 indicates not a known relation -> assume appearance
                if arity == -1: # Appearance/Category vector
                    if len(shape) == 1 and shape[0] == num_objects: is_valid_dim = True
                elif arity == 0: # Unary
                    if (len(shape) == 1 and shape[0] == num_objects) or \
                       (len(shape) == 2 and shape[0] == num_objects and shape[1] == num_objects): is_valid_dim = True
                elif arity == 1: # Binary
                    if len(shape) == 2 and shape[0] == num_objects and shape[1] == num_objects: is_valid_dim = True
                elif arity == 2: # Ternary
                     if len(shape) == 3 and shape[0] == num_objects and shape[1] == num_objects and shape[2] == num_objects: is_valid_dim = True

            if is_valid_dim:
                if feature_dim == -1: feature_dim = num_objects
                features_out[k] = v.to(device); keys_kept.append(k)
                # print(f"[Features] Kept feature '{k}' with shape {shape}") # Verbose
            # else: print(f"[Features] Skipping feature '{k}' shape {shape} (N={num_objects})") # Verbose

        elif isinstance(v, (int, float, str, list, dict)): # Keep non-tensor metadata
            features_out[k] = v; keys_kept.append(k)

    if not features_out or not any(torch.is_tensor(features_out[k]) for k in keys_kept if k in features_out and isinstance(features_out[k], torch.Tensor)):
        print(f"ERROR: No valid tensor features found matching {num_objects} objects.")
        return {}, -1

    print(f"[Features] Prepared {len(keys_kept)} features for {num_objects} objects.")
    return features_out, num_objects


# --- Geometry Building Functions (Originals, Unchanged) ---
def build_point_cloud(points, colors):
    # ... (keep original function - unchanged)
    """Builds an Open3D point cloud object."""
    pcd = o3d.geometry.PointCloud()
    if points is not None and len(points) > 0:
         try: pcd.points = o3d.utility.Vector3dVector(points)
         except Exception as e: print(f"ERROR creating points Vector3dVector: {e}"); return None
    else: print("WARN: No points provided for point cloud."); return None

    if colors is not None and len(colors) == len(points):
        try:
            if colors.max() > 1.0: colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception as e:
            print(f"WARN: Failed to set colors: {e}. Using uniform."); pcd.paint_uniform_color([0.5, 0.5, 0.5])
    elif pcd.has_points(): pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd

def build_bboxes(obj_locs, highlight_index=-1, object_ids=None, id_to_name_map=None):
    # ... (keep original function - unchanged)
    """Builds a list of Open3D bounding box objects, highlighting one."""
    bbox_list = []
    if obj_locs is None or len(obj_locs) == 0: return bbox_list
    for i, bbox in enumerate(obj_locs):
        if isinstance(bbox, torch.Tensor): bbox = bbox.cpu().numpy()
        if bbox is None or len(bbox) < 6: continue
        center, size = bbox[:3], bbox[3:6]; size = np.maximum(size, 1e-4)
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)): continue
        try:
            obb = o3d.geometry.OrientedBoundingBox(center.tolist(), np.eye(3).tolist(), size.tolist())
            obb.color = (0, 1, 0) if i == highlight_index else (1, 0, 0) # Green=Highlight, Red=Other
            bbox_list.append(obb)
        except Exception as e: print(f"ERROR creating OBB index {i}: {e}")
    return bbox_list


# --- Visualization for Identification Result ---
def visualize_result(scan_id, points, colors, obj_locs, highlight_index, sentence=""):
    # ... (keep original function - unchanged)
    """Visualize the scene point cloud and bounding boxes, highlighting the identified object."""
    print(f"\n--- Visualizing Identification Result: {scan_id} ---")
    print(f"Highlighting object index: {highlight_index}")
    if sentence: print(f"Query: '{sentence}'")
    pcd = build_point_cloud(points, colors)
    bbox_list = build_bboxes(obj_locs, highlight_index=highlight_index)
    geometries = []
    if pcd is not None: geometries.append(pcd)
    if bbox_list: geometries.extend(bbox_list)
    if geometries:
        window_title = f"Identified: Idx {highlight_index} | Scan: {scan_id} | Query: {sentence[:40]}..."
        try: o3d.visualization.draw_geometries(geometries, window_name=window_title)
        except Exception as e: print(f"ERROR during Open3D visualization: {e}")
    else: print("Error: No geometries to visualize.")


# --- Visualization for Activity Suggestion ---
def visualize_suggestion(scan_id, suggestion, points, colors, scene_data, id_to_name_map):
    """Visualize the scene, highlighting object(s) mentioned in the suggestion."""
    print(f"\n--- Visualizing Activity Suggestion: {scan_id} ---")
    print(f"Suggestion: {suggestion}")

    target_object_name = None
    highlight_indices = []

    # Attempt to extract object name from suggestion (simple regex)
    # Looks for patterns like "the [object]", "a [object]", "your [object]"
    match = re.search(r"(?:the|a|an|your)\s+([\w\s]+)", suggestion.lower())
    if match:
        # Extract potential object name(s) - handle multi-word names if needed
        potential_name = match.group(1).strip()
        # More robust: check against known object names
        object_names_in_scene = list(id_to_name_map.values())
        found_obj = None
        # Prioritize exact match, then substring match
        for obj_name in object_names_in_scene:
             if potential_name == obj_name.lower():
                  found_obj = obj_name
                  break
        if not found_obj:
             for obj_name in object_names_in_scene:
                  if potential_name in obj_name.lower():
                       found_obj = obj_name
                       # Don't break here, maybe a more specific match exists
                       # This might highlight multiple objects if names overlap (e.g., "chair", "armchair")
        target_object_name = found_obj
        print(f"Extracted target object from suggestion: '{target_object_name}' (from '{potential_name}')")

    if target_object_name and scene_data and 'obj_ids' in scene_data and 'pred_locs' in scene_data:
        obj_locs = scene_data['pred_locs']
        obj_ids = scene_data['obj_ids']
        # Find indices of objects matching the name
        for i, obj_id in enumerate(obj_ids):
            current_name = id_to_name_map.get(str(obj_id))
            if current_name and target_object_name.lower() == current_name.lower():
                 # Check if index is valid for obj_locs
                 if i < len(obj_locs):
                    highlight_indices.append(i)
                 else:
                    print(f"WARN: Found matching ID {obj_id} but index {i} is out of bounds for pred_locs ({len(obj_locs)}).")

        if not highlight_indices:
             print(f"WARN: Found target name '{target_object_name}' but couldn't map it to a valid object index/bbox.")
        else:
             print(f"Highlighting object indices for suggestion: {highlight_indices}")
    elif not target_object_name:
         print("WARN: Could not extract a known object name from the suggestion. No highlight.")
    else:
         print("WARN: Scene data or ID map missing, cannot highlight object for suggestion.")


    # Build geometries
    pcd = build_point_cloud(points, colors)
    # Highlight all found indices (usually one, but could be multiple if names overlap)
    # For simplicity, just pass the first highlighted index to build_bboxes for *coloring*,
    # but ideally, build_bboxes could take a list of indices to highlight.
    # Let's modify build_bboxes slightly for this.

    # --- Small modification to build_bboxes signature/logic ---
    def build_bboxes_multi_highlight(obj_locs, highlight_indices_list=[]):
        bbox_list = []
        if obj_locs is None or len(obj_locs) == 0: return bbox_list
        highlight_set = set(highlight_indices_list) # Faster lookup
        for i, bbox in enumerate(obj_locs):
            if isinstance(bbox, torch.Tensor): bbox = bbox.cpu().numpy()
            if bbox is None or len(bbox) < 6: continue
            center, size = bbox[:3], bbox[3:6]; size = np.maximum(size, 1e-4)
            if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)): continue
            try:
                obb = o3d.geometry.OrientedBoundingBox(center.tolist(), np.eye(3).tolist(), size.tolist())
                obb.color = (0, 1, 0) if i in highlight_set else (1, 0, 0) # Green if in highlight set
                bbox_list.append(obb)
            except Exception as e: print(f"ERROR creating OBB index {i}: {e}")
        return bbox_list
    # --- Use the modified version ---
    obj_locs_vis = scene_data.get('pred_locs') if scene_data else None
    bbox_list = build_bboxes_multi_highlight(obj_locs_vis, highlight_indices)


    # Visualize
    geometries = []
    if pcd is not None: geometries.append(pcd)
    if bbox_list: geometries.extend(bbox_list)

    if geometries:
        window_title = f"Suggestion | Scan: {scan_id} | Target: {target_object_name or 'None'}"
        try: o3d.visualization.draw_geometries(geometries, window_name=window_title)
        except Exception as e: print(f"ERROR during Open3D visualization: {e}")
    else: print("Error: No geometries to visualize for suggestion.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process natural language input for object identification OR activity suggestion in 3D scenes.")
    parser.add_argument("--scan_id", type=str, required=True, help="Scan ID (e.g., scene0025_00)")
    parser.add_argument("--sentence", type=str, required=True, help="Natural language query or prompt (e.g., 'sofa left of the door' OR 'I am bored')")
    parser.add_argument("--features_path", type=str, default=DEFAULT_FEATURES_PATH, help="Path to pre-computed scene features (.pth) (Needed for identification)")
    parser.add_argument("--no_visualization", action="store_true", help="Disable the Open3D visualization window")
    parser.add_argument("--raw_pcd_path_template", type=str, default=RAW_PCD_PATH_TEMPLATE, help="Template for raw point cloud paths")
    # parser.add_argument("--scanrefer_json", type=str, default=SCANREFER_JSON_PATH, help="Path to ScanRefer JSON for GT lookup (optional, identification only)") # Keep if GT check is desired

    args = parser.parse_args()
    if not args.scan_id or not args.sentence:
        parser.error("Both --scan_id and --sentence are required.")

    start_time = time.time()

    # --- Setup ---
    setup_nltk()
    # Gemini setup will happen only if needed (in the suggestion branch)

    # --- Load Common Data (Needed for both paths) ---
    print(f"\n--- Loading Scene Data and Geometry for {args.scan_id} ---")
    scene_data = None
    obj_locs = None
    obj_ids = None
    num_objects = 0
    points_raw, colors_raw = None, None
    id_to_name_map, object_names_list = {}, [] # For suggestions

    try:
        # Load scene structure (bboxes, ids)
        _, scene_data = load_one_scene(args.scan_id, label_type="pred") # Using predicted bboxes/labels
        if not scene_data: raise ValueError("load_one_scene returned empty data.")
        obj_locs = scene_data.get("pred_locs")
        obj_ids = scene_data.get("obj_ids")
        if obj_locs is None or obj_ids is None: raise ValueError("Scene data missing 'pred_locs' or 'obj_ids'.")
        num_objects = len(obj_ids)
        if num_objects == 0: print(f"WARN: No objects found in scene data for {args.scan_id}. Identification/Suggestion might fail.")

        # Load raw point cloud
        pcd_path = args.raw_pcd_path_template.format(scan_id=args.scan_id)
        if os.path.exists(pcd_path):
            pcd_data = torch.load(pcd_path, map_location="cpu") # Assume tuple format for now
            if isinstance(pcd_data, tuple) and len(pcd_data) >= 2:
                points_raw, colors_raw = pcd_data[0], pcd_data[1]
                if isinstance(points_raw, torch.Tensor): points_raw = points_raw.numpy()
                if isinstance(colors_raw, torch.Tensor): colors_raw = colors_raw.numpy()
                if colors_raw is not None and np.max(colors_raw) > 1.0: colors_raw = colors_raw / 255.0
            else: print(f"WARN: Unexpected format in {pcd_path}. Expected tuple.")
        else: print(f"WARN: Raw PCD file not found at {pcd_path}.")
        # Fallback geometry if raw pcd failed
        if points_raw is None and "room_points" in scene_data:
            print("INFO: Using 'room_points' from scene data as geometry.")
            points_raw = scene_data["room_points"]
            colors_raw = np.ones_like(points_raw) * 0.5 if points_raw is not None else None

        # Load instance names (useful for suggestions, optional for ID visualization)
        id_to_name_map, object_names_list = load_instance_names(args.scan_id)
        if not object_names_list:
             # Try extracting from scene_data if instance file failed/missing
             object_names_list = get_object_names_from_scene(scene_data, id_to_name_map)


    except Exception as e:
        print(f"ERROR loading common scene data/geometry for {args.scan_id}: {e}")
        print("Proceeding without geometry/full scene info might limit functionality.")
        # Allow continuing, maybe only suggestion works if Gemini is up.
        # sys.exit(1) # Or exit if scene data is critical

    # --- Decide Pipeline: Identification or Suggestion ---
    print(f"\n--- Analyzing Input: '{args.sentence}' ---")
    json_obj = parse_sentence_to_json_obj(args.sentence, ALL_VALID_RELATIONS)

    if json_obj is not None and num_objects > 0:
        # --- Identification Pipeline ---
        print("\n>>> Attempting Object Identification <<<")

        # 1. Load Features (Specific to Identification)
        print(f"\n--- Loading Features from {args.features_path} ---")
        features_this_scene = None
        if not os.path.exists(args.features_path):
            print(f"ERROR: Features file not found at {args.features_path}. Cannot perform identification.")
            sys.exit(1) # Features are essential here
        try:
            all_features = torch.load(args.features_path, map_location='cpu')
            features_this_scene = all_features.get(args.scan_id)
            if features_this_scene is None:
                print(f"ERROR: Features for scan_id '{args.scan_id}' not found in features file.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR loading features file: {e}"); sys.exit(1)

        # 2. Prepare Features
        print(f"\n--- Preparing Features ---")
        features_out, feat_dim_check = prepare_features(features_this_scene, num_objects, PARSER_DEVICE)
        if feat_dim_check == -1:
            print(f"ERROR: Feature preparation failed. Cannot perform identification.")
            sys.exit(1)

        # 3. Run Core Parsing/Scoring
        print(f"\n--- Calculating Object Scores ---")
        scores = parse(args.scan_id, json_obj, features_out, PARSER_DEVICE)

        if scores is None:
            print("ERROR: Failed to calculate object scores (parse returned None). Identification failed.")
        elif not torch.is_tensor(scores) or scores.shape[0] != num_objects:
            print(f"ERROR: Calculated scores are invalid. Expected ({num_objects},), got {type(scores)} shape {scores.shape if torch.is_tensor(scores) else 'N/A'}.")
        else:
            # 4. Get Result
            highlight_index = torch.argmax(scores).item()
            max_score = scores[highlight_index].item()
            print(f"\n--- Identification Result ---")
            print(f"Identified Object Index: {highlight_index} (Score: {max_score:.4f})")
            # Optional: Add GT comparison here if needed

            # 5. Visualize Identification
            if not args.no_visualization:
                visualize_result(
                    args.scan_id,
                    points_raw, colors_raw,
                    obj_locs,
                    highlight_index,
                    args.sentence
                )

    else:
        # --- Activity Suggestion Pipeline ---
        if num_objects == 0 and not json_obj:
             print("\n>>> Input appears vague, but no objects found in scene. Cannot provide suggestion. <<<")
        elif json_obj is None:
             print("\n>>> Input not parsed for specific object. Attempting Activity Suggestion <<<")
        else: # json_obj was not None, but num_objects was 0
             print("\n>>> Input parsed for object, but no objects found in scene data. Cannot perform identification. <<<")
             # Optionally, could still *try* suggestion if desired, but context is limited.
             # For now, we'll just end if num_objects is 0 unless it was clearly vague.
             if json_obj is None and num_objects == 0: # Only proceed if it was vague AND no objects
                 pass # Fall through to suggestion attempt below
             else:
                  print("Cannot proceed.")
                  sys.exit(1)


        # 1. Configure Gemini
        print("\n--- Setting up Suggestion Engine ---")
        if not configure_gemini():
            print("ERROR: Gemini API could not be configured. Cannot provide suggestion.")
        else:
            # 2. Get Object List for Prompt
            if not object_names_list:
                print("WARN: No object names available for context in suggestion prompt.")
                # Proceeding without object list in prompt

            # 3. Generate Suggestion
            suggestion = suggest_activity_with_object(args.sentence, object_names_list)

            # 4. Print Suggestion
            print(f"\n--- Activity Suggestion ---")
            print(f"Prompt: '{args.sentence}'")
            print(f"Suggestion: {suggestion}")

            # 5. Visualize Suggestion (Highlight suggested object)
            if not args.no_visualization and suggestion and "can't provide suggestions" not in suggestion and "encountered an issue" not in suggestion:
                visualize_suggestion(
                    args.scan_id,
                    suggestion,
                    points_raw, colors_raw,
                    scene_data, # Pass the loaded scene data
                    id_to_name_map # Pass the ID-to-name mapping
                )

    end_time = time.time()
    print(f"\nTotal Processing Time: {end_time - start_time:.2f} seconds")


# --- END OF FILE check3.py (Combined) ---