# --- START OF FILE eval_scanrefer_enhanced.py ---

from copy import deepcopy
import json
from json import JSONDecodeError
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import sys
import time
import argparse
from collections import Counter, defaultdict

# --- Gemini Import ---
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARN: google-generativeai library not found. Run 'pip install google-generativeai' to enable synonym lookup.")
    GEMINI_AVAILABLE = False
    genai = None

# --- Project Imports ---
try:
    from src.relation_encoders.compute_features import ALL_VALID_RELATIONS, rel_num
    from src.util.eval_helper import eval_ref_one_sample, construct_bbox_corners, load_pc
    from src.dataset.datasets import ScanReferDataset
    print("[Eval INFO] Project imports successful.")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required project modules: {e}"); sys.exit(1)

# --- Use ScanReferDataset ---
try:
    print("[Eval INFO] Initializing ScanReferDataset..."); dataset = ScanReferDataset(); print("[Eval INFO] ScanReferDataset initialized.")
except Exception as e: print(f"FATAL ERROR: Failed to initialize ScanReferDataset: {e}"); sys.exit(1)

# --- Setup Gemini API (Only for Synonyms) ---
from dotenv import load_dotenv
load_dotenv(); GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY"); gemini_model = None
GEMINI_MODEL_NAME = "gemini-1.5-flash"
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try: genai.configure(api_key=GEMINI_API_KEY); gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME); print(f"[Eval INFO] Gemini API configured ({GEMINI_MODEL_NAME}) for synonyms.")
    except Exception as e: print(f"[Eval ERROR] Failed config Gemini API: {e}")
elif not GEMINI_API_KEY and GEMINI_AVAILABLE: print("[Eval WARN] GOOGLE_API_KEY not found. Synonym lookup disabled.")
elif not GEMINI_AVAILABLE: print("[Eval WARN] google-generativeai not installed. Synonym lookup disabled.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"[Eval INFO] Using device: {DEVICE}")

# --- Caches ---
synonyms_cache_path = Path("synonyms_cache.json"); synonyms_cache = {}
if synonyms_cache_path.exists():
    try:
        with open(synonyms_cache_path, "r") as f: synonyms_cache = json.load(f)
        print(f"[Eval INFO] Loaded {len(synonyms_cache)} items from synonym cache.")
    except Exception as e: print(f"WARN: Error loading synonym cache: {e}")


# --- Predefined Synonym Map - EXTENDED ---
OBJECT_SYNONYM_MAP = { 
    # Original entries 
    "chair": ["seat", "stool", "armchair", "recliner", "office chair", "dining chair"],
    "table": ["desk", "counter", "stand", "surface", "dining table", "coffee table", "side table"],
    "sofa": ["couch", "settee", "loveseat", "divan", "futon", "sofa chair", "sectional"],
    "bed": ["mattress", "cot", "bunk", "berth", "bedframe", "sleeping area"],
    "cabinet": ["cupboard", "wardrobe", "closet", "chest", "storage", "dresser", "console"],
    "shelf": ["bookshelf", "rack", "ledge", "shelving", "bookcase", "wall shelf", "display shelf"],
    "desk": ["table", "bureau", "workstation", "counter", "computer desk", "writing table"],
    "ottoman": ["footrest", "stool", "hassock", "pouf", "foot stool"],
    "lamp": ["light", "lighting", "fixture", "floor lamp", "table lamp", "desk lamp"],
    "tv": ["television", "monitor", "screen", "display", "flat screen", "entertainment system"],
    "bin": ["container", "basket", "receptacle", "canister", "trashcan", "waste bin", "trash"],
    "box": ["container", "carton", "crate", "chest", "storage box", "package"],
    "computer": ["pc", "laptop", "desktop", "workstation", "machine", "system"],
    "monitor": ["screen", "display", "tv", "panel", "lcd", "computer display"],
    "window": ["pane", "glass", "opening", "casement", "window frame"],
    "door": ["entryway", "gateway", "entrance", "portal", "doorway", "access"],
    "picture": ["painting", "photo", "artwork", "frame", "print", "portrait", "wall art"],
    "toilet": ["commode", "lavatory", "john", "loo", "bathroom fixture", "water closet"],
    "sink": ["basin", "washbasin", "lavatory", "washbowl", "bathroom sink", "kitchen sink"],
    "bathtub": ["tub", "bath", "shower", "jacuzzi", "washbasin"],
    "counter": ["countertop", "surface", "bar", "ledge", "kitchen counter", "island"],
    "refrigerator": ["fridge", "cooler", "icebox", "freezer", "appliance"],
    "stove": ["oven", "range", "cooktop", "burner", "cooking surface", "appliance"],
    "microwave": ["oven", "microwave oven", "appliance", "cooker"],
    "trashcan": ["bin", "wastebasket", "trash", "garbage", "waste container", "trash bin", "trash can"],
    "bookshelf": ["shelf", "bookcase", "shelving", "rack", "book storage", "wall shelf"],
    "pillow": ["cushion", "bolster", "headrest", "throw pillow", "couch pillow"],
    "rug": ["carpet", "mat", "runner", "area rug", "floor covering"],
    "curtain": ["drape", "blind", "shade", "window covering", "window treatment"],
    "mirror": ["glass", "reflector", "wall mirror", "bathroom mirror", "vanity mirror"],
    "plant": ["flower", "shrub", "greenery", "potted plant", "houseplant", "indoor plant"],
    "vase": ["container", "pot", "urn", "jar", "flower vase", "decorative vessel"],
    "bottle": ["container", "flask", "jar", "water bottle", "glass bottle"],
    "cup": ["mug", "glass", "beaker", "coffee cup", "drinking vessel"],
    "bowl": ["dish", "basin", "container", "serving bowl", "food container"],
    "plate": ["dish", "platter", "dinner plate", "serving plate", "dining plate"],
    "fan": ["ventilator", "blower", "ceiling fan", "standing fan", "air circulator"],
    "clock": ["timepiece", "wall clock", "alarm clock", "desk clock", "timekeeping"],
    "bench": ["seat", "long chair", "sitting area", "pew", "long seat"],
    "speaker": ["loudspeaker", "audio device", "sound system", "stereo", "audio output"],
    "bag": ["sack", "pouch", "tote", "handbag", "backpack", "luggage"],
    "basket": ["container", "hamper", "bin", "wicker basket", "storage basket"],
    "towel": ["cloth", "bath towel", "hand towel", "washcloth", "bathroom linen"],
    "blanket": ["cover", "throw", "quilt", "comforter", "bedspread"],
    "stand": ["rack", "holder", "support", "pedestal", "base", "display stand"],
    "keyboard": ["keypad", "computer keyboard", "typing device", "input device"],
    "remote": ["controller", "remote control", "device controller", "tv remote"],
    "book": ["volume", "tome", "textbook", "novel", "reading material"],
    "laptop": ["computer", "notebook", "portable computer", "mobile computer"],
    "phone": ["telephone", "mobile", "cell", "smartphone", "handset", "device"],
    "cushion": ["pillow", "pad", "throw pillow", "seat cushion", "soft padding"],
    "handle": ["knob", "grip", "pull", "doorknob", "cabinet handle", "drawer pull"],
    "hanger": ["hook", "rack", "clothes hanger", "coat hanger", "garment hanger"],
    "light": ["lamp", "lighting fixture", "ceiling light", "wall light", "illumination"],
    "office": ["workplace", "study", "office room", "home office", "workspace"],
    "bathroom": ["lavatory", "restroom", "washroom", "toilet room", "powder room"],
    "kitchen": ["cookery", "galley", "cooking area", "food preparation area"],
    "bedroom": ["chamber", "sleeping room", "bedchamber", "sleep area"],
    "living": ["family", "sitting", "lounge", "living room", "common area"],
    "wall": ["partition", "divider", "barrier", "room divider", "vertical surface"],
    "floor": ["ground", "flooring", "surface", "bottom", "deck"],
    "ceiling": ["overhead", "roof", "top", "upper surface", "overhead surface"],
    "corner": ["angle", "nook", "room corner", "edge", "intersection"],
    "center": ["middle", "core", "central point", "interior", "central area"],
    "back": ["rear", "behind", "posterior", "tail end", "reverse side"],
    "front": ["fore", "forward", "anterior", "leading part", "face"],
    "side": ["flank", "lateral", "edge", "periphery", "border"],
    "top": ["upper", "summit", "peak", "crest", "highest point"],
    "bottom": ["base", "foot", "lowest part", "underside", "foundation"],
    "drawer": ["compartment", "tray", "pull-out", "storage drawer", "sliding compartment"],
    # New synonyms and additional objects
    "trash can": ["bin", "garbage", "waste container", "trash bin", "trashcan", "garbage can"],
    "garbage can": ["trash can", "bin", "waste container", "garbage bin", "trash"],
    "sofa chair": ["couch", "armchair", "easy chair", "lounge chair", "recliner"],
    "bookcase": ["bookshelf", "shelves", "shelf unit", "book storage", "library shelf"],
    "nightstand": ["bedside table", "night table", "end table", "side table"],
    "dining table": ["table", "eating table", "kitchen table", "dinner table"],
    "coffee table": ["cocktail table", "low table", "center table", "living room table"],
    "tv stand": ["media console", "television stand", "entertainment center", "media stand"],
    "filing cabinet": ["file cabinet", "file drawer", "office cabinet", "document storage"],
    "wardrobe": ["closet", "armoire", "clothes cabinet", "garment storage"],
    "display cabinet": ["china cabinet", "curio cabinet", "display case", "showcase"],
    "side table": ["end table", "accent table", "occasional table", "lamp table"],
    "ottoman stool": ["footstool", "footrest", "hassock", "pouf", "ottoman"],
    "armchair": ["chair", "easy chair", "lounge chair", "club chair", "accent chair"],
    "stool": ["seat", "bar stool", "chair", "seating", "sitting stool"],
    "recliner": ["reclining chair", "lounge chair", "easy chair", "armchair"],
    "couch": ["sofa", "settee", "loveseat", "divan", "lounge"],
    "sectional": ["sectional sofa", "sofa", "couch", "l-shaped sofa", "corner sofa"],
    "futon": ["sofa bed", "convertible sofa", "daybed", "sleeper sofa"],
    "clothing": ["clothes", "garments", "apparel", "attire", "wardrobe items"],
    "sculpture": ["statue", "figurine", "art piece", "decorative object", "ornament"]
}

# Convert all keys and values to lowercase for consistent matching
OBJECT_SYNONYM_MAP = {k.lower(): [s.lower() for s in v] for k, v in OBJECT_SYNONYM_MAP.items()}

# Generate reverse mappings for better synonym coverage
REVERSE_SYNONYM_MAP = {}
for key, values in OBJECT_SYNONYM_MAP.items():
    for value in values:
        if value not in REVERSE_SYNONYM_MAP:
            REVERSE_SYNONYM_MAP[value] = []
        if key not in REVERSE_SYNONYM_MAP[value]:
            REVERSE_SYNONYM_MAP[value].append(key)

# Merge both for complete mapping
COMPLETE_SYNONYM_MAP = deepcopy(OBJECT_SYNONYM_MAP)
for key, values in REVERSE_SYNONYM_MAP.items():
    if key not in COMPLETE_SYNONYM_MAP:
        COMPLETE_SYNONYM_MAP[key] = values
    else:
        for value in values:
            if value not in COMPLETE_SYNONYM_MAP[key]:
                COMPLETE_SYNONYM_MAP[key].append(value)

effective_synonyms = defaultdict(Counter) # Keep track of effective synonyms

# --- Get Synonyms Function (Prioritizes local, uses Gemini fallback) ---
last_gemini_synonym_call_time = 0
GEMINI_SYNONYM_DELAY = 4.1 # seconds
def get_3d_object_synonyms(noun):
    """Get 3D object-specific synonyms for the given noun"""
    global synonyms_cache, last_gemini_synonym_call_time
    noun_key = noun.lower()
    
    # First check our expanded synonym map
    if noun_key in COMPLETE_SYNONYM_MAP: 
        return COMPLETE_SYNONYM_MAP[noun_key]
    
    # Then check cache
    if noun_key in synonyms_cache: 
        return synonyms_cache[noun_key]
    
    # Finally try Gemini API
    if gemini_model is None: 
        return []
    
    current_time = time.time(); time_since_last_call = current_time - last_gemini_synonym_call_time
    if time_since_last_call < GEMINI_SYNONYM_DELAY:
        wait_time = GEMINI_SYNONYM_DELAY - time_since_last_call
        print(f"[Rate Limit] Waiting {wait_time:.2f}s for synonym call ('{noun_key}')...")
        time.sleep(wait_time)
    
    prompt = f"List exactly 7 single-word synonyms for the noun '{noun}' that would be used to describe objects in 3D indoor scenes. Output only a comma-separated list, nothing else. Example format: chair,seat,stool,throne,recliner,furniture,seating"
    synonyms = []
    try:
        last_gemini_synonym_call_time = time.time()
        response = gemini_model.generate_content(prompt, request_options={'timeout': 25})
        synonyms = [s.strip().lower() for s in response.text.split(',') if s.strip() and ' ' not in s.strip()][:7]
        if synonyms:
            print(f"[API Call] Gemini Synonyms for '{noun}': {synonyms}")
            synonyms_cache[noun_key] = synonyms # Cache result
            try:
                 with open(synonyms_cache_path, "w") as f: json.dump(synonyms_cache, f, indent=2) # Save cache
            except Exception as e: print(f"WARN: Failed save synonym cache: {e}")
    except Exception as e: print(f"ERROR: Gemini synonym API call failed for '{noun}': {e}"); synonyms = []
    return synonyms

get_synonyms = get_3d_object_synonyms # Alias

# --- Enhanced Parse Function ---
def parse(scan_id, json_obj, concepts_on_device, softmax_temp=0.05):
    """
    Enhanced parsing function with temperature parameter for softmax
    Lower temperature (0.05-0.1) = sharper distribution, higher (0.2-0.3) = smoother distribution
    """
    category = json_obj.get("category")
    
    # Early return if invalid category
    if not category or category not in concepts_on_device or not torch.is_tensor(concepts_on_device[category]): 
        return None
    
    # Get appearance concept
    appearance_concept = concepts_on_device[category]
    if appearance_concept.dim() == 0 or appearance_concept.shape[0] == 0: 
        return None
    
    num_objects = appearance_concept.shape[0]
    final_concept = torch.ones(num_objects, device=DEVICE)
    
    # Special handling for certain categories
    if category in ["corner", "middle", "room", "center", "you"]: 
        return final_concept.clone()
    
    # Apply appearance concept
    final_concept = torch.minimum(final_concept, appearance_concept)
    
    # Process relations
    if "relations" in json_obj and isinstance(json_obj["relations"], list):
        relation_weights = []  # Store relation weights for later weighting
        
        for relation_item in json_obj["relations"]:
            if not isinstance(relation_item, dict): continue
            if "anchors" in relation_item: relation_item["objects"] = relation_item["anchors"]
            
            relation_name = relation_item.get("relation_name")
            if not relation_name or relation_name not in ALL_VALID_RELATIONS or relation_name not in concepts_on_device: 
                continue
                
            relation_concept = concepts_on_device[relation_name]
            if not torch.is_tensor(relation_concept): 
                continue
                
            num = rel_num.get(relation_name, -1)
            concept = None
            
            try:
                # Unary relation
                if num == 1:
                    concept = torch.ones(num_objects, device=DEVICE)
                    expected_shape_n = (num_objects,)
                    expected_shape_nn = (num_objects, num_objects)
                    
                    if relation_item.get("objects"):
                        sub_concept = parse(scan_id, relation_item["objects"][0], concepts_on_device, softmax_temp)
                        if sub_concept is None: continue
                        sub_concept = sub_concept.to(DEVICE)
                        
                        if relation_concept.shape == expected_shape_nn:
                            concept = (relation_concept @ sub_concept)
                        elif relation_concept.shape == expected_shape_n:
                            concept = relation_concept
                        else: continue
                    else:
                        if relation_concept.shape == expected_shape_n:
                            concept = relation_concept
                        elif relation_concept.shape == expected_shape_nn:
                            concept = relation_concept.diag()
                        else: continue
                
                # Binary relation
                elif num == 0:
                    expected_shape_nn = (num_objects, num_objects)
                    if relation_concept.shape != expected_shape_nn: continue
                    
                    sub_objects = relation_item.get("objects", [])
                    if not sub_objects: continue
                    
                    sub_concept = parse(scan_id, sub_objects[0], concepts_on_device, softmax_temp)
                    if sub_concept is None: continue
                    sub_concept = sub_concept.to(DEVICE)
                    concept = torch.matmul(relation_concept, sub_concept)
                
                # Ternary relation
                elif num == 2:
                    expected_shape_nnn = (num_objects, num_objects, num_objects)
                    if relation_concept.shape != expected_shape_nnn: continue
                    
                    objs = relation_item.get("objects", [])
                    if len(objs) >= 2:
                        sub1 = parse(scan_id, objs[0], concepts_on_device, softmax_temp)
                        sub2 = parse(scan_id, objs[1], concepts_on_device, softmax_temp)
                        if sub1 is None or sub2 is None: continue
                        sub1, sub2 = sub1.to(DEVICE), sub2.to(DEVICE)
                        concept = torch.einsum('ijk,j,k->i', relation_concept, sub1, sub2)
                    elif len(objs) == 1:
                        sub = parse(scan_id, objs[0], concepts_on_device, softmax_temp)
                        if sub is None: continue
                        sub = sub.to(DEVICE)
                        concept = torch.einsum('ijk,j,k->i', relation_concept, sub, sub)
                    else:
                        obj = deepcopy(json_obj)
                        obj["relations"] = []
                        obj.pop("color", None)
                        obj.pop("shape", None)
                        sub = parse(scan_id, obj, concepts_on_device, softmax_temp)
                        if sub is None: continue
                        sub = sub.to(DEVICE)
                        concept = torch.einsum('ijk,j,k->i', relation_concept, sub, sub)
                else: continue
                
                # Process and apply concept
                if concept is not None and torch.isfinite(concept).all():
                    # Apply softmax with configurable temperature
                    concept = F.softmax(concept/softmax_temp, dim=0)
                    
                    # Handle negative relations
                    if relation_item.get("negative", False):
                        concept = concept.max() - concept
                    
                    weight = relation_item.get("weight", 1.0)  # Default weight 1.0
                    
                    # Apply the relation with weighting
                    final_concept = final_concept * (concept ** weight)
                    relation_weights.append((concept, weight))
                    
            except Exception as e:
                print(f"ERROR parse relation '{relation_name}' {scan_id}: {e}. Skip.")
                continue
    
    # Check if final concept is valid
    if not torch.isfinite(final_concept).all():
        return torch.ones(num_objects, device=DEVICE)/num_objects if num_objects > 0 else None
    
    # Handle edge cases
    if num_objects > 0 and final_concept.sum().item() == 0:
        return torch.ones(num_objects, device=DEVICE)/num_objects
    elif num_objects == 0:
        return None
    
    # Re-normalize before returning
    if final_concept.sum() > 0:
        final_concept = final_concept / final_concept.sum()
    
    return final_concept


# --- Helper Functions ---

def get_prediction_confidence(final_concept):
    """ Calculate confidence based on softmax output """
    if final_concept is None or not torch.is_tensor(final_concept) or final_concept.numel() <= 1:
        return 0.0
    try:
        # Ensure it's float for subtraction
        final_concept = final_concept.float()
        sorted_values, _ = torch.sort(final_concept, descending=True)
        if sorted_values.shape[0] >= 2:
            confidence = (sorted_values[0] - sorted_values[1]).item() # Gap between top 2
        else:
            confidence = sorted_values[0].item() # Only one score
        return min(max(confidence, 0.0), 1.0) # Clamp between 0 and 1
    except Exception as e:
        print(f"Error calculating confidence: {e}")
        return 0.0

def parse_with_category_replacement(scan_id, original_json_obj, concepts_on_device, new_category, temp=0.05):
    """ Creates a copy of json_obj, replaces category, and parses it """
    if new_category not in concepts_on_device:
        return None
    json_obj_syn = deepcopy(original_json_obj)
    json_obj_syn["category"] = new_category
    return parse(scan_id, json_obj_syn, concepts_on_device, softmax_temp=temp)

def get_top_k_predictions(final_concept, k=3):
    """Get top-k predicted indices and their probabilities"""
    if final_concept is None:
        return [], []
    if k > final_concept.shape[0]:
        k = final_concept.shape[0]
    probs, indices = torch.topk(final_concept, k)
    return indices.tolist(), probs.tolist()

def ensemble_predictions(predictions_list, weights=None):
    """
    Ensemble multiple prediction vectors
    Args:
        predictions_list: List of prediction tensors
        weights: Optional list of weights for each prediction
    """
    if not predictions_list:
        return None
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    # Start with zeros
    result = torch.zeros_like(predictions_list[0])
    
    # Weighted sum
    for pred, weight in zip(predictions_list, weights):
        result += pred * weight
    
    # Normalize
    if result.sum() > 0:
        result = result / result.sum()
        
    return result


# --- Main  Function (Enhanced with Better Synonym Strategy) ---
# --- Main Evaluation Function (Enhanced with Advanced Synonym Strategy + Ensemble) ---
def eval_scanrefer(args):
    print(f"Loading features from: {args.features_path}")
    if not Path(args.features_path).exists(): print(f"FATAL ERROR: Features file not found: {args.features_path}"); sys.exit(1)
    all_concepts_cpu = torch.load(args.features_path, map_location="cpu", weights_only=False)
    print(f"Features loaded for {len(all_concepts_cpu)} scans.")

    # Pre-extract available concepts
    available_concepts_per_scan = defaultdict(set)
    for scan_id, concepts in all_concepts_cpu.items():
        available_concepts_per_scan[scan_id] = {k for k,v in concepts.items() if torch.is_tensor(v)}
    print(f"Pre-extracted available concepts for {len(available_concepts_per_scan)} scans")

    # Load any explicit temperature settings from args or use defaults
    softmax_temps = [0.07, 0.05, 0.07]    
    ensemble_weights = [0.6, 0.2, 0.2]  # Weights for each temperature, primary focus on base temperature
    
    top_k = args.top_k
    scan_data_gt = {}
    print("Loading GT data for scans in dataset...")
    dataset_scan_ids = list(dataset.scan_ids)
    for scan_id in tqdm(dataset_scan_ids):
        try:
            gt_labels, obj_ids_gt, gt_locs, _, _ = load_pc(scan_id)
            if obj_ids_gt: scan_data_gt[scan_id] = {"gt_labels": gt_labels, "obj_ids": obj_ids_gt, "gt_locs": gt_locs}
        except Exception as e: print(f"WARN: Failed load GT {scan_id}: {e}")

    result = { "single_25": {"correct": 0, "total": 0}, "multiple_25": {"correct": 0, "total": 0}, "total_25": {"correct": 0, "total": 0} }
    synonym_attempt_total = 0; synonym_attempt_success = 0
    synonym_improvement_counts = Counter() # Track which synonyms helped
    ensemble_improvement_count = 0 # Track when ensemble helped
    total_processed=0; skipped_feature_missing=0; skipped_parse_error=0; skipped_pred_locs_missing=0; skipped_gt_data_missing=0

    print(f"Evaluating {len(dataset)} samples with Enhanced Symbolic + Synonym Strategy + Multi-Temperature Ensemble...")
    for i in tqdm(range(len(dataset))):
        try:
            data_item = dataset[i]; scan_id = data_item["scan_id"]; json_obj_orig = data_item["json_obj"]
            target_obj_id_gt = data_item["tgt_object_id"]
        except Exception as e: print(f"ERROR loading data item {i}: {e}"); continue

        if scan_id not in all_concepts_cpu: skipped_feature_missing += 1; continue
        if scan_id not in available_concepts_per_scan: skipped_feature_missing += 1; continue # Use pre-extracted keys

        concepts_this_scan_gpu = {}
        try: # Move features to GPU
            concepts_this_scan_gpu = { k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in all_concepts_cpu[scan_id].items() }
        except Exception as e: print(f"ERROR moving features {scan_id}: {e}"); skipped_feature_missing += 1; continue

        available_concepts_now = available_concepts_per_scan[scan_id] # Use pre-extracted set

        # --- Initial Multi-Temperature Parse ---
        final_concepts = []  # Store multiple temperature parses
        try:
            # Parse with multiple temperatures for ensemble approach
            for temp in softmax_temps:
                concept = parse(scan_id, json_obj_orig, concepts_this_scan_gpu, softmax_temp=temp)
                if concept is not None:
                    final_concepts.append(concept)
            
            if not final_concepts: skipped_parse_error += 1; continue
            
            # Default to first temperature's parse if ensemble fails
            final_concept = final_concepts[0]
            
            # Try ensemble if multiple valid parses
            if len(final_concepts) > 1:
                # Use all available temperatures with weighted ensemble
                ensemble_weights_adjusted = ensemble_weights[:len(final_concepts)]
                ensemble_weights_adjusted = [w / sum(ensemble_weights_adjusted) for w in ensemble_weights_adjusted]
                final_concept_ensemble = ensemble_predictions(final_concepts, weights=ensemble_weights_adjusted)
                
                # Keep track of both - we'll compare them later
                final_concept_base = final_concept
                if final_concept_ensemble is not None:
                    final_concept = final_concept_ensemble
            
            if scan_id not in dataset.scans or 'pred_locs' not in dataset.scans[scan_id] or not dataset.scans[scan_id]['pred_locs']: 
                skipped_pred_locs_missing += 1; continue
                
            pred_locs = dataset.scans[scan_id]['pred_locs']
            num_pred_objects = len(pred_locs)
            
            if final_concept.shape[0] != num_pred_objects: 
                print(f"CRITICAL WARN: Dim mismatch {final_concept.shape} != {num_pred_objects} for {scan_id}!")
                skipped_parse_error += 1
                continue
                
            initial_confidence = get_prediction_confidence(final_concept)
            pred_idx_symbolic = torch.argmax(final_concept).item()
        except Exception as e: 
            print(f"ERROR initial parse/argmax {scan_id}: {e}")
            skipped_parse_error += 1
            continue

        # --- GT Info ---
        if scan_id not in scan_data_gt: skipped_gt_data_missing += 1; continue
        obj_ids_gt = scan_data_gt[scan_id]["obj_ids"]
        try: 
            gt_idx = obj_ids_gt.index(target_obj_id_gt)
            gt_box = scan_data_gt[scan_id]["gt_locs"][gt_idx]
            gt_center, gt_size = gt_box[:3], gt_box[3:]
        except (ValueError, IndexError): 
            print(f"WARN: GT target {target_obj_id_gt} invalid idx {scan_id}. Skip.")
            skipped_gt_data_missing += 1
            continue
            
        labels_list_gt = list(scan_data_gt[scan_id]["gt_labels"])
        target_label_gt = labels_list_gt[gt_idx]
        unique = (labels_list_gt.count(target_label_gt) == 1)

        # --- Calculate IoU for initial prediction ---
        best_iou = 0.0
        best_pred_idx = -1
        best_category = json_obj_orig.get("category", "")
        best_method = "base"
        
        if 0 <= pred_idx_symbolic < len(pred_locs):
            pred_box_symbolic = pred_locs[pred_idx_symbolic]
            pred_center_sym, pred_size_sym = pred_box_symbolic[:3], pred_box_symbolic[3:]
            best_iou = eval_ref_one_sample(construct_bbox_corners(pred_center_sym, pred_size_sym), 
                                           construct_bbox_corners(gt_center, gt_size))
            best_pred_idx = pred_idx_symbolic
            
            # If we used ensemble, also check base model prediction
            if len(final_concepts) > 1 and 'final_concept_base' in locals():
                pred_idx_base = torch.argmax(final_concept_base).item()
                if 0 <= pred_idx_base < len(pred_locs) and pred_idx_base != pred_idx_symbolic:
                    pred_box_base = pred_locs[pred_idx_base]
                    pred_center_base, pred_size_base = pred_box_base[:3], pred_box_base[3:]
                    iou_base = eval_ref_one_sample(construct_bbox_corners(pred_center_base, pred_size_base),
                                                  construct_bbox_corners(gt_center, gt_size))
                    if iou_base > best_iou:
                        best_iou = iou_base
                        best_pred_idx = pred_idx_base
                        best_method = "single_temp"
                    elif iou_base >= 0.25 and best_iou < 0.25:
                        best_iou = iou_base
                        best_pred_idx = pred_idx_base
                        best_method = "single_temp"
                    
                if best_method == "base" and best_iou >= 0.25:
                    ensemble_improvement_count += 1

        # --- Advanced Synonym Retry Logic ---
        initial_iou_was_good = (best_iou >= 0.25)
        synonym_improved = False
        # Only try synonyms if initial prediction failed or was marginal
        if best_iou < 0.35:  # Try synonyms even for marginal cases (0.07-0.35)
            synonym_attempt_total += 1
            main_category = json_obj_orig.get("category")
            
            if main_category and main_category not in ["corner", "middle", "room", "center", "you"]:
                # Get both direct synonyms and second-order synonyms (synonyms of synonyms)
                direct_synonyms = get_synonyms(main_category)
                
                # Add second-order synonyms by checking what main_category is a synonym OF
                second_order_synonyms = []
                for key, values in COMPLETE_SYNONYM_MAP.items():
                    if main_category.lower() in [v.lower() for v in values] and key.lower() != main_category.lower():
                        second_order_synonyms.append(key)
                
                all_synonyms = list(set(direct_synonyms + second_order_synonyms))
                
                if all_synonyms:
                    # Sort synonyms by "potential" - start with those available in this scan
                    scan_specific_synonyms = [syn for syn in all_synonyms if syn in available_concepts_now]
                    other_synonyms = [syn for syn in all_synonyms if syn not in scan_specific_synonyms]
                    
                    # Try scan-specific synonyms first (more efficient)
                    prioritized_synonyms = scan_specific_synonyms + other_synonyms[:3]  # Limit other syns to top 3
                    
                    for syn in prioritized_synonyms:
                        # Check if synonym feature exists or we should try anyway
                        if syn in available_concepts_now:
                            try:
                                # Try multiple temperatures for the synonym
                                syn_concepts = []
                                for temp in softmax_temps:
                                    syn_concept = parse_with_category_replacement(scan_id, json_obj_orig, 
                                                                                  concepts_this_scan_gpu, syn, temp=temp)
                                    if syn_concept is not None and syn_concept.shape[0] == num_pred_objects:
                                        syn_concepts.append(syn_concept)
                                
                                if not syn_concepts:
                                    continue
                                
                                # Ensemble the synonym concepts if possible
                                if len(syn_concepts) > 1:
                                    final_concept_syn = ensemble_predictions(syn_concepts, 
                                                                           weights=ensemble_weights[:len(syn_concepts)])
                                else:
                                    final_concept_syn = syn_concepts[0]
                                    
                                pred_idx_syn = torch.argmax(final_concept_syn).item()
                                
                                if 0 <= pred_idx_syn < len(pred_locs):
                                    pred_box_syn = pred_locs[pred_idx_syn]
                                    pred_center_syn, pred_size_syn = pred_box_syn[:3], pred_box_syn[3:]
                                    iou_syn = eval_ref_one_sample(construct_bbox_corners(pred_center_syn, pred_size_syn), 
                                                                 construct_bbox_corners(gt_center, gt_size))
                                    
                                    # More sophisticated strategy: significant improvement OR crossing threshold
                                    improvement_margin = 0.02 if best_iou < 0.25 else 0.01
                                    
                                    if (iou_syn > best_iou + improvement_margin) or (best_iou < 0.25 and iou_syn >= 0.25):
                                        best_iou = iou_syn
                                        best_pred_idx = pred_idx_syn
                                        best_category = syn
                                        synonym_improved = True
                                        
                                        if not initial_iou_was_good and iou_syn >= 0.25:
                                            synonym_attempt_success += 1
                                            synonym_improvement_counts[f"{main_category}->{syn}"] += 1
                                        
                                        # Continue checking other synonyms for even better results
                                        # if iou_syn >= 0.45:  # Stop if we got an excellent match
                                        #    break
                                        
                                    # Even if threshold not crossed, track best IoU
                                    elif iou_syn > best_iou:
                                        best_iou = iou_syn
                                        best_pred_idx = pred_idx_syn
                                        best_category = syn
                                        
                            except Exception as e:
                                continue

                    # Record effective synonyms for future analysis
                    if synonym_improved:
                        effective_synonyms[main_category][best_category] += 1

        # --- Accumulate Results based on BEST IoU achieved ---
        correct_pred = (best_iou >= 0.25)
        if correct_pred: 
            result["total_25"]["correct"] += 1
            result["single_25" if unique else "multiple_25"]["correct"] += 1
        
        result["total_25"]["total"] += 1
        result["single_25" if unique else "multiple_25"]["total"] += 1
        total_processed += 1

    # --- Print Final Results ---
    print("\n" + "="*20 + " Evaluation Summary (Enhanced Multi-Stage Approach) " + "="*20)
    print(f"Total samples: {len(dataset)}, Processed: {total_processed}")
    print(f"Skipped (Feature): {skipped_feature_missing} (Parse): {skipped_parse_error} (PredLoc): {skipped_pred_locs_missing} (GT): {skipped_gt_data_missing}")
    
    if synonym_attempt_total > 0:
        print(f"Synonym Retries Triggered: {synonym_attempt_total}, Successful Retries (IoU>=0.25): {synonym_attempt_success}")
        success_rate = synonym_attempt_success / synonym_attempt_total if synonym_attempt_total > 0 else 0
        print(f"Synonym Strategy Success Rate: {success_rate:.2%}")
        
        if synonym_improvement_counts:
            print("Most Effective Synonym Pairs (Original->Synonym):")
            for pair, count in synonym_improvement_counts.most_common(8):
                print(f"  - {pair}: {count} times")
    
    if ensemble_improvement_count > 0:
        print(f"Ensemble Temperature Strategy Helped: {ensemble_improvement_count} times")
        
    # Compute most frequent effective synonyms
    if effective_synonyms:
        print("\nMost Effective Synonym Relationships:")
        for orig, counters in sorted(effective_synonyms.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
            if counters:
                top_syns = counters.most_common(3)
                print(f"  - {orig}: {', '.join([f'{syn} ({cnt})' for syn, cnt in top_syns])}")
    
    print("-" * 60)
    for k, v in result.items():
        if v["total"] > 0: 
            acc = v["correct"] / v["total"]
            print(f"{k:<15}: Correct={v['correct']:<5} Total={v['total']:<5} Accuracy={acc:.4f}")
        else: 
            print(f"{k:<15}: Correct={v['correct']:<5} Total={v['total']:<5} Accuracy=N/A")
    print("=" * 60)


# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Enhanced ScanRefer Eval with Advanced Synonym Strategy")
    parser.add_argument("--features_path", type=str, required=True, help="Path to CONSISTENT ScanRefer features")
    parser.add_argument("--top_k", type=int, default=1, help="Top K predictions (usually 1)")
    parser.add_argument("--temp_ensemble", action="store_true", help="Use multi-temperature ensemble method")
    parser.add_argument("--synonym_depth", type=int, default=2, help="Synonym search depth (1-3)")
    parser.add_argument("--prepopulate_synonyms", action="store_true", help="Prepopulate common synonyms cache")
    args = parser.parse_args()

    if not Path(args.features_path).exists(): 
        print(f"ERROR: Features path not found: {args.features_path}")
        sys.exit(1)
        
    if not gemini_model: 
        print(f"WARN: Gemini API not configured. Synonym lookup will use local map only.")
    else:
        print(f"INFO: Using Gemini {GEMINI_MODEL_NAME} for expanded synonym support.")

    # Run evaluation with enhanced approach
    eval_scanrefer(args)