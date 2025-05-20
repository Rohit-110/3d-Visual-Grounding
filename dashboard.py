# --- START OF FILE dashboard.py ---

import streamlit as st
import sys
import os
import tempfile
import numpy as np
import json
import io # Not used currently, but potentially useful
import base64 # Not used currently
from PIL import Image
import importlib # Use importlib for checks
import traceback # For detailed error printing
import glob # For finding scene files

# --- Check Required Libraries ---
required_libraries = {
    'streamlit': 'streamlit', 'open3d': 'open3d', 'torch': 'torch',
    'numpy': 'numpy', 'PIL': 'pillow', 'dotenv': 'python-dotenv',
    'google.generativeai': 'google-generativeai', 'nltk': 'nltk',
    'cv2': 'opencv-python', 'pandas': 'pandas', 'skimage': 'scikit-image'
}
missing_libs_for_pip = []
print("--- Dashboard: Checking required libraries ---")
for module_name, pip_name in required_libraries.items():
    try: importlib.import_module(module_name.replace('-', '_')); print(f"  [OK] {module_name}") # Handle potential hyphens in module names
    except ImportError: print(f"  [MISSING] {module_name} (pip install {pip_name})"); missing_libs_for_pip.append(pip_name)
    except Exception as e: print(f"  [ERROR] Checking {module_name}: {e}"); missing_libs_for_pip.append(pip_name)

if missing_libs_for_pip:
    st.error(f"Required libraries missing: {', '.join(missing_libs_for_pip)}. Please install them (e.g., pip install {' '.join(missing_libs_for_pip)}).")
    st.info(f"Current Python environment: {sys.executable}")
    st.stop()
print("--- Dashboard: Library check complete ---")

# --- Path Setup ---
# Assumes dashboard.py is in project root, and check3.py is also in project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT); print(f"[Dashboard INFO] Added to sys.path: {PROJECT_ROOT}")
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH); print(f"[Dashboard INFO] Added to sys.path: {SRC_PATH}")

# --- Import from check3 and O3D ---
try:
    # Import directly from project root
    from check3 import (
        process_scene, 
        setup_nltk, 
        configure_gemini, 
        CHECK2_DEVICE, 
        build_point_cloud,
        visualize_suggestion,
        load_instance_names
    )
    
    import open3d as o3d
    import torch
    import numpy as np
    print("[Dashboard INFO] Successfully imported from check3.py and open3d")
    # Perform one-time setups
    setup_nltk()
    configure_gemini()
except ImportError as e:
    st.error(f"Error importing 'check3' or 'open3d': {e}")
    st.error(f"Ensure 'check3.py' is in '{PROJECT_ROOT}' and Open3D is installed.")
    st.stop()
except Exception as e:
    st.error(f"Error during initial setup (nltk/gemini): {e}"); st.stop()

# --- Page Config ---
st.set_page_config(page_title="3D Scene Activity Suggestions", page_icon="🏠", layout="wide")

# --- Find Available Scenes ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache for an hour
def get_available_scenes():
    """Find all available scenes in the data directory"""
    pcd_dir = "data/referit3d/scan_data/pcd_with_global_alignment"
    if not os.path.exists(pcd_dir):
        print(f"WARNING: PCD directory not found: {pcd_dir}")
        return []
    
    scene_files = glob.glob(os.path.join(pcd_dir, "scene*.pth"))
    scene_ids = [os.path.basename(f).replace(".pth", "") for f in scene_files]
    scene_ids.sort()
    return scene_ids

# --- Visualization Capture Function (Returns Bytes) ---
@st.cache_data(max_entries=50, show_spinner=False) # Cache rendering result, increased cache size for scene browser
def capture_visualization_bytes(_scan_id, _prompt, pcd_points, pcd_colors, bbox_data):
    """ Renders Open3D scene off-screen and returns image bytes. """
    print(f"[Visualization] Capturing view for cache key: ({_scan_id}, {_prompt[:20] if _prompt else 'browse'}...)")
    vis = None; img_bytes = None; tmp_path = None
    try:
        pcd = o3d.geometry.PointCloud()
        if pcd_points is not None and len(pcd_points) > 0:
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            if pcd_colors is not None and len(pcd_colors) == len(pcd_points):
                 try: pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
                 except Exception: pcd.paint_uniform_color([0.5, 0.5, 0.5])
            else: pcd.paint_uniform_color([0.5, 0.5, 0.5])

        bbox_list = []
        if bbox_data:
            for center, size, color in bbox_data:
                try:
                    center_list=np.array(center).tolist(); size_list=np.array(size).tolist(); color_list=np.array(color).tolist()
                    obb = o3d.geometry.OrientedBoundingBox(center_list, np.eye(3).tolist(), size_list)
                    obb.color = color_list; bbox_list.append(obb)
                except Exception as box_e: print(f"WARN: Failed create bbox C={center}, S={size}, Col={color}. Error: {box_e}")

        if not pcd.has_points() and not bbox_list: print("[Visualization] No valid geometries."); return None

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file: tmp_path = tmp_file.name

        vis = o3d.visualization.Visualizer(); vis.create_window(visible=False, width=800, height=600)
        view_set = False
        if pcd.has_points(): vis.add_geometry(pcd); view_set = True
        if bbox_list:
             for bbox in bbox_list: vis.add_geometry(bbox)
             if not view_set and bbox_list: ctr=bbox_list[0].get_center(); vis.get_view_control().set_lookat(ctr); view_set = True
        if not view_set: vis.get_view_control().set_front([0,0,-1]); vis.get_view_control().set_up([0,-1,0])
        vis.get_view_control().set_zoom(0.7); vis.poll_events(); vis.update_renderer(); vis.capture_screen_image(tmp_path, do_render=True); vis.destroy_window(); vis = None

        if os.path.exists(tmp_path):
            with open(tmp_path, "rb") as f: img_bytes = f.read()
            print(f"[Visualization] Captured {len(img_bytes)} bytes.")
        else: print("[Visualization] Error: Temp image file missing post-capture.")
    except Exception as e: print(f"ERROR during O3D capture: {e}"); print(traceback.format_exc()); img_bytes = None
    finally:
        if vis is not None: vis.destroy_window() # Ensure cleanup
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except Exception as e_unlink: print(f"WARN: Failed delete temp image {tmp_path}: {e_unlink}")
    return img_bytes

# --- Scene Preview Function ---
@st.cache_data(max_entries=50, show_spinner=False)
def get_scene_preview(scan_id):
    """Generate a preview of the scene"""
    # Load point cloud data
    raw_pcd_path = f"data/referit3d/scan_data/pcd_with_global_alignment/{scan_id}.pth"
    pcd_points_np = None
    pcd_colors_np = None
    
    try:
        points_raw, colors_raw, _, _ = torch.load(raw_pcd_path, map_location="cpu", weights_only=False)
        if colors_raw is not None and colors_raw.max() > 1.0: 
            colors_raw = colors_raw / 255.0
        pcd_points_np = points_raw.numpy() if torch.is_tensor(points_raw) else points_raw
        pcd_colors_np = colors_raw.numpy() if torch.is_tensor(colors_raw) else colors_raw
    except Exception as e:
        print(f"Could not load point cloud for {scan_id}: {e}")
        return None
    
    # Get instance names to count objects
    instance_mapping = load_instance_names(scan_id)
    object_count = len(instance_mapping) if instance_mapping else 0
    
    # Try to load scene data for bounding boxes
    try:
        from src.dataset.datasets import load_one_scene
        _, scene = load_one_scene(scan_id, label_type="pred")
        
        # Create bounding boxes
        bbox_data_tuples = []
        if scene and "pred_locs" in scene:
            obj_locs = scene.get("pred_locs", [])
            for i, bbox in enumerate(obj_locs):
                center, size = bbox[:3], np.maximum(bbox[3:], 1e-6)
                if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)):
                    continue
                # Convert to hashable tuples
                center_tuple = tuple(float(x) for x in center)
                size_tuple = tuple(float(x) for x in size)
                color = (1, 0, 0)  # All red for preview
                bbox_data_tuples.append((center_tuple, size_tuple, color))
        
        # Generate image
        image_bytes = capture_visualization_bytes(
            scan_id,
            None,  # No prompt for previews
            pcd_points_np,
            pcd_colors_np,
            tuple(sorted(bbox_data_tuples)) if bbox_data_tuples else None
        )
        
        return {
            "image_bytes": image_bytes,
            "object_count": object_count,
            "objects": list(instance_mapping.values()) if instance_mapping else []
        }
    except Exception as e:
        print(f"Error generating preview for {scan_id}: {e}")
        
        # Fallback to just point cloud without bounding boxes
        image_bytes = capture_visualization_bytes(
            scan_id,
            None,
            pcd_points_np,
            pcd_colors_np,
            None
        )
        
        return {
            "image_bytes": image_bytes,
            "object_count": object_count,
            "objects": list(instance_mapping) if instance_mapping else []
        }

# --- Paginator Helper ---
def paginator(items, items_per_page=10, key="paginator"):
    """Simple paginator for items"""
    # Initialize page number
    if f'{key}_page' not in st.session_state:
        st.session_state[f'{key}_page'] = 0
    
    # Get current page number
    page_num = st.session_state[f'{key}_page']
    
    # Calculate total pages
    total_pages = (len(items) - 1) // items_per_page + 1
    
    # Get items for current page
    start_idx = page_num * items_per_page
    end_idx = min(start_idx + items_per_page, len(items))
    page_items = items[start_idx:end_idx]
    
    # Navigation buttons
    cols = st.columns([1, 1, 2, 1, 1])
    
    if cols[0].button("⏪ First", key=f"{key}_first"):
        st.session_state[f'{key}_page'] = 0
        st.rerun()
    
    if cols[1].button("◀️ Prev", key=f"{key}_prev", disabled=(page_num <= 0)):
        st.session_state[f'{key}_page'] -= 1
        st.rerun()
    
    cols[2].write(f"Page {page_num + 1} of {total_pages}")
    
    if cols[3].button("Next ▶️", key=f"{key}_next", disabled=(page_num >= total_pages - 1)):
        st.session_state[f'{key}_page'] += 1
        st.rerun()
    
    if cols[4].button("Last ⏩", key=f"{key}_last"):
        st.session_state[f'{key}_page'] = total_pages - 1
        st.rerun()
    
    return page_items

# --- UI Definition ---
st.title("🏠 3D Scene Activity Suggestion Dashboard")

# Create tabs for different sections
tab_main, tab_browser = st.tabs(["Activity Suggestions", "Scene Browser"])

# --- Scene Browser Tab ---
with tab_browser:
    st.header("📋 Available 3D Scenes")
    st.markdown("Browse all available scenes. Click on a scene to use it for activity suggestions.")
    
    # Load all available scenes
    all_scenes = get_available_scenes()
    
    if not all_scenes:
        st.warning("No scenes found. Please check the data directory.")
    else:
        st.success(f"Found {len(all_scenes)} available scenes")
        
        # Display scene grid with pagination
        scenes_per_page = 6  # Show 6 scenes per page
        page_scenes = paginator(all_scenes, scenes_per_page, "scene_browser")
        
        # Create rows of 3 scenes each
        for i in range(0, len(page_scenes), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(page_scenes):
                    scene_id = page_scenes[i+j]
                    with cols[j]:
                        st.subheader(f"{scene_id}")
                        
                        # Get scene preview
                        preview_data = get_scene_preview(scene_id)
                        
                        if preview_data and preview_data["image_bytes"]:
                            st.image(preview_data["image_bytes"], caption=f"Scene {scene_id}", use_column_width=True)
                            st.caption(f"Objects: {preview_data['object_count']}")
                            
                            # Truncate object list if too long
                            if preview_data["objects"]:
                                if len(preview_data["objects"]) > 5:
                                    object_text = ", ".join(preview_data["objects"][:5]) + f"... (+{len(preview_data['objects'])-5} more)"
                                else:
                                    object_text = ", ".join(preview_data["objects"])
                                st.caption(f"Contains: {object_text}")
                            
                            # Button to use this scene
                            if st.button(f"Use this scene", key=f"use_scene_{scene_id}"):
                                st.session_state.scene_id_input = scene_id
                                st.session_state.active_tab = "Activity Suggestions"
                                st.rerun()
                        else:
                            st.error(f"Could not load preview for {scene_id}")

# --- Activity Suggestions Tab ---
with tab_main:
    st.header("🧠 Get Activity Suggestions")
    st.markdown("Input a Scene ID and a natural language prompt to get activity suggestions based on objects in the scene.")

    col_main, col_sidebar = st.columns([3, 1])

    with col_sidebar:
        st.header("Settings & Examples")
        st.subheader("Configuration")
        st.caption(f"Using device: {CHECK2_DEVICE}")
        
        st.subheader("Examples")
        examples = [
            {"id": "scene0025_00", "prompt": "I am bored, what should I do?"},
            {"id": "scene0025_00", "prompt": "I need to relax"},
            {"id": "scene0050_00", "prompt": "I want to be productive"},
            {"id": "scene0050_00", "prompt": "What's a fun activity I could do right now?"},
        ]
        for i, example in enumerate(examples):
            if st.button(f"Use: {example['id']} - \"{example['prompt'][:30]}...\"", key=f"ex_{i}"):
                st.session_state.scene_id_input = example['id'] # Update state for text_input
                st.session_state.prompt_input = example['prompt']
                st.session_state.run_processing = True # Flag to trigger processing
                st.rerun()

    with col_main:
        st.subheader("Input")
        # Bind text inputs to session state using the 'key' argument
        scene_id = st.text_input("Scene ID", key="scene_id_input", value=st.session_state.get('scene_id_input', "scene0025_00"))
        prompt = st.text_input("Prompt", key="prompt_input", value=st.session_state.get('prompt_input', "I am bored, what should I do?"))

        if st.button("Get Activity Suggestions", type="primary"):
            st.session_state.run_processing = True # Set flag when button is clicked

        # Check if processing should run
        if st.session_state.get('run_processing', False):
            # Clear the flag
            st.session_state.run_processing = False

            current_scene_id = st.session_state.scene_id_input # Read from state
            current_prompt = st.session_state.prompt_input

            if not current_scene_id or not current_prompt:
                st.error("Please provide both Scene ID and Prompt.")
            else:
                st.divider(); st.subheader("Activity Suggestion")
                with st.spinner(f"Processing {current_scene_id} for '{current_prompt}'..."):
                    # Call the process_scene function from check3.py
                    results = process_scene(current_scene_id, current_prompt)

                    if results:
                        st.success("Processing complete!")
                        
                        # Display the suggestion
                        suggestion = results.get('suggestion', 'No suggestion available')
                        st.info(f"**Suggestion:** {suggestion}")
                        # Capture and Display Visualization
                        st.markdown("---")
                        st.subheader("3D Visualization")
                        
                        scene_data = results.get("scene")
                        
                        if scene_data:
                            # Extract the object name from the suggestion to highlight
                            import re
                            object_name_match = re.search(r"the\s+(\w+)", suggestion.lower())
                            target_object = object_name_match.group(1) if object_name_match else None
                            
                            # Load point cloud data
                            raw_pcd_path = f"data/referit3d/scan_data/pcd_with_global_alignment/{current_scene_id}.pth"
                            pcd_points_np = None
                            pcd_colors_np = None
                            
                            try:
                                points_raw, colors_raw, _, _ = torch.load(raw_pcd_path, map_location="cpu", weights_only=False)
                                if colors_raw is not None and colors_raw.max() > 1.0: 
                                    colors_raw = colors_raw / 255.0
                                pcd_points_np = points_raw.numpy() if torch.is_tensor(points_raw) else points_raw
                                pcd_colors_np = colors_raw.numpy() if torch.is_tensor(colors_raw) else colors_raw
                            except Exception as e:
                                st.warning(f"Could not load point cloud: {e}")
                                # Fallback to room points from scene data
                                room_points = scene_data.get("room_points")
                                if room_points is not None:
                                    if torch.is_tensor(room_points): 
                                        pcd_points_np = room_points.numpy()
                                    else:
                                        pcd_points_np = room_points
                                    pcd_colors_np = np.ones_like(pcd_points_np) * 0.5
                            
                            # Create bounding boxes
                            bbox_data_tuples = []
                            obj_locs = scene_data.get("pred_locs", [])
                            instance_mapping = load_instance_names(current_scene_id)

                            # Find objects matching the target from suggestion
                            highlighted_indices = []
                            if target_object:
                                for idx, name in enumerate(instance_mapping):
                                    if target_object.lower() in name.lower():
                                        try:
                                            obj_idx = scene_data.get("obj_ids", []).index(int(idx))
                                            highlighted_indices.append(obj_idx)
                                        except (ValueError, TypeError):
                                            continue

                            # ----- patched loop --------------------------------------------------
                            for i, bbox in enumerate(obj_locs):
                                center, size = bbox[:3], np.maximum(bbox[3:], 1e-6)
                                if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)):
                                    continue

                                # convert NumPy arrays to plain hashable tuples **here**
                                center_tuple = tuple(float(x) for x in center)
                                size_tuple   = tuple(float(x) for x in size)

                                color = (0, 1, 0) if i in highlighted_indices else (1, 0, 0)
                                bbox_data_tuples.append((center_tuple, size_tuple, color))
                            # ---------------------------------------------------------------------

                            # Use capture function
                            image_bytes = capture_visualization_bytes(
                                current_scene_id,
                                current_prompt,
                                pcd_points_np if pcd_points_np is not None else None,
                                pcd_colors_np if pcd_colors_np is not None else None,
                                tuple(sorted(bbox_data_tuples))   # ← sorting is now safe
                            )
                                                    
                            if image_bytes:
                                try: 
                                    st.image(image_bytes, caption=f"Scene: {current_scene_id} | Suggestion: {suggestion}", use_column_width=True)
                                    if target_object:
                                        st.caption(f"Green boxes highlight potential '{target_object}' objects mentioned in the suggestion")
                                except Exception as display_e: 
                                    st.error(f"Streamlit failed to display image bytes: {display_e}")
                            else: 
                                st.warning("Could not generate visualization image.")
                        else:
                            st.warning("No scene data available for visualization.")
                    else:
                        st.error("Processing failed. Check console logs for errors.")

# Footer
st.markdown("---"); st.markdown("<div style='text-align: center;'>3D Scene Activity Suggestion Dashboard</div>", unsafe_allow_html=True)

# --- END OF FILE dashboard.py ---