# =========================== demo_ease.py ===========================
"""
Minimal scene viewer + “find object” demo using **only** EaSe outputs.

Example
-------
$ python demo_ease.py --scan scene0010_00 \
                      --prompt "monitor on table near window"
"""

from __future__ import annotations
import argparse, sys, pathlib, numpy as np, torch, open3d as o3d

# ----------------------------------------------------------------------
# 1) put repo root on PYTHONPATH so that `src.` imports work everywhere
# ----------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = (THIS_DIR / "..").resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ----------------------------------------------------------------------
# 2) project-internal imports
# ----------------------------------------------------------------------
from src.dataset.datasets       import ScanReferDataset
from src.util.eval_helper       import load_pc, construct_bbox_corners, eval_ref_one_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
def get_scene_point_cloud(scan_id: str) -> np.ndarray:
    """
    Return an (N,3) NumPy array of XYZ points for `scan_id`.

    Tries three fallbacks in order:
    1. `ScanReferDataset().scans[scan_id]["pcd"]`
    2. Any (N,3) tensor/ndarray inside the tuple returned by `load_pc`
    3. The ScanNet `.ply` file  <repo>/data/scannet/scans/<scan>/..._vh_clean_2.ply
    """
    # 1. from ScanReferDataset cache ----------------------------------
    ds_tmp = ScanReferDataset()
    if "pcd" in ds_tmp.scans[scan_id]:
        xyz = ds_tmp.scans[scan_id]["pcd"]
        return np.asarray(xyz, dtype=np.float64)

    # 2. walk through load_pc tuple -----------------------------------
    tup = load_pc(scan_id)                       # returns many items
    for item in tup:
        if isinstance(item, torch.Tensor) and item.ndim == 2 and item.shape[1] == 3:
            return item.cpu().numpy().astype(np.float64)
        if isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[1] == 3:
            return item.astype(np.float64)

    # 3. fall back to raw .ply ----------------------------------------
    ply_path = ROOT_DIR / "data" / "scannet" / "scans" / scan_id / f"{scan_id}_vh_clean_2.ply"
    if ply_path.exists():
        pcd = o3d.io.read_point_cloud(str(ply_path))
        return np.asarray(pcd.points, dtype=np.float64)

    # if everything fails raise
    raise RuntimeError(f"[point-cloud] None found for {scan_id}")


# ----------------------------------------------------------------------
def run_detection(scan_id: str, prompt: str, top_k: int = 5) -> None:
    # ---------- load dataset & meta ----------
    ds    = ScanReferDataset()
    scene = ds.scans[scan_id]

    pred_locs = scene["pred_locs"]              # (M,6) cx,cy,cz,w,h,d
    obj_ids   = scene["obj_ids"]

    labels = scene.get("gt_labels", scene.get("pred_labels", [""] * len(obj_ids)))

    # ---------- keyword filter ----------
    keyword    = prompt.split()[0].lower()
    candidates = [i for i, lbl in enumerate(labels) if keyword in lbl.lower()]
    if not candidates:
        candidates = list(range(len(obj_ids)))
    candidates = candidates[:top_k]             # cap

    sel_idx = candidates[0]                     # choose first candidate
    sel_box = pred_locs[sel_idx]

    # ---------- IoU against GT ----------
    if "gt_locs" in scene:
        try:
            gt_box = scene["gt_locs"][obj_ids.index(obj_ids[sel_idx])]
            iou = eval_ref_one_sample(
                construct_bbox_corners(sel_box[:3], sel_box[3:]),
                construct_bbox_corners(gt_box [:3], gt_box [3:])
            )
            print(f"IoU with GT = {iou:.3f}")
        except Exception:
            pass

    # ---------- visualise ----------
    try:
        pcd_xyz = get_scene_point_cloud(scan_id)
    except RuntimeError as e:
        print(e)
        return

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_xyz))

    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = sel_box[:3]
    obb.extent = sel_box[3:]
    obb.color  = (1, 0, 0)                      # red box

    print(f"Showing {scan_id} — obj_id {obj_ids[sel_idx]} "
          f'→ label “{labels[sel_idx]}”')
    o3d.visualization.draw_geometries([pcd, obb])


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan",   required=True, help="ScanNet scan id, e.g. scene0010_00")
    ap.add_argument("--prompt", required=True, help="Free-form referring expression")
    args = ap.parse_args()

    run_detection(scan_id=args.scan, prompt=args.prompt)
# ======================================================================
