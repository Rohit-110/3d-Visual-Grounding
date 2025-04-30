# =========================== solver.py ===========================
"""
Minimal, pure-Python constraint solver for CSVG → EaSe bridge.

Callable from anywhere via
    from third_party.CSVG.solver import solve

`solve(expr, boxes)` returns a list of indices (0…k-1) inside `boxes`
that satisfy *all* relations in `expr`.  If no candidate satisfies every
relation it returns [].
"""

from __future__ import annotations
import numpy as np
from typing import List

# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _center(b: np.ndarray) -> np.ndarray:   # (6,) → (3,)
    return b[:3]

# binary relation predicates -------------------------------------------
def left(a,b):    return a[0] <  b[0] - 0.05
def right(a,b):   return a[0] >  b[0] + 0.05
def front(a,b):   return a[1] <  b[1] - 0.05      # ScanNet: +y towards back
def behind(a,b):  return a[1] >  b[1] + 0.05
def above(a,b):   return a[2] >  b[2] + 0.05
def below(a,b):   return a[2] <  b[2] - 0.05
def near(a,b):    return np.linalg.norm(a-b) < 1.0
def far(a,b):     return np.linalg.norm(a-b) > 2.0

REL_FUN = dict(
    left=left, right=right, front=front, behind=behind,
    above=above, below=below, near=near, far=far
)

# ----------------------------------------------------------------------
def _sat(expr: dict,
         boxes: np.ndarray,
         idx: int,
         mask: np.ndarray) -> bool:
    """Recursive satisfiability check for candidate `idx`."""
    if "relations" not in expr:
        return True      # only category -> ignore category here

    c_loc = _center(boxes[idx])

    for rel in expr["relations"]:
        name = rel["relation_name"]
        if name not in REL_FUN and name != "between":
            continue

        # build anchor mask -------------------------------------------
        if rel.get("objects"):
            # recurse on explicit anchor expression(s)
            tmp = np.zeros_like(mask)
            for j in np.where(mask)[0]:
                if _sat(rel["objects"][0], boxes, j, mask):
                    tmp[j] = True
            anch = tmp
        else:
            anch = mask

        # evaluate -----------------------------------------------------
        if name == "between":                       # ternary
            good = False
            inds = np.where(anch)[0]
            for j in inds:
                for k in inds:
                    if j == k: continue
                    p, q = _center(boxes[j]), _center(boxes[k])
                    t = np.dot(c_loc-p, q-p) / (np.linalg.norm(q-p)**2 + 1e-6)
                    if 0.0 < t < 1.0:
                        good = True; break
                if good: break
        else:                                       # binary
            fun  = REL_FUN[name]
            good = any(fun(c_loc, _center(boxes[j]))
                       for j in np.where(anch)[0] if j != idx)
            if rel.get("negative", False):
                good = not good

        if not good:
            return False
    return True

# ----------------------------------------------------------------------
def solve(expression_json: dict,
          boxes: np.ndarray) -> List[int]:
    """
    Parameters
    ----------
    expression_json : EaSe symbolic expression (dict)
    boxes : (k,6) np.ndarray (cx,cy,cz,w,h,d)

    Returns
    -------
    list[int]  indices of candidates that satisfy ALL constraints
    """
    k = len(boxes)
    if k == 0:
        return []

    ok = []
    all_mask = np.ones(k, dtype=bool)
    for i in range(k):
        mask = all_mask.copy()
        mask[i] = False         # anchors cannot be the target itself
        if _sat(expression_json, boxes, i, mask):
            ok.append(i)
    return ok
# ======================================================================
