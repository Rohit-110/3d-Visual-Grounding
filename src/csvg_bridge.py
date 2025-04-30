# =========================== csvg_bridge.py =============================
"""
Wrapper around CSVG's hard constraint solver.

`solve()` returns indices (inside the candidate list) that satisfy *all*
relations in the expression.  We map them back to real indices of the
original object list.
"""
from third_party.CSVG.solver import solve


def pick_with_csvg(expression_json, boxes, top_idx, k_keep=1):
    """
    Parameters
    ----------
    expression_json : dict
        The parsed symbolic expression.
    boxes : (N,6) torch.Tensor  (cx,cy,cz,w,h,d)
    top_idx : 1-D LongTensor (k,)
        Candidate indices INTO the full object list.
    k_keep : int
        How many satisfying indices to keep (top of list).

    Returns
    -------
    list[int] : indices into the *original* object list (length >= 1).
    """
    cand_boxes = boxes[top_idx].cpu().numpy()          # (k,6)
    valid_in_sub = solve(expression_json, cand_boxes)  # returns indices 0..k-1

    if not valid_in_sub:
        return top_idx[:1].tolist()                    # fallback: keep top-1

    mapped = [top_idx[i].item() for i in valid_in_sub[:k_keep]]
    return mapped
# ======================================================================
