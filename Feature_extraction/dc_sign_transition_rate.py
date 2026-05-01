# Feature_extraction/dc_sign_transition_rate.py
import numpy as np
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor

def _sign(x: int) -> int:
    if x > 0: return 1
    if x < 0: return -1
    return 0

def dcstr_each_component(dccof, component_type, row, col, N=8):
    """
    Per-block DC sign-transition indicator for one component.
    Returns: (num_blocks, 1) int16 with 1 if sign flipped vs previous block, else 0.
    """
    _, dcarr = jdcdecColor(dccof, component_type, 'E')
    dcarr = np.asarray(dcarr, dtype=np.int32)

    num_blocks = int(row * col / (N * N))
    assert dcarr.shape[0] == num_blocks, "DC diff count must match number of blocks"

    out = np.zeros((num_blocks, 1), dtype=np.int16)
    if num_blocks == 0:
        return out

    prev_s = _sign(int(dcarr[0]))
    # First block has no "previous"; define 0 flip by convention
    out[0, 0] = 0

    for i in range(1, num_blocks):
        s = _sign(int(dcarr[i]))
        out[i, 0] = 1 if (s * prev_s) == -1 else 0
        prev_s = s if s != 0 else prev_s  # keep previous non-zero sign for stability

    return out

def dcstr_all_components(dcallY, dcallCb, dcallCr, img_size):
    """
    Concatenate per-block DC sign flip indicators across Y, Cb, Cr:
      shape = (num_blocks, 3), dtype=int16
    """
    row, col = int(img_size[0]), int(img_size[1])
    y  = dcstr_each_component(dcallY, 'Y',  row, col, N=8)
    cb = dcstr_each_component(dcallCb, 'Cb', row, col, N=8)
    cr = dcstr_each_component(dcallCr, 'Cr', row, col, N=8)
    return np.concatenate([y, cb, cr], axis=1).astype(np.int16, copy=False)
