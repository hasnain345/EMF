# Feature_extraction/run_length_profile.py
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor

def _nonzero_run_hist(ac_block_vals: np.ndarray, Lmax: int) -> np.ndarray:
    """
    Build a histogram of lengths of consecutive non-zero AC values in a *single block* sequence.
    Lengths > Lmax are clipped into the last bin.
    """
    h = np.zeros(Lmax, dtype=np.int16)
    run = 0
    for v in ac_block_vals:
        if v != 0:
            run += 1
        else:
            if run > 0:
                idx = min(run, Lmax) - 1
                h[idx] += 1
                run = 0
    # flush tail
    if run > 0:
        idx = min(run, Lmax) - 1
        h[idx] += 1
    return h

def rlp_each_component(accof, dccof, row, col, component_type, N=8, Lmax=8):
    """
    Per-block Run-Length Profile (RLP) histograms for one component.
    Returns: (num_blocks, Lmax) int16
    """
    # Decode syntax-invariant streams (we only need AC values)
    _, acarr = jacdecColor(accof, component_type)   # flattened AC per image, with EOB markers (=999)
    _, dcarr = jdcdecColor(dccof, component_type, 'E')
    acarr = np.asarray(acarr, dtype=np.int32)
    dcarr = np.asarray(dcarr, dtype=np.int32)

    num_blocks = int(row * col / (N * N))
    out = np.zeros((num_blocks, Lmax), dtype=np.int16)

    # Find per-block EOB positions in the flattened AC stream
    EOB = np.where(acarr == 999)[0]
    assert EOB.shape[0] == num_blocks, "EOB count must match number of blocks"
    ac_ptr = 0

    # Iterate blocks in MCU raster order
    for blk in range(num_blocks):
        ac_block = acarr[ac_ptr:EOB[blk]]
        ac_ptr = EOB[blk] + 1
        # Build run length histogram of *non-zero* sequences only
        h = _nonzero_run_hist(ac_block, Lmax=Lmax)
        out[blk, :] = h

    return out  # (num_blocks, Lmax)

def rlp_all_components(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size, Lmax=8):
    """
    Concatenate per-block RLP histograms across Y, Cb, Cr:
      shape = (num_blocks, Lmax*3), dtype=int16
    """
    row, col = int(img_size[0]), int(img_size[1])
    rY  = rlp_each_component(acallY,  dcallY,  row, col, 'Y',  N=8, Lmax=Lmax)
    rCb = rlp_each_component(acallCb, dcallCb, row, col, 'Cb', N=8, Lmax=Lmax)
    rCr = rlp_each_component(acallCr, dcallCr, row, col, 'Cr', N=8, Lmax=Lmax)
    return np.concatenate([rY, rCb, rCr], axis=1).astype(np.int16, copy=False)
