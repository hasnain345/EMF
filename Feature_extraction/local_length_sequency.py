# local_length_sequency.py
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor

def _cat_length(val: int) -> int:
    """JPEG-like category length: 0 -> 0, else floor(log2(abs(v))) + 1."""
    if val == 0:
        return 0
    v = abs(int(val))
    # fastest integer bit-length for positive ints
    return int(v.bit_length())

def length_sequence_each_component(accof, dccof, row, col, comp_type, N=8):
    """
    For each 8x8 block, build a 64-length sequence of per-coefficient category lengths.
    For chroma, return first 32 positions as in your original pipeline.
    """
    _, acarr = jacdecColor(accof, comp_type)
    _, dcarr = jdcdecColor(dccof, comp_type, 'E')
    acarr = np.asarray(acarr, dtype=np.int32)
    dcarr = np.asarray(dcarr, dtype=np.int32)

    num_blocks = int(row * col / (N * N))
    ret = np.zeros((num_blocks, 64), dtype=np.int16)

    # Find end-of-block markers for each block in the flattened AC stream
    eob_idx = np.where(acarr == 999)[0]
    ac_ptr = 0
    dc_ptr = 0
    blk = 0

    for _m in range(0, row, N):
        for _n in range(0, col, N):
            # slice AC for this block
            ac_blk = acarr[ac_ptr: eob_idx[blk]]
            ac_ptr = eob_idx[blk] + 1
            # prepend DC diff
            acc = np.concatenate([[dcarr[dc_ptr]], ac_blk], axis=0)
            dc_ptr += 1
            # pad to 64
            if acc.shape[0] < 64:
                acc = np.pad(acc, (0, 64 - acc.shape[0]), constant_values=0)

            # map to category lengths
            # vectorize bit_length safely
            cat = np.fromiter((_cat_length(int(v)) for v in acc), dtype=np.int16, count=64)
            ret[blk, :] = cat
            blk += 1

    if comp_type == 'Y':
        return ret
    else:
        return ret[:, :32]

def length_sequence_all_component(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size):
    row, col = int(img_size[0]), int(img_size[1])
    featY  = length_sequence_each_component(acallY,  dcallY,  row, col, 'Y')
    featCb = length_sequence_each_component(acallCb, dcallCb, row, col, 'Cb')
    featCr = length_sequence_each_component(acallCr, dcallCr, row, col, 'Cr')
    featAll = np.concatenate([featY, featCb, featCr], axis=1).astype(np.int16, copy=False)
    return featAll
