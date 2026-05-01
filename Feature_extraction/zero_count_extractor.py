# zero_count_extractor.py
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor

def zero_count_each_component(accof, dccof, row, col, component_type, N=8):
    """
    Count AC zeros per block (exclude DC position).
    Returns (num_blocks, 1) int16 array.
    """
    _, acarr = jacdecColor(accof, component_type)
    _, dcarr = jdcdecColor(dccof, component_type, 'E')
    acarr = np.asarray(acarr, dtype=np.int32)
    dcarr = np.asarray(dcarr, dtype=np.int32)

    num_blocks = int(row * col / (N * N))
    out = np.zeros((num_blocks, 1), dtype=np.int16)

    EOB = np.where(acarr == 999)[0]
    ac_ptr = 0
    dc_ptr = 0
    blk = 0

    for _m in range(0, row, N):
        for _n in range(0, col, N):
            ac_blk = acarr[ac_ptr: EOB[blk]]
            ac_ptr = EOB[blk] + 1
            dcval  = dcarr[dc_ptr]; dc_ptr += 1

            # build 64-length stream [DC, AC...], pad
            acc = np.concatenate([[dcval], ac_blk], axis=0)
            if acc.shape[0] < 64:
                acc = np.pad(acc, (0, 64 - acc.shape[0]), constant_values=0)

            # count AC zeros only
            out[blk, 0] = int(np.sum(acc[1:] == 0))
            blk += 1

    return out

def zero_count_all_components(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size):
    row, col = int(img_size[0]), int(img_size[1])
    y  = zero_count_each_component(acallY,  dcallY,  row, col, 'Y')
    cb = zero_count_each_component(acallCb, dcallCb, row, col, 'Cb')
    cr = zero_count_each_component(acallCr, dcallCr, row, col, 'Cr')
    return np.concatenate([y, cb, cr], axis=1).astype(np.int16, copy=False)
