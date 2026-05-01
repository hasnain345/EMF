# cipherimageRgbGenerate.py
import os
import numpy as np
import cv2

from JPEG.jacdecColorHuffman import jacdecColor
from JPEG.jdcdecColorHuffman import jdcdecColor
from JPEG.invzigzag import invzigzag
from JPEG.rgbandycbcr import ycbcr2rgb
from JPEG.DCT import idctJPEG
from JPEG.Quantization import iQuantization

BLOCK = 8

def _deentropy(acall, dcall, rows, cols, comp_type, N=8, QF=100):
    """
    Inverse entropy decode using only syntax-invariant elements.
    Rebuilds per-block coefficient arrays without accessing plaintext pixels.
    """
    _, acarr = jacdecColor(acall, comp_type)
    _, dcarr = jdcdecColor(dcall, comp_type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)

    rows = int(rows)
    cols = int(cols)

    EOB = np.where(acarr == 999)[0]
    e_ptr = 0
    blk_idx = 0
    xq = np.zeros([rows, cols], dtype=np.float32)

    for m in range(0, rows, N):
        for n in range(0, cols, N):
            # slice AC sequence for this block
            ac = acarr[e_ptr:EOB[blk_idx]]
            e_ptr = EOB[blk_idx] + 1
            blk_idx += 1

            # prepend DC and pad to 64
            acc = np.concatenate([np.atleast_1d(dcarr[blk_idx - 1]), ac])
            if acc.shape[0] < 64:
                acc = np.pad(acc, (0, 64 - acc.shape[0]))

            # inverse zigzag, de-quantize, inverse DCT
            coeff = invzigzag(acc, N, N)
            coeff = iQuantization(coeff, QF, comp_type)
            spatial = idctJPEG(coeff) + 128.0
            xq[m:m + N, n:n + N] = spatial

    return xq


def Gen_cipher_images(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size, src_path, out_dir="../data/cipherimages"):
    """
    Reconstruct a decoder-compatible RGB view from encrypted entropy streams.
    This respects the EMF syntax-preserving design and does NOT reveal plaintext magnitudes.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows, cols = int(img_size[0]), int(img_size[1])
    Y  = _deentropy(acallY,  dcallY,  rows, cols, 'Y')
    Cb = _deentropy(acallCb, dcallCb, rows, cols, 'U')
    Cr = _deentropy(acallCr, dcallCr, rows, cols, 'V')

    ycbcr = np.dstack([Y, Cb, Cr]).astype(np.float32)
    rgb = ycbcr2rgb(np.clip(np.round(ycbcr), 0, 255).astype(np.uint8))

    # OpenCV expects BGR for imwrite
    bgr = cv2.merge([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]])

    fname = os.path.basename(src_path)
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"[ok] cipher-compatible JPEG written -> {out_path}")
