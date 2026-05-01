# owner_encryption.py
import os
import glob
import copy
import datetime
import multiprocessing as mul
import numpy as np
import cv2
from typing import Tuple

from encryption_utils import (
    derive_image_key,
    prg_bytes,
    yates_shuffle,
    loadImageFiles,
)
from JPEG.rgbandycbcr import rgb2ycbcr
from JPEG.jdcencColor import jdcencColor
from JPEG.jacencColor import jacencColor
from JPEG.zigzag import zigzag
from JPEG.Quantization import Quantization

from cipherimageRgbGenerate import Gen_cipher_images

# -----------------------------
# Hyperparameters
# -----------------------------
QF = 100
BLOCK_SIZE = 8
ENABLE_BLOCK_PERMUTE = False  # EMF default: False. Turn True only for ablations.
PERMUTE_SEED = 12345          # used if ENABLE_BLOCK_PERMUTE

# I/O
PLAIN_GLOB = '../data/plainimages/*.jpg'
BIT_OUT_DIR = '../data/cipherimages'   # stores .npy bit payloads per image

os.makedirs(BIT_OUT_DIR, exist_ok=True)


def _pad_to_block(img: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    row, col, _ = img.shape
    ycbcr = rgb2ycbcr(img).astype(np.float32)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    # right pad
    col_pad = int(BLOCK_SIZE * np.ceil(col / BLOCK_SIZE) - col)
    if col_pad > 0:
        Y = np.c_[Y, np.tile(Y[:, -1:], (1, col_pad))]
        Cb = np.c_[Cb, np.tile(Cb[:, -1:], (1, col_pad))]
        Cr = np.c_[Cr, np.tile(Cr[:, -1:], (1, col_pad))]
    # bottom pad
    row_pad = int(BLOCK_SIZE * np.ceil(row / BLOCK_SIZE) - row)
    if row_pad > 0:
        Y = np.r_[Y, np.tile(Y[-1:, :], (row_pad, 1))]
        Cb = np.r_[Cb, np.tile(Cb[-1:, :], (row_pad, 1))]
        Cr = np.r_[Cr, np.tile(Cr[-1:, :], (row_pad, 1))]

    R = int(BLOCK_SIZE * np.ceil(row / BLOCK_SIZE))
    C = int(BLOCK_SIZE * np.ceil(col / BLOCK_SIZE))
    return R, C, Y, Cb, Cr


def _make_channel_keystreams(img_key: bytes, est_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-channel keystream byte arrays of length >= est_len.
    The keystream is consumed by jdcencColor/jacencColor exactly as before.
    """
    # small distinct nonces per channel
    nY  = b"Y"
    nCb = b"Cb"
    nCr = b"Cr"
    kY  = prg_bytes(img_key, nY,  est_len, info=b"DCAC")
    kCb = prg_bytes(img_key, nCb, est_len, info=b"DCAC")
    kCr = prg_bytes(img_key, nCr, est_len, info=b"DCAC")
    return kY.astype(np.uint8), kCb.astype(np.uint8), kCr.astype(np.uint8)


def _encrypt_component(comp: np.ndarray, ks: np.ndarray, comp_type: str, R: int, C: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Syntax-preserving pipeline: DCT -> Quantization -> DC/AC entropy encode with magnitude masking via keystream.
    Returns concatenated DC and AC bit symbols as int8 arrays.
    """
    # collect 8x8 blocks
    blocks = []
    for m in range(0, R, BLOCK_SIZE):
        for n in range(0, C, BLOCK_SIZE):
            blk = comp[m:m + BLOCK_SIZE, n:n + BLOCK_SIZE] - 128.0
            blocks.append(blk)
    blocks = np.stack(blocks, axis=0)  # [B, 8, 8]

    # optional deterministic block permutation for ablation only
    if ENABLE_BLOCK_PERMUTE:
        order = yates_shuffle(list(range(len(blocks))), seed=PERMUTE_SEED)
        blocks = blocks[np.array(order)]

    # entropy encode each block with masking handled inside jdcencColor/jacencColor via keystream consumption
    dccof = []
    accof = []
    prev_dc = None
    ks_ptr = 0  # simple pointer into keystream bytes

    for i in range(len(blocks)):
        t = cv2.dct(blocks[i])
        tq = Quantization(t, type=comp_type)  # QF used inside Quantization if needed

        # DC differential
        if prev_dc is None:
            dc_val = tq[0, 0]
            dc_diff = dc_val
            prev_dc = dc_val
        else:
            dc_diff = tq[0, 0] - prev_dc
            prev_dc = tq[0, 0]

        # jdcencColor returns (num_bytes_consumed, dc_component_symbols)
        key_used, dc_component = jdcencColor(dc_diff, comp_type, ks[ks_ptr:])
        ks_ptr += int(key_used)
        dccof.append(dc_component)

        # AC zigzag up to EOB, then encode with jacencColor
        zz = zigzag(tq)
        eobi = 0
        for j in range(63, -1, -1):
            if zz[j] != 0:
                eobi = j
                break
        if eobi == 0:
            ac_seq = np.array([999], dtype=np.int32)
        else:
            ac_seq = np.concatenate([zz[1:eobi + 1].astype(np.int32), np.array([999], dtype=np.int32)])
        key_used, ac_component = jacencColor(ac_seq, comp_type, ks[ks_ptr:])
        ks_ptr += int(key_used)
        accof.append(ac_component)

    dccof = np.concatenate([np.atleast_1d(x) for x in dccof]).astype(np.int8, copy=False)
    accof = np.concatenate([np.atleast_1d(x) for x in accof]).astype(np.int8, copy=False)
    return dccof, accof


def process_single_image(image_path: str):
    # 1) Load RGB image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[warn] could not read {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2) Padding to 8x8 grid and split channels
    R, C, Y, Cb, Cr = _pad_to_block(img)

    # 3) Per-image key rotation: derive a 256-bit key from a unique seed
    # A good default is filename + timestamp; here we just use the filename
    img_name = os.path.basename(image_path)
    img_key = derive_image_key(f"{img_name}")

    # 4) Rough upper bound for keystream length consumption per component
    #    Empirical over-estimate keeps logic simple.
    est_blocks = int((R * C) // (BLOCK_SIZE * BLOCK_SIZE))
    est_len = int(6 * est_blocks + 4096)  # generous pad

    ksY, ksCb, ksCr = _make_channel_keystreams(img_key, est_len)

    # 5) Entropy encode each component with magnitude masking via keystream
    dccY, accY = _encrypt_component(Y,  ksY,  "Y",  R, C)
    dccCb, accCb = _encrypt_component(Cb, ksCb, "Cb", R, C)
    dccCr, accCr = _encrypt_component(Cr, ksCr, "Cr", R, C)

    # 6) Persist ciphertext-side payload for later feature extraction or reconstruction
    payload = {
        "dccofY": dccY,  "accofY": accY,
        "dccofCb": dccCb, "accofCb": accCb,
        "dccofCr": dccCr, "accofCr": accCr,
        "size": (int(R), int(C)),
        "qf": QF,
        "image": img_name,
    }
    out_path = os.path.join(BIT_OUT_DIR, f"{os.path.splitext(img_name)[0]}.npy")
    np.save(out_path, payload, allow_pickle=True)
    print(f"[ok] {img_name} encrypted and saved -> {out_path}")


def main():
    imageFiles = loadImageFiles(PLAIN_GLOB)
    if len(imageFiles) == 0:
        print(f"[warn] no images found at {PLAIN_GLOB}")
        return
    print(datetime.datetime.now())
    with mul.Pool(processes=max(1, os.cpu_count() // 2)) as pool:
        pool.map(process_single_image, imageFiles)
    print(datetime.datetime.now())


if __name__ == "__main__":
    main()
