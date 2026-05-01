# encryption_utils.py
import os
import math
import glob
import copy
import hashlib
from typing import Tuple
import numpy as np
import cv2

# -----------------------------
# Per-image PRG and keying
# -----------------------------

def derive_image_key(seed: str) -> bytes:
    """
    Derive a 256-bit per-image key from an arbitrary seed string.
    Use a unique seed per image, e.g., filename + timestamp + user secret.
    """
    if not isinstance(seed, (str, bytes)):
        seed = str(seed)
    if isinstance(seed, str):
        seed = seed.encode("utf-8")
    return hashlib.sha256(seed).digest()  # 32 bytes


def prg_bytes(key: bytes, nonce: bytes, nbytes: int, info: bytes = b"") -> np.ndarray:
    """
    Simple SHA-256 counter-mode PRG.
    Returns nbytes pseudorandom bytes as uint8 ndarray, deterministic given (key, nonce, info).
    This is sufficient for research code and reproducibility.
    """
    assert isinstance(key, (bytes, bytearray)) and len(key) == 32, "key must be 32 bytes from derive_image_key"
    if not isinstance(nonce, (bytes, bytearray)):
        nonce = bytes(str(nonce), "utf-8")
    if not isinstance(info, (bytes, bytearray)):
        info = bytes(str(info), "utf-8")

    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        h = hashlib.sha256()
        # domain separation: key || nonce || info || counter
        h.update(key)
        h.update(b"|N|")
        h.update(nonce)
        h.update(b"|I|")
        h.update(info)
        h.update(b"|C|")
        h.update(counter.to_bytes(8, "big"))
        out.extend(h.digest())
        counter += 1
    return np.frombuffer(bytes(out[:nbytes]), dtype=np.uint8)


# -----------------------------
# Optional deterministic block shuffle for ablations
# -----------------------------

def yates_shuffle(indices, seed: int):
    """
    Deterministic Fisher–Yates using NumPy RNG. Keeps behavior stable across runs.
    EMF uses syntax-preserving magnitude masking; block shuffle is NOT required
    and should remain disabled by default at call sites.
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    p = list(indices)
    rng.shuffle(p)
    return p


# -----------------------------
# Metrics and I/O helpers
# -----------------------------

def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    mse = np.mean((ref_data - target_data) ** 2.0)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def npcr_uaci(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
    """
    Compute NPCR and UACI on uint8 images of the same shape.
    """
    assert img1.shape == img2.shape
    diff = img1.astype(np.int16) - img2.astype(np.int16)
    H, W = img1.shape[:2]
    npcr = (np.count_nonzero(diff) / (H * W)) * 100.0
    uaci = (np.mean(np.abs(diff)) / 255.0) * 100.0
    return npcr, uaci


def loadImgSizes(path):
    return np.load(path, allow_pickle=True)


def loadEncBit(path):
    return np.load(path, allow_pickle=True)


def loadImageFiles(srcFiles):
    return glob.glob(srcFiles)


def loadImageSet(srcFiles):
    imageFiles = loadImageFiles(srcFiles)
    plainimages = []
    for imageName in imageFiles:
        img = cv2.imread(imageName)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plainimages.append(img)
    return plainimages
