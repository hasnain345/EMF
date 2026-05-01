# utils.py
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# Root directories where per-image .npy are saved by All_feature.py
ROOT = 'C:/Users/AI/Desktop/FinalEViT/data/features'
DIR_VLI   = os.path.join(ROOT, 'difffeature_matrix')     # per-image, shape [B, 128] typically
DIR_TZCPB = os.path.join(ROOT, 'zero_count')             # per-image, shape [B, 3]
DIR_RLP   = os.path.join(ROOT, 'rlp_hist')               # per-image, shape [B, 24]
DIR_DCSTR = os.path.join(ROOT, 'dc_sign_flip')           # per-image, shape [B, 3]
DIR_HUFF  = os.path.join(ROOT, 'huffman_feature')        # per-image, shape [528]

def _load_dir_2d(dir_path):
    """Load variable-length per-image [B, D] arrays, left-pad to max B, stack to [N, Bmax, D]."""
    files = sorted(glob.glob(os.path.join(dir_path, '*.npy')))
    arrays, max_B, D = [], 0, None
    for f in files:
        a = np.load(f, allow_pickle=True)
        if a.ndim == 1:
            # This should not happen for local features but handle defensively
            a = a[None, :]
        if D is None:
            D = a.shape[1]
        max_B = max(max_B, a.shape[0])
        arrays.append(a.astype(np.float32))
    if not arrays:
        raise FileNotFoundError(f'No .npy files in {dir_path}')
    padded = []
    for a in arrays:
        pad_len = max_B - a.shape[0]
        if pad_len > 0:
            a = np.pad(a, ((0, pad_len), (0, 0)), mode='constant')
        padded.append(a)
    return np.stack(padded, axis=0)  # [N, Bmax, D]

def _load_dir_1d(dir_path):
    """Load fixed-size per-image global vectors to [N, D]."""
    files = sorted(glob.glob(os.path.join(dir_path, '*.npy')))
    arrays = []
    for f in files:
        a = np.load(f, allow_pickle=True).astype(np.float32)
        a = a.reshape(-1)  # ensure 1D
        arrays.append(a)
    if not arrays:
        raise FileNotFoundError(f'No .npy files in {dir_path}')
    return np.stack(arrays, axis=0)  # [N, D]

def split_data(split_type='Corel10-a'):
    """
    Returns:
        train_vli [Ntr, B, Dvli], test_vli [Nte, B, Dvli]
        train_huf [Ntr, Dh],     test_huf [Nte, Dh]
        train_lbl [Ntr],         test_lbl [Nte]
        train_tzc [Ntr, B, 3],   test_tzc [Nte, B, 3]
        train_rlp [Ntr, B, 24],  test_rlp [Nte, B, 24]
        train_dcs [Ntr, B, 3],   test_dcs [Nte, B, 3]
    """
    vli   = _load_dir_2d(DIR_VLI)
    tzcpb = _load_dir_2d(DIR_TZCPB)
    rlp   = _load_dir_2d(DIR_RLP)
    dcstr = _load_dir_2d(DIR_DCSTR)
    huff  = _load_dir_1d(DIR_HUFF)

    N = vli.shape[0]
    if not (tzcpb.shape[0] == rlp.shape[0] == dcstr.shape[0] == huff.shape[0] == N):
        raise ValueError('Feature counts mismatch across directories')

    # labels: 100 classes x 100 images, in filename order; fallback to round-robin if unknown
    labels = np.array([i for i in range(100) for _ in range(N // 100 or 1)], dtype=np.int64)[:N]
    # simple reproducible split
    if split_type.lower() in ('corel10-a', 'corel10k-a', 'corel10'):
        idx = np.arange(N)
        Ntr = int(N * 0.7)
        tr_idx, te_idx = idx[:Ntr], idx[Ntr:]
    else:
        tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.3, stratify=labels, random_state=2025)

    def take(x, ids):
        return x[ids]

    return (
        take(vli, tr_idx), take(vli, te_idx),
        take(huff, tr_idx), take(huff, te_idx),
        labels[tr_idx].tolist(), labels[te_idx].tolist(),
        take(tzcpb, tr_idx), take(tzcpb, te_idx),
        take(rlp, tr_idx),   take(rlp, te_idx),
        take(dcstr, tr_idx), take(dcstr, te_idx),
    )
