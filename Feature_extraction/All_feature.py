# All_feature.py (patched additions are marked >>>)
import os
import glob
import datetime
import multiprocessing as mul
from tqdm import tqdm
import numpy as np

from Feature_extraction.global_huffman_code_frequency import global_feature
from Feature_extraction.local_length_sequency import length_sequence_all_component
from Feature_extraction.zero_count_extractor import zero_count_all_components
# >>> new imports:
from Feature_extraction.run_length_profile import rlp_all_components
from Feature_extraction.dc_sign_transition_rate import dcstr_all_components

from Encryption_algorithm.encryption_utils import loadEncBit

# -----------------------------
# Output roots (edit these once)
# -----------------------------
ROOT_SAVE = "C:/Users/AI/Desktop/FinalEViT/data/features"
SAVE_LOCAL = os.path.join(ROOT_SAVE, "difffeature_matrix")
SAVE_GLOBAL = os.path.join(ROOT_SAVE, "huffman_feature")
SAVE_ZERO  = os.path.join(ROOT_SAVE, "zero_count")
# >>> new output dirs:
SAVE_RLP   = os.path.join(ROOT_SAVE, "rlp_hist")
SAVE_DCSTR = os.path.join(ROOT_SAVE, "dc_sign_flip")

for d in (SAVE_LOCAL, SAVE_GLOBAL, SAVE_ZERO, SAVE_RLP, SAVE_DCSTR):
    os.makedirs(d, exist_ok=True)

def _basename_noext(p):
    return os.path.splitext(os.path.basename(p))[0]

def _save_numpy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def process_one(bit_path):
    payload = loadEncBit(bit_path).item()

    dY, aY = payload['dccofY'],  payload['accofY']
    dCb, aCb = payload['dccofCb'], payload['accofCb']
    dCr, aCr = payload['dccofCr'], payload['accofCr']
    img_size = payload['size']
    stem = _basename_noext(bit_path)

    # ----- Local VLI bit-length sequence -----
    local_feat = length_sequence_all_component(dY, aY, dCb, aCb, dCr, aCr, img_size)
    _save_numpy(os.path.join(SAVE_LOCAL, f"{stem}.npy"), local_feat)

    # ----- Global Huffman frequency -----
    global_feat = global_feature(dY, aY, dCb, aCb, dCr, aCr)
    _save_numpy(os.path.join(SAVE_GLOBAL, f"{stem}.npy"), global_feat)

    # ----- TZCPB (AC zero counts per block) -----
    zero_feat = zero_count_all_components(dY, aY, dCb, aCb, dCr, aCr, img_size)
    _save_numpy(os.path.join(SAVE_ZERO, f"{stem}.npy"), zero_feat)

    # >>> ----- RLP (run-length profiles) -----
    rlp_feat = rlp_all_components(dY, aY, dCb, aCb, dCr, aCr, img_size, Lmax=8)
    _save_numpy(os.path.join(SAVE_RLP, f"{stem}.npy"), rlp_feat)

    # >>> ----- DC sign transition per block -----
    dcstr_feat = dcstr_all_components(dY, dCb, dCr, img_size)
    _save_numpy(os.path.join(SAVE_DCSTR, f"{stem}.npy"), dcstr_feat)

    return stem

def main(bit_glob):
    files = glob.glob(bit_glob)
    if len(files) == 0:
        print(f"[warn] no bit payloads found for pattern: {bit_glob}")
        return

    print(datetime.datetime.now())
    with mul.Pool(processes=max(1, os.cpu_count() // 2)) as pool:
        for _ in tqdm(pool.imap_unordered(process_one, files), total=len(files)):
            pass
    print('✓ Feature extraction finished.')
    print(datetime.datetime.now())

if __name__ == '__main__':
    BIT_GLOB = 'C:/Users/AI/Desktop/FinalEViT/data/cipherimages/*.npy'
    main(BIT_GLOB)
