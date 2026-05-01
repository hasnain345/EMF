# dataloader.py
from torch.utils.data import Dataset
import torch
import numpy as np

class EMFPair(Dataset):
    """
    Returns two views for MoCo:
      vli1, vli2: [B, Dvli]
      tz1,  tz2 : [B, 3]
      rlp1, rlp2: [B, 24]
      dcs1, dcs2: [B, 3]
      huff      : [Dh]
    Augmentations in transform are expected to operate on 2D arrays [B, D].
    """
    def __init__(self, vli, huff, tzcpb, rlp, dcstr, transform=None):
        self.vli = vli
        self.huff = huff
        self.tz = tzcpb
        self.rlp = rlp
        self.dcs = dcstr
        self.transform = transform

    def __len__(self):
        return self.huff.shape[0]

    def _apply(self, arr):
        if self.transform is None:
            return arr, arr
        a1 = self.transform(arr)
        a2 = self.transform(arr)
        return a1, a2

    def __getitem__(self, idx):
        vli  = self.vli[idx]
        tz   = self.tz[idx]
        rlp  = self.rlp[idx]
        dcs  = self.dcs[idx]
        huff = self.huff[idx]

        v1, v2 = self._apply(vli)
        t1, t2 = self._apply(tz)
        r1, r2 = self._apply(rlp)
        d1, d2 = self._apply(dcs)

        # convert to tensors
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        return to_t(v1), to_t(v2), to_t(t1), to_t(t2), to_t(r1), to_t(r2), to_t(d1), to_t(d2), to_t(huff)
