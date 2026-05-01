from torch.utils.data import Dataset
import torch
import numpy as np

class EMFSupervised(Dataset):
    """
    Supervised dataset yielding all EMF features.
      vli   : [B, Dvli]  (per-block VLI bit-lengths concatenated across channels)
      tzcpb : [B, 3]     (per-block AC zero counts for Y, Cb, Cr)
      rlp   : [B, 24]    (per-block run-length profile, Lmax=8 per channel => 3*8)
      dcstr : [B, 3]     (per-block DC sign-flip indicators per channel)
      huff  : [528]      (global Huffman histogram vector)
      label : int
    """
    def __init__(self, vli, huff, tzcpb, rlp, dcstr, labels, transform=None):
        self.vli = vli
        self.huff = huff
        self.tz = tzcpb
        self.rlp = rlp
        self.dcs = dcstr
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vli  = self.vli[idx].astype('float32')   # [B, Dvli]
        tz   = self.tz[idx].astype('float32')    # [B, 3]
        rlp  = self.rlp[idx].astype('float32')   # [B, 24]
        dcs  = self.dcs[idx].astype('float32')   # [B, 3]
        huff = self.huff[idx].astype('float32')  # [528]
        label = int(self.labels[idx])

        # Apply augmentation only to VLI stream (others are statistics)
        if self.transform is not None:
            vli = self.transform(vli)
        else:
            vli = torch.tensor(vli, dtype=torch.float32)

        # Convert the rest to tensors
        tz   = torch.tensor(tz,   dtype=torch.float32)
        rlp  = torch.tensor(rlp,  dtype=torch.float32)
        dcs  = torch.tensor(dcs,  dtype=torch.float32)
        huff = torch.tensor(huff, dtype=torch.float32)
        label= torch.tensor(label, dtype=torch.long)

        return vli, tz, rlp, dcs, huff, label
