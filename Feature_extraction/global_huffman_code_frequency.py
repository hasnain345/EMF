# global_huffman_code_frequency.py
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor

# Fixed bins (DC categories 0..12, AC run-symbols 0..162)
_BIN_DCH = np.arange(0, 13, 1)     # 13 bins
_BIN_ACH = np.arange(0, 163, 1)    # 163 bins

def _hist_pair(dch, ach):
    # dch, ach are Python lists or ndarrays
    h_dc, _ = np.histogram(dch, bins=_BIN_DCH)
    h_ac, _ = np.histogram(ach, bins=_BIN_ACH)
    return h_dc.astype(np.int32), h_ac.astype(np.int32)

def global_feature(dccofY, accofY, dccofCb, accofCb, dccofCr, accofCr):
    """
    Return concatenated global histogram:
      [Y_DC(13), Y_AC(163), Cb_DC(13), Cb_AC(163), Cr_DC(13), Cr_AC(163)]
      shape = (13+163)*3 = 528
    """
    # Decode syntax-invariant sequences
    Yach, _ = jacdecColor(accofY, 'Y');   Ydch, _ = jdcdecColor(dccofY, 'Y')
    Cbach, _ = jacdecColor(accofCb, 'C'); Cbdch, _ = jdcdecColor(dccofCb, 'C')
    Crach, _ = jacdecColor(accofCr, 'C'); Crdch, _ = jdcdecColor(dccofCr, 'C')

    # Histograms per channel
    Y_dc,  Y_ac  = _hist_pair(Ydch,  Yach)
    Cb_dc, Cb_ac = _hist_pair(Cbdch, Cbach)
    Cr_dc, Cr_ac = _hist_pair(Crdch, Crach)

    # Concatenate in a fixed order
    global_vec = np.concatenate([Y_dc, Y_ac, Cb_dc, Cb_ac, Cr_dc, Cr_ac], axis=0).astype(np.int32)
    return global_vec
