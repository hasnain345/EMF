# backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class GatedFusion(nn.Module):
    def __init__(self, d_in_vli, d_in_tz, d_model, d_aux_rlp=24, d_aux_dcs=3):
        super().__init__()
        self.proj_vli = nn.Linear(d_in_vli, d_model)
        self.proj_tz  = nn.Linear(d_in_tz,  d_model)
        self.gater    = nn.Sequential(nn.Linear(2*d_model, d_model),
                                      nn.GELU(),
                                      nn.Linear(d_model, d_model),
                                      nn.Sigmoid())
        # lightweight residuals for RLP and DCSTR
        self.proj_rlp = nn.Linear(d_aux_rlp, d_model)
        self.proj_dcs = nn.Linear(d_aux_dcs, d_model)
        self.apply(_init)

    def forward(self, vli, tz, rlp, dcs):
        # vli [B, T, Dvli], tz [B, T, 3], rlp [B, T, 24], dcs [B, T, 3]
        V = self.proj_vli(vli)
        T = self.proj_tz(tz)
        gate = self.gater(torch.cat([V, T], dim=-1))
        fused = gate * V + (1.0 - gate) * T
        # add auxiliary channels as residual hints
        fused = fused + 0.1 * self.proj_rlp(rlp) + 0.1 * self.proj_dcs(dcs)
        return fused  # [B, T, d]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=6, dim_ff=2048, drop=0.1, attn_drop=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, drop, batch_first=True, activation='gelu')
        layer.self_attn.dropout = nn.Dropout(attn_drop)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.enc(x)
        h = self.norm(h)
        z = F.normalize(h.mean(dim=1), dim=-1)
        return z, h

class VisionTransformer(nn.Module):
    """
    Inputs:
      vli   [B, T, Dvli]  e.g., 128 per block
      tz    [B, T, 3]
      rlp   [B, T, 24]
      dcs   [B, T, 3]
      huff  [B, 528]
    """
    def __init__(self, d_vli=128, d_model=256, nhead=8, depth=6, dim_ff=2048, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.gate = GatedFusion(d_in_vli=d_vli, d_in_tz=3, d_model=d_model, d_aux_rlp=24, d_aux_dcs=3)
        # three global tokens
        self.g_dc   = nn.Linear(13*3, d_model)   # DC hist across 3 channels
        self.g_ac   = nn.Linear(163*3, d_model)  # AC hist across 3 channels
        self.g_dcsr = nn.Linear(3, d_model)      # DC sign stats per channel
        self.pos = nn.Parameter(torch.zeros(1, 2048, d_model))  # generous cap
        self.enc = TransformerEncoder(d_model, nhead, depth, dim_ff, drop, attn_drop)
        self.pre = nn.LayerNorm(d_model)
        self.apply(_init)

    def _split_huffman(self, h):
        # h : [B, 528] as [Y_DC(13), Y_AC(163), Cb_DC(13), Cb_AC(163), Cr_DC(13), Cr_AC(163)]
        Ydc  = h[:, 0:13];    Yac  = h[:, 13:176]
        Cbdc = h[:, 176:189]; Cbac = h[:, 189:352]
        Crdc = h[:, 352:365]; Crac = h[:, 365:528]
        DC = torch.cat([Ydc, Cbdc, Crdc], dim=1)    # [B, 39]
        AC = torch.cat([Yac, Cbac, Crac], dim=1)    # [B, 489]
        return DC, AC

    def forward(self, vli, tz, rlp, dcs, huff):
        B, T, _ = vli.shape
        # local fusion
        local = self.gate(vli, tz, rlp, dcs)  # [B, T, d]
        # global tokens
        DC, AC = self._split_huffman(huff)
        # DC sign stats: mean over blocks per channel from dcs [B, T, 3]
        dcsr = dcs.mean(dim=1)  # [B,3]
        gdc = self.g_dc(DC).unsqueeze(1)
        gac = self.g_ac(AC).unsqueeze(1)
        gsr = self.g_dcsr(dcsr).unsqueeze(1)
        seq = torch.cat([gdc, gac, gsr, local], dim=1)  # [B, 3+T, d]
        # light positional encoding
        seq = seq + self.pos[:, :seq.size(1), :]
        z, h = self.enc(seq)
        z = self.pre(z)
        return z  # [B, d]
