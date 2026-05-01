import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- ArcFace margin head (your original) -------------------------------------

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=32, m=0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m

        cos_th_m = torch.where(cos_th > self.th.to(cos_th.device), 
                               cos_th_m, 
                               cos_th - self.mm.to(cos_th.device))

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros_like(cos_th)
        labels = labels.long().to(cos_th.device)
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


# ---- Loss helpers (kept for compatibility) -----------------------------------

def normalize(x, axis=-1):
    return 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()


# ---- Supervised network wrapping the EMF backbone ----------------------------

class supervised_net(nn.Module):
    """
    Wraps a ViT-style backbone that consumes EMF features and returns a 256-D embedding.
    Then applies ArcFace margin classification during training; at eval, returns embeddings.
    """
    def __init__(self, net, out_dim=100, emb_dim=256, arc_s=32, arc_m=0.4):
        super().__init__()
        self.net = net                     # expects (vli, tz, rlp, dcs, huff) -> [B, emb_dim]
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.margin = ArcModule(in_features=emb_dim, out_features=out_dim, s=arc_s, m=arc_m)

    def forward(self, vli, tzcpb, rlp, dcs, huffman, label=None):
        # backbone produces an L2-normalized embedding
        emb = self.net(vli, tzcpb, rlp, dcs, huffman)   # [B, emb_dim]
        if label is not None:
            emb_n = F.normalize(emb, dim=1)
            logits = self.margin(emb_n, label)
            return logits
        else:
            return emb
