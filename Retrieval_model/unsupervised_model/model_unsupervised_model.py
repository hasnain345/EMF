# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import VisionTransformer

class NetWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, vli, tz, rlp, dcs, huff):
        return self.net(vli, tz, rlp, dcs, huff)

class ModelMoCo(nn.Module):
    def __init__(self, dim=256, K=4096, m=0.99, T=0.1, symmetric=True, batch_size=8):
        super().__init__()
        self.K, self.m, self.T, self.symmetric = K, m, T, symmetric
        self.encoder_q = NetWrapper(VisionTransformer())
        self.encoder_k = NetWrapper(VisionTransformer())
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data); p_k.requires_grad = False
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = self.m * p_k.data + (1.0 - self.m) * p_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bsz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % bsz == 0
        self.queue[:, ptr:ptr+bsz] = keys.T
        self.queue_ptr[0] = (ptr + bsz) % self.K

    @torch.no_grad()
    def _shuffle(self, *tensors):
        idx = torch.randperm(tensors[0].shape[0], device=tensors[0].device)
        inv = torch.argsort(idx)
        return [t[idx] for t in tensors], inv
    @torch.no_grad()
    def _unshuffle(self, x, inv):
        return x[inv]

    def _contrastive(self, vli_q, vli_k, tz_q, tz_k, rlp_q, rlp_k, dcs_q, dcs_k, huff):
        q = self.encoder_q(vli_q, tz_q, rlp_q, dcs_q, huff)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            (vli_k, tz_k, rlp_k, dcs_k), inv = self._shuffle(vli_k, tz_k, rlp_k, dcs_k)
            k = self.encoder_k(vli_k, tz_k, rlp_k, dcs_k, huff)
            k = F.normalize(k, dim=1)
            k = self._unshuffle(k, inv)
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(1)
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss, k

    def forward(self, v1, v2, t1, t2, r1, r2, d1, d2, h):
        with torch.no_grad():
            self._momentum_update_key_encoder()
        if self.symmetric:
            loss1, k1 = self._contrastive(v1, v2, t1, t2, r1, r2, d1, d2, h)
            loss2, k2 = self._contrastive(v2, v1, t2, t1, r2, r1, d2, d1, h)
            loss = loss1 + loss2
            self._dequeue_and_enqueue(torch.cat([k1, k2], dim=0))
        else:
            loss, k = self._contrastive(v1, v2, t1, t2, r1, r2, d1, d2, h)
            self._dequeue_and_enqueue(k)
        return loss
