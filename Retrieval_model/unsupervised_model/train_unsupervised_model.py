# train.py
if __name__ == '__main__':
    import argparse, os
    import torch
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch.cuda.amp import autocast, GradScaler
    import torch.nn.functional as F

    from ..dataAug import Exchange_Block, Concat_Prior_to_Last
    from .dataloader import EMFPair
    from ..utils import split_data
    from ..schedule import get_cosine_schedule_with_warmup
    from .model import ModelMoCo

    parser = argparse.ArgumentParser(description='Train EMF with gated fusion and multi-global tokens')
    args = parser.parse_args('')
    args.lr = 1e-3
    args.weight_decay = 5e-5
    args.epochs = 200
    args.type = 'Corel10-a'
    args.batch_size = 8

    # Load features
    tr_vli, te_vli, tr_huf, te_huf, tr_lbl, te_lbl, tr_tzc, te_tzc, tr_rlp, te_rlp, tr_dcs, te_dcs = split_data(args.type)

    # Augmentations operate on [B, D] arrays
    train_transform = transforms.Compose([
        Exchange_Block(0.3),
        Concat_Prior_to_Last(0.3),
    ])
    test_transform = None

    # Datasets
    train_set = EMFPair(tr_vli, tr_huf, tr_tzc, tr_rlp, tr_dcs, transform=train_transform)
    test_set  = EMFPair(te_vli, te_huf, te_tzc, te_rlp, te_dcs, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # Model and optim
    model = ModelMoCo().cuda()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scaler = GradScaler()
    scheduler = get_cosine_schedule_with_warmup(optimizer=optim, num_warmup_steps=10, num_training_steps=args.epochs)

    os.makedirs('checkpoints', exist_ok=True)

    def train_epoch(ep, start_iter=0):
        model.train()
        total, seen = 0.0, 0
        it = start_iter
        for v1, v2, t1, t2, r1, r2, d1, d2, h in tqdm(train_loader, desc=f"Epoch {ep}"):
            v1=v1.cuda(); v2=v2.cuda(); t1=t1.cuda(); t2=t2.cuda()
            r1=r1.cuda(); r2=r2.cuda(); d1=d1.cuda(); d2=d2.cuda(); h=h.cuda()
            with autocast():
                loss = model(v1, v2, t1, t2, r1, r2, d1, d2, h)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            total += loss.item() * v1.size(0); seen += v1.size(0)
            it += 1
        scheduler.step()
        return total / max(seen, 1), it

    @torch.no_grad()
    def evaluate():
        model.eval()
        feats, labs = [], torch.tensor(te_lbl, device='cuda')
        for v1, _, t1, _, r1, _, d1, _, h in tqdm(test_loader, desc="Eval"):
            z = model.encoder_q.net(v1.cuda(), t1.cuda(), r1.cuda(), d1.cuda(), h.cuda())
            z = F.normalize(z, dim=1)
            feats.append(z)
        feats = torch.cat(feats, dim=0)
        # mAP@100
        ap = []
        for i in range(len(feats)):
            sim = torch.mv(feats, feats[i])
            idx = torch.topk(sim, k=min(100, len(sim))).indices
            same = (labs[idx] == labs[i]).tolist()[1:]
            pos = 0; prec = []
            for j, m in enumerate(same, 1):
                if m:
                    pos += 1
                    prec.append(pos / j)
            ap.append(np.mean(prec) if prec else 0.0)
        print(f"Test mAP@100: {float(np.mean(ap)):.4f}")

    start_iter = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, start_iter = train_epoch(ep, start_iter)
        if ep % 25 == 0:
            torch.save({'epoch': ep, 'state_dict': model.state_dict()}, f'checkpoints/ep_{ep}.pth')
    evaluate()
