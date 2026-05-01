if __name__ == '__main__':
    import argparse, os
    import torch
    import numpy as np
    from tqdm import tqdm
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch.cuda.amp import autocast, GradScaler

    # project-local imports
    from ..schedule import get_cosine_schedule_with_warmup
    from ..dataAug import Exchange_Block, Concat_Prior_to_Last
    from ..utils import split_data
    from .dataloader import EMFSupervised
    from .model import CrossEntropyLabelSmooth, supervised_net
    from .backbone import VisionTransformer  # must implement forward(vli, tz, rlp, dcs, huff) -> [B,256]

    parser = argparse.ArgumentParser(description='Supervised EMF training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--split', type=str, default='Corel10-a')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)

    # 1) Load features (expects utils.split_data to return EMF tensors)
    tr_vli, te_vli, tr_huf, te_huf, tr_lbl, te_lbl, tr_tz, te_tz, tr_rlp, te_rlp, tr_dcs, te_dcs = split_data(args.split)

    # 2) Datasets & loaders
    train_tf = transforms.Compose([Exchange_Block(0.3), Concat_Prior_to_Last(0.3)])
    test_tf  = None
    train_ds = EMFSupervised(tr_vli, tr_huf, tr_tz, tr_rlp, tr_dcs, tr_lbl, transform=train_tf)
    test_ds  = EMFSupervised(te_vli, te_huf, te_tz, te_rlp, te_dcs, te_lbl, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 3) Model, loss, optim
    backbone = VisionTransformer()               # 256-D embedding
    net = supervised_net(net=backbone, out_dim=args.num_classes, emb_dim=256).cuda()
    ce = CrossEntropyLabelSmooth(smoothing=0.1)
    optim = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    scaler = GradScaler()
    scheduler = get_cosine_schedule_with_warmup(optimizer=optim, num_warmup_steps=20, num_training_steps=args.epochs)

    # 4) Optional resume
    start_ep = 0
    if args.resume:
        cks = [f for f in os.listdir('checkpoints') if f.startswith('supervised')]
        if cks:
            cks.sort()
            ck = torch.load(os.path.join('checkpoints', cks[-1]))
            net.load_state_dict(ck['state_dict'])
            optim.load_state_dict(ck['optimizer'])
            start_ep = int(ck.get('epoch', 0)) + 1
            print(f"[resume] {cks[-1]}")

    # 5) Train/eval loops
    def train_one_epoch(epoch):
        net.train()
        total, seen = 0.0, 0
        for vli, tz, rlp, dcs, huff, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            vli=vli.cuda(); tz=tz.cuda(); rlp=rlp.cuda(); dcs=dcs.cuda(); huff=huff.cuda(); label=label.cuda()
            with autocast():
                logits = net(vli, tz, rlp, dcs, huff, label)
                loss = ce(logits, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            total += loss.item() * vli.size(0); seen += vli.size(0)
        scheduler.step()
        return total / max(seen, 1)

    @torch.no_grad()
    def evaluate():
        net.eval()
        feats, labs = [], torch.tensor(te_lbl, device='cuda')
        for vli, tz, rlp, dcs, huff, _ in tqdm(test_loader, desc="Eval"):
            z = net(vli.cuda(), tz.cuda(), rlp.cuda(), dcs.cuda(), huff.cuda(), label=None)
            z = F.normalize(z, dim=1)
            feats.append(z)
        feats = torch.cat(feats, dim=0)
        # mAP@100
        ap = []
        for i in range(len(feats)):
            sim = torch.mv(feats, feats[i])
            k = min(100, sim.numel())
            idx = torch.topk(sim, k=k).indices
            same = (labs[idx] == labs[i]).tolist()[1:]
            pos, precs = 0, []
            for j, ok in enumerate(same, 1):
                if ok:
                    pos += 1
                    precs.append(pos / j)
            ap.append(float(np.mean(precs)) if precs else 0.0)
        mAP = float(np.mean(ap))
        print(f"[eval] mAP@100 = {mAP:.4f}")
        return mAP

    for ep in range(start_ep, args.epochs):
        tr_loss = train_one_epoch(ep)
        print(f"[train] epoch={ep} loss={tr_loss:.4f}")
        if (ep + 1) % 25 == 0:
            path = f'checkpoints/supervised_{args.split}_epoch_{ep}_loss_{tr_loss:.4f}.pth'
            torch.save({'epoch': ep, 'state_dict': net.state_dict(), 'optimizer': optim.state_dict()}, path)
            print(f"[ckpt] saved -> {path}")

    evaluate()
    torch.save({'epoch': args.epochs, 'state_dict': net.state_dict(), 'optimizer': optim.state_dict()},
               f'supervised_{args.split}_model_last.pth')
