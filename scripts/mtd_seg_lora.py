import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import AutoModelForSemanticSegmentation
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtd_data import load_items, make_splits  # noqa: E402

OUT = ROOT / "out" / "mtd_seg"
OUT.mkdir(parents=True, exist_ok=True)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 384


def group_of(filename):
    return filename.split("_")[0]


def make_group_splits(items, split_seed=0):
    """Leave-one-session-out split: test/val are whole expN groups."""
    groups = sorted({group_of(it["filename"]) for it in items if group_of(it["filename"]) != "exp0"})
    rng = np.random.RandomState(split_seed)
    rng.shuffle(groups)
    test_g, val_g = groups[0], groups[1]
    train_g = set(groups[2:]) | {"exp0"}
    splits = {"train": [], "val": [], "test": []}
    for it in items:
        g = group_of(it["filename"])
        if g == test_g:
            splits["test"].append(it)
        elif g == val_g:
            splits["val"].append(it)
        else:
            splits["train"].append(it)
    print(f"group split test={test_g} val={val_g} train={sorted(train_g)}")
    for k in splits:
        print(k, len(splits[k]), {c: sum(1 for x in splits[k] if x["label"] == c) for c in range(6)})
    return splits


class MTDSeg(Dataset):
    def __init__(self, items, train=False):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["path"]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        mask_path = it.get("mask_path") or it["path"].with_suffix(".png")
        mask = Image.open(mask_path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        x = T.functional.to_tensor(img)
        m = np.array(mask)
        thr = 0 if m.max() <= 1 else 127
        y = torch.from_numpy((m > thr).astype(np.float32))
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[1])
            # brightness / contrast jitter for grayscale industrial images
            if torch.rand(1).item() < 0.5:
                x = torch.clamp(x * (0.85 + 0.3 * torch.rand(1)), 0, 1)
            if torch.rand(1).item() < 0.5:
                m = x.mean()
                x = torch.clamp((x - m) * (0.85 + 0.3 * torch.rand(1)) + m, 0, 1)
        x = T.functional.normalize(x, MEAN, STD)
        return x, y


class ConvHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        return self.block(x)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2)) + eps
    den = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


def metrics(logits, target):
    pred = (torch.sigmoid(logits[:, 1]) > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    iou = inter / max(union, 1e-6)
    dice = 2 * inter / max(pred.sum().item() + target.sum().item(), 1e-6)
    acc = (pred == target).float().mean().item()
    return iou, dice, acc


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


TARGET_SETS = {
    "qv": ["q_proj", "v_proj"],
    "qv_mlp": ["q_proj", "v_proj", "fc1", "fc2"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qkvo_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
}


def build_model(mode="lora", rank=8, target_set="qv"):
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
    )
    # Replace the 150-class ADE head with a small binary head.
    model.decode_head.classifier = nn.Conv2d(256, 2, 1)
    if mode == "frozen":
        for p in model.segformer.parameters():
            p.requires_grad = False
        for p in model.decode_head.parameters():
            p.requires_grad = True
        return model, "frozen"
    if mode == "lora":
        for p in model.parameters():
            p.requires_grad = False
        cfg = LoraConfig(
            r=rank,
            lora_alpha=16,
            target_modules=TARGET_SETS[target_set],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["decode_head"],
        )
        model = get_peft_model(model, cfg)
        return model, "lora"
    if mode == "adapter":
        for p in model.parameters():
            p.requires_grad = False
        adapters = nn.ModuleList()
        for stage in model.segformer.stages:
            for blk in stage.blocks:
                dim = blk.layernorm_before.weight.shape[0]
                adapter = BottleneckAdapter(dim)
                adapters.append(adapter)
                blk.register_forward_hook(lambda m, i, o, a=adapter: a(o))
        model.adapters = adapters
        for p in model.adapters.parameters():
            p.requires_grad = True
        for p in model.decode_head.parameters():
            p.requires_grad = True
        return model, "adapter"
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model, "full"
    if mode == "bitfit":
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if n.endswith(".bias"):
                p.requires_grad = True
        for p in model.decode_head.parameters():
            p.requires_grad = True
        return model, "bitfit"
    raise ValueError(mode)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def evaluate(model, loader):
    model.eval()
    inter = union = pred_sum = target_sum = correct = total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            logits = model(pixel_values=x).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits[:, 1]) > 0.5).float()
            inter += (pred * y).sum().item()
            union += pred.sum().item() + y.sum().item() - (pred * y).sum().item()
            pred_sum += pred.sum().item()
            target_sum += y.sum().item()
            correct += (pred == y).float().sum().item()
            total += y.numel()
    iou = inter / max(union, 1e-6)
    dice = 2 * inter / max(pred_sum + target_sum, 1e-6)
    acc = correct / max(total, 1.0)
    return float(iou), float(dice), float(acc)


def run(mode, train_items, val_items, test_items, seed=0, epochs=15, batch=4,
        rank=8, target_set="qv", lr_head=1e-3, lr_lora=5e-4, lr_full=6e-5, patience_limit=3,
        save_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, tag = build_model(mode, rank=rank, target_set=target_set)
    model.cuda()
    trainable, total = count_trainable(model)
    tr = DataLoader(MTDSeg(train_items, train=True), batch_size=batch, shuffle=True, num_workers=0)
    va = DataLoader(MTDSeg(val_items), batch_size=4, shuffle=False, num_workers=0)
    te = DataLoader(MTDSeg(test_items), batch_size=4, shuffle=False, num_workers=0)

    if mode == "lora":
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "decode_head" in n]
        opt = torch.optim.AdamW([
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ], weight_decay=1e-4)
    elif mode == "full":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_full, weight_decay=1e-4)
    elif mode == "adapter":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_head, weight_decay=1e-4)

    # pos_weight for the rare defect pixels (resize masks only, no RGB decode)
    pos = 0.0
    neg = 0.0
    for it in train_items:
        mask_path = it.get("mask_path") or it["path"].with_suffix(".png")
        m = np.array(Image.open(mask_path).convert("L").resize((SIZE, SIZE), Image.NEAREST))
        thr = 0 if m.max() <= 1 else 127
        pos += float((m > thr).sum())
        neg += float((m <= thr).sum())
    pos_weight = min(20.0, max(1.0, neg / max(pos, 1.0)))
    class_weight = torch.tensor([1.0, pos_weight], device="cuda")
    ce = nn.CrossEntropyLoss(weight=class_weight)

    best = (-1.0, -1)
    best_state = None
    patience = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            logits = model(pixel_values=x).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = ce(logits, y.long()) + dice_loss(logits[:, 1].float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        viou, vdice, vacc = evaluate(model, va)
        if vdice > best[0]:
            best = (vdice, ep)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(f"  {mode} ep={ep+1} val_iou={viou:.4f} val_dice={vdice:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        if patience >= patience_limit:
            break
    model.load_state_dict(best_state)
    if save_path is not None:
        torch.save(best_state, save_path)
    tiou, tdice, tacc = evaluate(model, te)
    return {"mode": mode, "iou": tiou, "dice": tdice, "acc": tacc, "best_epoch": best[1],
            "trainable": trainable, "total": total, "train_sec": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["frozen", "lora"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--targets", type=str, default="qv", choices=list(TARGET_SETS))
    parser.add_argument("--lr-full", type=float, default=6e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--group-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--out", type=str, default=str(OUT / "seg_results.json"))
    args = parser.parse_args()

    items = load_items()
    splits = make_group_splits(items, args.split_seed) if args.group_split else make_splits(items)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"seg split train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    results = {}
    for seed in args.seeds:
        for mode in args.modes:
            key = f"{mode}_s{seed}"
            print(f"RUN {key}", flush=True)
            save_path = None
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(args.save_dir) / f"{key}.pt"
            r = run(mode, tr, va, te, seed=seed, epochs=args.epochs, batch=args.batch, rank=args.rank,
                    target_set=args.targets, lr_full=args.lr_full, patience_limit=args.patience,
                    save_path=save_path)
            results[key] = r
            print(f"RESULT {key} iou={r['iou']:.4f} dice={r['dice']:.4f} acc={r['acc']:.4f} "
                  f"best_ep={r['best_epoch']} trainable={r['trainable']}/{r['total']} sec={r['train_sec']:.0f}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
