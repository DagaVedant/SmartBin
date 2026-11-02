from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "firmware"))
from distill import augment
from export_teacher import pack
from smartbin import nn as snn
from smartbin.graphnet import same_pads

DEVICE = "cuda"


def consumers_of(ops, name):
    return [op for op in ops if name in op["in"]]


def producer_of(ops, name):
    return next(op for op in ops if op["out"] == name)


def prune_blocks(ops, ratio: float, multiple: int = 8):
    ops = copy.deepcopy(ops)
    for dw in [op for op in ops if op["op"] == "dw"]:
        expand_act = producer_of(ops, dw["in"][0])
        if expand_act["op"] != "act":
            continue
        expand = producer_of(ops, expand_act["in"][0])
        if expand["op"] != "conv" or expand["w"].shape[0] != 1:
            continue

        act = consumers_of(ops, dw["out"])[0]
        hidden = act["out"]
        heads = consumers_of(ops, hidden)
        project = None
        gap = None
        for op in heads:
            if op["op"] == "gap":
                gap = op
            elif op["op"] == "mul":
                project = consumers_of(ops, op["out"])[0]
            elif op["op"] == "conv":
                project = op
        if project is None or project["op"] != "conv":
            continue

        count = dw["w"].shape[-1]
        keep_n = max(multiple, int(round(count * (1 - ratio) / multiple)) * multiple)
        if keep_n >= count:
            continue
        score = np.linalg.norm(expand["w"].reshape(-1, count), axis=0) * np.linalg.norm(
            project["w"].transpose(2, 0, 1, 3).reshape(count, -1), axis=1
        )
        keep = np.sort(np.argsort(-score)[:keep_n])

        expand["w"], expand["b"] = expand["w"][..., keep], expand["b"][keep]
        dw["w"], dw["b"] = dw["w"][..., keep], dw["b"][keep]
        project["w"] = project["w"][:, :, keep, :]
        if gap is not None:
            squeeze = consumers_of(ops, gap["out"])[0]
            squeeze["w"] = squeeze["w"][:, :, keep, :]
            relu = consumers_of(ops, squeeze["out"])[0]
            excite = consumers_of(ops, relu["out"])[0]
            excite["w"], excite["b"] = excite["w"][..., keep], excite["b"][keep]
    return ops


class GraphModule(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.specs = [{k: v for k, v in op.items() if k not in ("w", "b")} for op in ops]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.slot = {}
        for index, op in enumerate(ops):
            if "w" not in op:
                continue
            w = torch.from_numpy(op["w"])
            if op["op"] == "conv":
                w = w.permute(3, 2, 0, 1)
            elif op["op"] == "dw":
                w = w.permute(2, 0, 1).unsqueeze(1)
            self.slot[index] = len(self.weights)
            self.weights.append(nn.Parameter(w.contiguous()))
            self.biases.append(nn.Parameter(torch.from_numpy(op["b"])))

    def forward(self, x):
        x = x.float()
        env = {"input": x}
        last = {}
        for index, spec in enumerate(self.specs):
            for name in spec["in"]:
                last[name] = index
        out = None
        for index, spec in enumerate(self.specs):
            kind = spec["op"]
            args = [env[name] for name in spec["in"]]
            a = args[0]
            if kind in ("conv", "dw"):
                w, b = self.weights[self.slot[index]], self.biases[self.slot[index]]
                k = w.shape[-1]
                s = spec["stride"]
                if spec["pad"] == "same" and k > 1:
                    ph, pw = same_pads(a.shape[2], k, s), same_pads(a.shape[3], k, s)
                    a = F.pad(a, (pw[0], pw[1], ph[0], ph[1]))
                groups = w.shape[0] if kind == "dw" else 1
                out = F.conv2d(a, w, b, stride=s, groups=groups)
            elif kind == "act":
                fn = spec["fn"]
                if fn == "relu":
                    out = F.relu(a)
                elif fn == "relu6":
                    out = a.clamp(0, 6)
                elif fn == "hard_sigmoid":
                    out = (a + 3).clamp(0, 6) / 6
                else:
                    out = a * ((a + 3).clamp(0, 6) / 6)
            elif kind == "pad":
                (t, bt), (l, r) = spec["pads"]
                out = F.pad(a, (l, r, t, bt))
            elif kind == "add":
                out = args[0] + args[1]
            elif kind == "mul":
                out = args[0] * args[1]
            elif kind == "gap":
                out = a.mean(dim=(2, 3), keepdim=bool(spec["keepdims"]))
            elif kind == "rescale":
                out = a * spec["scale"] + spec["offset"]
            elif kind == "dense":
                w, b = self.weights[self.slot[index]], self.biases[self.slot[index]]
                out = a @ w + b
            elif kind == "softmax":
                out = a
            env[spec["out"]] = out
            for name in spec["in"]:
                if last.get(name) == index and name != "input":
                    env.pop(name, None)
        return out

    def to_ops(self, ops):
        result = []
        for index, op in enumerate(ops):
            op = dict(op)
            if index in self.slot:
                w = self.weights[self.slot[index]].detach().cpu()
                if op["op"] == "conv":
                    w = w.permute(2, 3, 1, 0)
                elif op["op"] == "dw":
                    w = w[:, 0].permute(1, 2, 0)
                op["w"] = w.numpy().astype(np.float32)
                op["b"] = self.biases[self.slot[index]].detach().cpu().numpy()
            result.append(op)
        return result


def load_ops(path: Path):
    network = snn.load_network(path)
    return [dict(op) for op in network.ops], network


def count_macs(ops, size=224):
    shapes = {"input": (size, size)}
    macs = 0
    for op in ops:
        h, w = shapes[op["in"][0]]
        if op["op"] in ("conv", "dw"):
            k = op["w"].shape[0]
            s = op["stride"]
            if op["pad"] == "same":
                h, w = -(-h // s), -(-w // s)
            else:
                h, w = (h - k) // s + 1, (w - k) // s + 1
            cin = op["w"].shape[2] if op["op"] == "conv" else 1
            macs += h * w * k * k * cin * op["w"].shape[-1]
        elif op["op"] == "pad":
            (t, b), (l, r) = op["pads"]
            h, w = h + t + b, w + l + r
        elif op["op"] == "gap":
            h, w = (1, 1)
        shapes[op["out"]] = (h, w)
    return macs


@torch.no_grad()
def evaluate(model, x, y, batch=128):
    model.eval()
    hits = 0
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(DEVICE).permute(0, 3, 1, 2).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        hits += (logits.argmax(1).cpu() == y[i : i + batch]).sum().item()
    return hits / len(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("firmware/models/smartbin-teacher.npz"))
    parser.add_argument("--cache", type=Path, default=Path("ml/cache"))
    parser.add_argument("--out", type=Path, default=Path("firmware/models/smartbin-pruned.npz"))
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--version", default="smartbin-pruned-1")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.92)
    base_ops, network = load_ops(args.model)
    pruned_ops = prune_blocks(base_ops, args.ratio)
    full, small = count_macs(base_ops), count_macs(pruned_ops)
    print(f"macs {full / 1e6:.0f}M -> {small / 1e6:.0f}M ({small / full:.2f}x)", flush=True)

    teacher = GraphModule(base_ops).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = GraphModule(pruned_ops).to(DEVICE)

    load = lambda name: torch.from_numpy(np.load(args.cache / name, mmap_mode="r")[:])
    xtr, ytr = load("train_x.npy"), torch.from_numpy(np.load(args.cache / "train_y.npy"))
    if args.limit:
        xtr, ytr = xtr[: args.limit], ytr[: args.limit]
    xtr = xtr.pin_memory()
    ytr = ytr.to(DEVICE)
    splits = {
        name: (load(f"{name}_x.npy"), torch.from_numpy(np.load(args.cache / f"{name}_y.npy")))
        for name in ("val", "test_id", "ext_test")
    }

    print(f"pruned before fine-tune: val {evaluate(student, *splits['val']):.4f}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = len(xtr) // args.batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1
    )
    best, best_state = -1.0, None
    start = time.time()
    for epoch in range(args.epochs):
        student.train()
        order = torch.randperm(len(xtr))
        running = 0.0
        for step in range(steps):
            idx = order[step * args.batch : (step + 1) * args.batch]
            xb = augment(xtr[idx].to(DEVICE, non_blocking=True))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                target = F.softmax(teacher(xb).float() / args.temperature, dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = student(xb).float()
            distill = F.kl_div(
                F.log_softmax(logits / args.temperature, dim=1), target, reduction="batchmean"
            ) * args.temperature**2
            hard = F.cross_entropy(logits, ytr[idx.to(DEVICE)], label_smoothing=0.05)
            loss = args.alpha * distill + (1 - args.alpha) * hard
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item()
            if step % 40 == 0:
                print(f"  step {step}/{steps} {time.time() - start:.0f}s", flush=True)
        val = evaluate(student, *splits["val"])
        torch.cuda.empty_cache()
        if val > best:
            best = val
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        print(f"epoch {epoch + 1}/{args.epochs} loss {running / steps:.4f} val {val:.4f} ({time.time() - start:.0f}s)", flush=True)

    student.load_state_dict(best_state)
    for name, (x, y) in splits.items():
        print(f"FINAL {name}: {evaluate(student, x, y):.4f}", flush=True)

    meta = {"classes": list(network.classes), "version": args.version}
    arrays = pack(meta, student.to_ops(pruned_ops), network.input_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
