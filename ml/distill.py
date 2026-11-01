"""Distill the DumpGuard MobileNetV3Large teacher into the small NumPy-runnable student.

    python ml/distill.py --teacher <DumpGuard>/.../out/run3/best.weights.h5 \
        --cache ml/cache --out firmware/models/smartbin.npz

Augmentation runs on the GPU. Teacher and student see the same augmented view:
the teacher at 224 px, the student an antialiased 128 px downscale of it.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
import torch.nn.functional as F

from student import CLASSES, INPUT_SIZE, MEAN, STD, Student, export_npz

DEVICE = "cuda"


def load_teacher(weights: Path):
    import keras

    base = keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3), include_top=False, weights=None,
        include_preprocessing=True, pooling="avg", minimalistic=False,
    )
    x = keras.layers.Dropout(0.3)(base.output)
    out = keras.layers.Dense(len(CLASSES), activation="softmax", dtype="float32", name="probs")(x)
    model = keras.Model(base.input, out)
    model.load_weights(str(weights))
    return model


@torch.no_grad()
def teacher_probs(teacher, nchw: torch.Tensor) -> torch.Tensor:
    """nchw: float 0..255. The keras model wants NHWC."""
    # keras holds every intermediate until the forward finishes (~12 GB at batch 256), so chunk it
    outs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk in nchw.split(64):
            outs.append(torch.as_tensor(teacher(chunk.permute(0, 2, 3, 1).contiguous(), training=False)).float())
    return torch.cat(outs)


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: uint8 NHWC on GPU -> float NCHW 0..255 at the same size."""
    b = x.shape[0]
    x = x.permute(0, 3, 1, 2).float()

    zoom = torch.empty(b, device=x.device).uniform_(0.67, 1.0)
    aspect = torch.empty(b, device=x.device).uniform_(0.87, 1.15)
    sx, sy = zoom * aspect.sqrt(), zoom / aspect.sqrt()
    angle = torch.empty(b, device=x.device).uniform_(-25, 25) * math.pi / 180
    angle = angle * (torch.rand(b, device=x.device) < 0.5)
    flip = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
    tx = (torch.rand(b, device=x.device) * 2 - 1) * (1 - sx).clamp(min=0)
    ty = (torch.rand(b, device=x.device) * 2 - 1) * (1 - sy).clamp(min=0)

    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack(
        [
            torch.stack([sx * cos * flip, -sy * sin, tx], dim=1),
            torch.stack([sx * sin * flip, sy * cos, ty], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    brightness = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.5, 1.3)
    contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)
    saturation = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.5, 1.4)
    x = x * brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * saturation + gray
    to_gray = (torch.rand(b, 1, 1, 1, device=x.device) < 0.05)
    x = torch.where(to_gray, gray.expand_as(x), x)
    noisy = (torch.rand(b, 1, 1, 1, device=x.device) < 0.2)
    x = x + torch.randn_like(x) * 8 * noisy
    return x.clamp(0, 255)


def to_student_input(nchw255: torch.Tensor) -> torch.Tensor:
    small = F.interpolate(
        nchw255, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", antialias=True, align_corners=False
    )
    mean = torch.tensor(MEAN, device=small.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=small.device).view(1, 3, 1, 1)
    return ((small / 255.0 - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(student, teacher, x, y, batch=256):
    student.eval()
    hits_s = hits_t = 0
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(DEVICE).permute(0, 3, 1, 2).float()
        yb = y[i : i + batch].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = student(to_student_input(xb))
        hits_s += (logits.argmax(1) == yb).sum().item()
        if teacher is not None:
            hits_t += (teacher_probs(teacher, xb).argmax(1) == yb).sum().item()
    return hits_s / len(x), (hits_t / len(x) if teacher is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("ml/cache"))
    parser.add_argument("--out", type=Path, default=Path("firmware/models/smartbin.npz"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7, help="weight of the distillation loss")
    parser.add_argument("--limit", type=int, default=0, help="train on a subset (smoke test)")
    parser.add_argument("--version", default="smartbin-student-1")
    args = parser.parse_args()

    torch.manual_seed(0)
    # stay under the card's dedicated memory: past it Windows spills into system RAM and steps get 3x slower
    torch.cuda.set_per_process_memory_fraction(0.92)
    cache = args.cache
    load = lambda name: torch.from_numpy(np.load(cache / name, mmap_mode="r")[:])
    xtr, ytr = load("train_x.npy"), torch.from_numpy(np.load(cache / "train_y.npy"))
    if args.limit:
        xtr, ytr = xtr[: args.limit], ytr[: args.limit]
    # the train set (5 GB) stays in system RAM: on a 16 GB card it spills into shared memory and crawls
    xtr = xtr.pin_memory()
    ytr = ytr.to(DEVICE)
    splits = {
        name: (load(f"{name}_x.npy"), torch.from_numpy(np.load(cache / f"{name}_y.npy")))
        for name in ("val", "test_id", "ext_test")
    }

    teacher = load_teacher(args.teacher)
    for name, (x, y) in splits.items():
        _, acc = evaluate(Student(args.width).to(DEVICE), teacher, x, y)
        print(f"teacher {name}: {acc:.4f}")
        torch.cuda.empty_cache()

    student = Student(args.width).to(DEVICE).to(memory_format=torch.channels_last)
    print("student params:", sum(p.numel() for p in student.parameters()))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=2e-2)
    steps_per_epoch = len(xtr) // args.batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )

    best, best_state = -1.0, None
    start = time.time()
    for epoch in range(args.epochs):
        student.train()
        order = torch.randperm(len(xtr))
        running = 0.0
        for step in range(steps_per_epoch):
            idx = order[step * args.batch : (step + 1) * args.batch]
            xb = xtr[idx].to(DEVICE, non_blocking=True)
            view = augment(xb)
            target = teacher_probs(teacher, view).clamp_min(1e-8)
            soft = F.softmax(target.log() / args.temperature, dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = student(to_student_input(view)).float()
            distill = F.kl_div(
                F.log_softmax(logits / args.temperature, dim=1), soft, reduction="batchmean"
            ) * args.temperature**2
            hard = F.cross_entropy(logits, ytr[idx.to(DEVICE)], label_smoothing=0.05)
            loss = args.alpha * distill + (1 - args.alpha) * hard
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item()
            if step % 20 == 0:
                print(f"  step {step}/{steps_per_epoch} {time.time() - start:.0f}s alloc {torch.cuda.memory_allocated() / 2**30:.1f}G reserved {torch.cuda.memory_reserved() / 2**30:.1f}G", flush=True)

        val_acc, _ = evaluate(student, None, *splits["val"])
        torch.cuda.empty_cache()
        if val_acc > best:
            best = val_acc
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {running / steps_per_epoch:.4f} "
            f"val {val_acc:.4f} ({time.time() - start:.0f}s)",
            flush=True,
        )

    student.load_state_dict(best_state)
    for name, (x, y) in splits.items():
        acc, teacher_acc = evaluate(student, teacher, x, y)
        print(f"FINAL {name}: student {acc:.4f}  teacher {teacher_acc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_npz(student, args.out, args.version)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
