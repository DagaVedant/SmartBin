import argparse
import json
from pathlib import Path

from dataset import load_images, load_manifest, split_by_session
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--images", default="../datasets/raw")
    parser.add_argument("--out", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    train_rows, val_rows, holdout = split_by_session(rows, seed=args.seed)

    if not val_rows:
        raise SystemExit("holdout is empty - you need at least 2 sessions")

    print(len(train_rows), "train frames,", len(val_rows), "holdout frames")
    print("holdout sessions:", holdout)

    x_train, y_train = load_images(train_rows, args.images)
    x_val, y_val = load_images(val_rows, args.images)

    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "smartbin.keras")
    (out / "holdout.json").write_text(json.dumps(holdout, indent=2))

    print("saved", out / "smartbin.keras")


if __name__ == "__main__":
    main()
