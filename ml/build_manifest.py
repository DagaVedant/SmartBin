import argparse
import csv
from pathlib import Path

from dataset import IMAGE_SUFFIXES, LABELS


def scan(root):
    root = Path(root)
    rows = []

    for session in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted(session.rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            label = image.parent.name
            if label not in LABELS:
                label = "unknown"

            rows.append({
                "session_id": session.name,
                "filename": str(image.relative_to(session)).replace("\\", "/"),
                "label": label,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../datasets/raw")
    parser.add_argument("--out", default="manifest.csv")
    args = parser.parse_args()

    rows = scan(args.images)
    if not rows:
        raise SystemExit("no images found under " + args.images)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "filename", "label"])
        writer.writeheader()
        writer.writerows(rows)

    unlabelled = sum(1 for row in rows if row["label"] == "unknown")
    print(len(rows), "frames from", len({r["session_id"] for r in rows}), "sessions ->", args.out)
    if unlabelled:
        print(unlabelled, "frames landed in unknown - label them by folder name")


if __name__ == "__main__":
    main()
