#!/usr/bin/env python3
"""Compare tracker corners with synthetic ground truth; no accuracy claims for real videos."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


COORDINATES = [name for index in range(4) for name in (f"x{index}", f"y{index}")]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        missing = [key for key in ["frame", *COORDINATES] if key not in row]
        if missing:
            raise ValueError(f"Missing CSV columns in {path}: {missing}")
        index = int(row["frame"])
        if index in indexed:
            raise ValueError(f"Duplicate frame {index} in {path}")
        indexed[index] = row
    if not indexed:
        raise ValueError(f"No frame rows in {path}")
    return indexed


def corners(row):
    values = [row.get(name, "") for name in COORDINATES]
    if any(value in ("", None) for value in values):
        return None
    points = np.array([float(value) for value in values], np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        return None
    return points


def polygon_iou(first, second):
    if not cv2.isContourConvex(first) or not cv2.isContourConvex(second):
        return 0.0
    area1, area2 = abs(cv2.contourArea(first)), abs(cv2.contourArea(second))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area1 + area2 - intersection
    return float(np.clip(intersection / union, 0.0, 1.0)) if union > 0 else 0.0


def evaluate(ground_truth_path, tracking_path, output_path):
    truth, tracking = read_rows(ground_truth_path), read_rows(tracking_path)
    if set(truth) != set(tracking):
        raise ValueError("Tracker must contain exactly one row for every ground-truth frame")
    rows = []
    for index in sorted(truth):
        visible = bool(int(truth[index]["visible"]))
        target, estimate = corners(truth[index]), corners(tracking[index])
        if visible and target is None:
            raise ValueError(f"Missing ground-truth corners for visible frame {index}")
        detected = estimate is not None
        error = float(np.linalg.norm(estimate - target, axis=1).mean()) if visible and detected else None
        iou = polygon_iou(target, estimate) if visible and detected else 0.0
        rows.append({"frame": index, "visible": int(visible), "detected": int(detected),
                     "mean_corner_error_px": error, "iou": iou if visible else None})
    visible_rows = [row for row in rows if row["visible"]]
    detected_visible = [row for row in visible_rows if row["detected"]]
    absent_rows = [row for row in rows if not row["visible"]]
    reappearance_indices = [index for index in range(1, len(rows))
                            if rows[index]["visible"] and not rows[index - 1]["visible"]]
    summary = {
        "scope": "Synthetic benchmark only; this is not a self-recorded real video.",
        "frames": len(rows), "visible_frames": len(visible_rows),
        "detected_visible_frames": len(detected_visible),
        "recall_visible": len(detected_visible) / len(visible_rows) if visible_rows else None,
        "absent_frames": len(absent_rows),
        "false_positives_absent": sum(row["detected"] for row in absent_rows),
        "mean_corner_error_px_detected": float(np.mean([row["mean_corner_error_px"] for row in detected_visible])) if detected_visible else None,
        "max_mean_corner_error_px_detected": max((row["mean_corner_error_px"] for row in detected_visible), default=None),
        "mean_iou_visible_including_misses": float(np.mean([row["iou"] for row in visible_rows])) if visible_rows else None,
        "reappearance_frames": [rows[index]["frame"] for index in reappearance_indices],
        "detected_on_first_reappearance_frames": [bool(rows[index]["detected"]) for index in reappearance_indices],
        "ground_truth_path": str(ground_truth_path), "tracking_path": str(tracking_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_path.with_name("errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=Path("results/synthetic/ground_truth.csv"))
    parser.add_argument("--tracking", type=Path, default=Path("results/synthetic/tracking.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/synthetic/evaluation.json"))
    args = parser.parse_args()
    try:
        result = evaluate(args.ground_truth, args.tracking, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
