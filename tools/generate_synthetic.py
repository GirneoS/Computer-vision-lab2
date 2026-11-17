#!/usr/bin/env python3
"""Generate a deterministic geometric test; this is not a self-recorded real video.

Python/OpenCV is used only to prepare known ground truth, independently of the
C++ tracker. Default output: results/synthetic/input.avi and ground_truth.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def create_reference(width=640, height=360, seed=17):
    """A planar target with reproducible texture at several scales."""
    rng = np.random.default_rng(seed)
    image = np.clip(225 + rng.normal(0, 6, (height, width, 3)), 0, 255).astype(np.uint8)
    for _ in range(180):
        center = tuple(int(v) for v in rng.integers([12, 12], [width - 12, height - 12]))
        color = tuple(int(v) for v in rng.integers(10, 230, size=3))
        cv2.circle(image, center, int(rng.integers(3, 14)), color,
                   int(rng.choice([-1, 2])), cv2.LINE_AA)
    for _ in range(35):
        points = rng.integers([10, 10], [width - 10, height - 10], size=(2, 2))
        color = tuple(int(v) for v in rng.integers(5, 150, size=3))
        cv2.line(image, tuple(points[0]), tuple(points[1]), color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (3, 3), (width - 4, height - 4), (20, 20, 20), 4)
    for index, text in enumerate(["PLANAR TARGET", "VISION LAB 2", "SIFT  17  42"]):
        cv2.putText(image, text, (25, 55 + index * 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (15, 15, 15), 2, cv2.LINE_AA)
    return image


def corners_of(image):
    height, width = image.shape[:2]
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                     [0, height - 1]], dtype=np.float32)


def render_scene(reference, corners, occlude=False, blur_sigma=0):
    width, height = 640, 360
    yy, xx = np.indices((height, width))
    gray = (35 + 12 * xx / width + 10 * yy / height).astype(np.uint8)
    frame = cv2.merge([gray, gray, gray])
    if corners is not None:
        matrix = cv2.getPerspectiveTransform(corners_of(reference), np.asarray(corners, np.float32))
        warped = cv2.warpPerspective(reference, matrix, (width, height))
        mask = cv2.warpPerspective(np.full(reference.shape[:2], 255, np.uint8),
                                   matrix, (width, height), flags=cv2.INTER_NEAREST)
        frame[mask > 0] = warped[mask > 0]
        if occlude:
            center = np.mean(corners, axis=0).astype(int)
            cv2.rectangle(frame, tuple(center - [45, 35]), tuple(center + [45, 35]),
                          (95, 70, 30), -1)
    return cv2.GaussianBlur(frame, (0, 0), blur_sigma) if blur_sigma > 0 else frame


def generate(output_dir, frame_count=96):
    cv2.setNumThreads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = create_reference()
    fps = 20.0
    writer = cv2.VideoWriter(str(output_dir / "input.avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Cannot create synthetic input.avi (MJPG).")
    rows, previews = [], []
    preview_indices = {0, 15, 28, 40, 45, 50, 66, 75, frame_count - 1}
    try:
        for index in range(frame_count):
            visible = not 40 <= index < 50
            occluded, blurred = 64 <= index < 72, 24 <= index < 32
            phase = 2 * math.pi * index / frame_count
            if index == 0:
                corners = corners_of(reference)
            else:
                scale = 0.50 + 0.11 * math.sin(phase)
                angle = 0.42 * math.sin(phase * 1.2)
                rectangle = np.array([[-320, -180], [319, -180], [319, 179], [-320, 179]], np.float32)
                rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]], np.float32)
                corners = rectangle @ rotation.T * scale
                corners += np.array([320 + 85 * math.sin(phase),
                                     180 + 30 * math.cos(phase)], np.float32)
                corners[1] += [12 * math.sin(phase * 2), 5]
                corners[2] += [-7, 9 * math.cos(phase)]
            frame = render_scene(reference, corners if visible else None, occluded,
                                 1.3 if blurred else 0)
            row = {"frame": index, "time_s": index / fps, "visible": int(visible),
                   "occluded": int(occluded), "blurred": int(blurred)}
            for corner in range(4):
                row[f"x{corner}"] = float(corners[corner, 0]) if visible else None
                row[f"y{corner}"] = float(corners[corner, 1]) if visible else None
            rows.append(row)
            writer.write(frame)
            if index in preview_indices:
                annotated = frame.copy()
                if visible:
                    cv2.polylines(annotated, [np.rint(corners).astype(np.int32)], True,
                                  (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"frame {index}; visible={int(visible)}; GT=cyan", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                previews.append(cv2.resize(annotated, (320, 180)))
    finally:
        writer.release()
    with (output_dir / "ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    metadata = {
        "description": "Deterministic synthetic planar target; not a self-recorded real video.",
        "seed": 17, "frames": frame_count, "fps": fps, "size": [640, 360],
        "visible_frames": frame_count - 10,
        "absent_frames_inclusive": [40, 49], "first_reappearance_frame": 50,
        "blurred_frames_inclusive": [24, 31], "occluded_frames_inclusive": [64, 71],
        "corner_order": "top-left, top-right, bottom-right, bottom-left in reference coordinates",
        "coordinates": "pixel coordinates; first frame reference spans (0,0) through (639,359)",
        "ground_truth": "ground_truth.csv; invisible frame corners are empty",
        "codec": "MJPG (lossy); geometric ground truth precedes video encoding",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    while len(previews) < 9:
        previews.append(np.zeros((180, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(previews[index:index + 3]) for index in range(0, 9, 3)])
    if not cv2.imwrite(str(output_dir / "ground_truth_preview.jpg"), sheet):
        raise RuntimeError("Cannot write preview image.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    if args.frames < 80:
        parser.error("--frames must be at least 80 to include every scenario")
    generate(args.output_dir, args.frames)


if __name__ == "__main__":
    main()
