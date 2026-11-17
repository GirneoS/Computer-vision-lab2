#!/usr/bin/env bash
set -euo pipefail
lab2_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for lab2_name in mona-lisa mona-lisa-blur mona-lisa-blur-extra-credit synthetic; do
    ffmpeg -hide_banner -loglevel error -y \
        -i "$lab2_root/results/$lab2_name/annotated.avi" \
        -c:v libx264 -crf 22 -preset fast -pix_fmt yuv420p -movflags +faststart \
        "$lab2_root/results/$lab2_name/preview.mp4"
done
