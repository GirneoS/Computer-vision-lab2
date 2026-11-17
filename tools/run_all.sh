#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory; test videos stay outside version control.
lab2_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lab2_video_dir="${1:-$lab2_root/test-videos}"
cmake -S "$lab2_root" -B "$lab2_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$lab2_root/build" --parallel
ctest --test-dir "$lab2_root/build" --output-on-failure
for lab2_video_name in mona-lisa mona-lisa-blur mona-lisa-blur-extra-credit; do
    "$lab2_root/build/lab2_tracker" \
        --input "$lab2_video_dir/$lab2_video_name.avi" \
        --output-dir "$lab2_root/results/$lab2_video_name" --label "Mona Lisa"
done
