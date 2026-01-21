#!/bin/bash
MOT17_BASE="/path/to/MOT17/train"
for seq in $MOT17_BASE/*; do
    if [ -d "$seq" ]; then
        echo "Running sequence: $seq"
        mkdir -p "$seq/byteyolo_det"
        ./byteyolo_nhwc --seq "$seq"
    fi
done
