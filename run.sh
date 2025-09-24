#!/bin/bash

VIDEO_ID="$1"
VIDEO_FPS="$2"

TIMESERIES_HOME=/path/to/keypoints.npz
PATH_TO_SCENES=/to/path/scenes.csv
SAVE_TO=/where/to/save/results/

python pseudolabels.py \
    --timeseries-home "$TIMESERIES_HOME" \
    --path-to-scenes "$PATH_TO_SCENES" \
    --video-id "$VIDEO_ID" \
    --video-fps "$VIDEO_FPS"

python create_tiers.py \
    --timeseries-home "$TIMESERIES_HOME" \
    --path-to-scenes "$PATH_TO_SCENES" \
    --video-id "$VIDEO_ID" \
    --video-fps "$VIDEO_FPS" \
    --save-to "$SAVE_TO"
