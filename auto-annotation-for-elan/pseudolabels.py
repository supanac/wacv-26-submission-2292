import argparse
from os import path

import numpy as np

from constants import FPS
import annotation
from io_helpers import load_scenes, parse_pseudo_labels_config
from annotation.create_annotation import create_annotation


def main(args):
    timeseries_home = args.timeseries_home
    video_id = args.video_id
    video_fps = args.video_fps
    path_to_scenes = args.path_to_scenes
    if args.annotation_fn == "":
        saveto = path.join(timeseries_home, video_id + "_annotation.npz")
    else:
        saveto = path.join(timeseries_home, args.annotation_fn + ".npz")
    npz_path = path.join(timeseries_home, video_id + ".npz")
    data = np.load(npz_path)
    tiers = parse_pseudo_labels_config("configs/pseudolabels_config.yaml")
    all_scenes = load_scenes(
        path_to_scenes, 
        from_boundaries=True,
        boundaries=data["boundaries"],
        fps=video_fps)
    save_dict = {
        tier: create_annotation(tier, data, all_scenes)
        for tier in tiers
    }
    np.savez(saveto, **save_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries-home", default="", type=str)
    parser.add_argument("--path-to-scenes", default="", type=str)
    parser.add_argument("--video-id", default="", type=str)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--annotation-fn", default="")
    args = parser.parse_args()
    main(args)
