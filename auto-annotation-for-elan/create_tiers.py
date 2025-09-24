import argparse
from os import path

import yaml

from elan.elan_helpers import (
    create_elan, write_tiers_to_eaf,
    write_eaf_file, write_psfx_file, write_tsconf_file
)
from elan.extractors import extract_timeseries
from elan.time_helpers import filename2frames
from  io_helpers import (
    create_working_folders,
    load_scenes, load_annotations, load_data,
    save_timeseries_to_csv, handle_eaf_fn
    )

def main(args):
    with open("configs/elan_config.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    tiers = config_data["tiers"]
    timeseries_names = config_data["timeseries"]
    annotations = load_annotations(args.timeseries_home, args.video_id, args.annotation_fn)
    data = load_data(args.timeseries_home, args.video_id)
    all_scenes = load_scenes(
        args.path_to_scenes, 
        from_boundaries=True,
        boundaries=data["boundaries"],
        fps=args.video_fps)
    start_frame, end_frame = filename2frames(args.eaf_file, args.video_fps)
    if end_frame is not None:
        end_frame = min(end_frame, all_scenes[-1][1])
    else:
        end_frame = all_scenes[-1][1]
    _, elans_folder, csvs_folder = create_working_folders(args)
    eaf_fn = handle_eaf_fn(args.eaf_file, args.video_id, start_frame, end_frame, args.video_fps)
    eaf_fn = path.basename(eaf_fn)
    if timeseries_names is not None:
        timeseries = extract_timeseries(
            data, 
            timeseries_names,
            start_frame, 
            end_frame,
            args.video_fps,
            annotations=annotations,
            all_scenes=all_scenes
            )
    else:
        timeseries = dict()
    save_timeseries_to_csv(timeseries, csvs_folder, eaf_fn)
    write_tsconf_file(elans_folder, eaf_fn, timeseries)

    eaf = create_elan(args.eaf_file)
    eaf = write_tiers_to_eaf(eaf, tiers, annotations, start_frame, end_frame, args.video_fps)
    write_eaf_file(elans_folder, eaf, args.video_id, eaf_fn)
    write_psfx_file(elans_folder, eaf_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries-home", default="")
    parser.add_argument("--eaf-file", default="")
    parser.add_argument("--path-to-scenes", default="")
    parser.add_argument("--save-to", default="")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--annotation-fn", default="")
    args = parser.parse_args()
    main(args)
