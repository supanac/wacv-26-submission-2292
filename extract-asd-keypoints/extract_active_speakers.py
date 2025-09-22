import argparse
from os import path

import numpy as np

from extract_asd_keypoints import load_scores_tracks
from extract_asd_keypoints import SpeakerKeypoints


def main(args):
    path_to_asd_output = args.path_to_asd_output
    path_to_jsons = args.path_to_jsons
    video_id = args.video_id
    video_fps = args.video_fps
    video_width = args.video_width
    video_height = args.video_height
    save_to = args.save_to
    save_to_npz = path.join(save_to, video_id)
    scores, tracks = load_scores_tracks(path_to_asd_output)
    speaker_kpts = SpeakerKeypoints(
        scores, 
        tracks, 
        path_to_jsons, 
        video_fps, 
        video_width, 
        video_height
        )
    data_dict = speaker_kpts.get_data()
    np.savez_compressed(save_to_npz, **data_dict)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-asd-output", type=str)
    parser.add_argument("--path-to-jsons", type=str)
    parser.add_argument("--video-id", type=str)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--video-width", type=int)
    parser.add_argument("--video-height", type=int)
    parser.add_argument("--save-to", type=str)
    args = parser.parse_args()
    main(args)
