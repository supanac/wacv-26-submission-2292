from os import path, makedirs
from pathlib import Path
from itertools import product
from typing import Optional

from omegaconf import OmegaConf
import numpy as np
import pandas as pd


def correct_timecode(timecode: str) -> str:
    H, M, S_MS = timecode.split(":")
    S, MS = S_MS.split(".")
    if S == "60":
        S = "00"
        M = str(int(M) + 1).zfill(2)
        timecode = H + ":" + M + ":" + S + "." + MS
    return timecode

def load_scenes_from_npz(path_to_scenes):
    return np.load(path_to_scenes, allow_pickle=True)["all_scenes"]

def load_scenes_from_csv(path_to_scenes: str, fps: float):
    with open(path_to_scenes, "r") as file:
        all_lines = file.readlines()
    all_scenes = list()
    for line in all_lines[2:]:
        splitted_line = line.split(',')
        _, start_frame, _, _, end_frame, _, *_ = splitted_line
        start_frame = int(start_frame) - 1
        end_frame = int(end_frame) - 1
        all_scenes.append((start_frame, end_frame))
    return all_scenes

def load_scenes(
        path_to_scenes: str,
        from_boundaries: bool = False,
        boundaries: Optional[list] = None,
        fps: float=25.
        ) -> Optional[list]:
    if path_to_scenes is None:
        return None
    if from_boundaries:
        if boundaries is not None:
            if fps != 25:
                return [
                    [round(elem[0] / 25 * fps), round(elem[1] / 25 * fps)]
                    for elem in boundaries
                    ]
            else:
                return boundaries
        else:
            raise ValueError("Boundaries cannot be None!")
    try:
        return load_scenes_from_npz(path_to_scenes)
    except:
        return load_scenes_from_csv(path_to_scenes, fps)

def read_pseudo_labels_config(path_to_config):
    return OmegaConf.load("pseudolabels_config.yaml")

def parse_pseudo_labels_config(path_to_config):
    conf = read_pseudo_labels_config(path_to_config)
    all_fingers = ["thumb", "index", "middle", "ring", "little"]
    all_hands = ["right", "left"]
    all_wrist_tiers = list()
    all_finger_tiers = list()
    all_eyelid_tiers = list()
    for hand in all_hands:
        wrist_annotation_list = eval("conf.wrists." + hand)
        wrist_annotation_list = [
            hand + "_wrist_" + elem for elem in wrist_annotation_list
            ]
        all_wrist_tiers.extend(wrist_annotation_list)
    for hand, finger in product(all_hands, all_fingers):
        finger_annotation_list = eval("conf.fingers." + hand + "." + finger)
        finger_annotation_list = [
            hand + "_" + finger + "_finger_" + elem for elem in finger_annotation_list
            ]
        all_finger_tiers.extend(finger_annotation_list)
    for eye in all_hands:
        eyelid_annotation_list = conf['eyelid'][eye]
        eyelid_annotation_list = [
            eye +"_eyelid_"+ elem for elem in eyelid_annotation_list
        ]
        all_eyelid_tiers.extend(eyelid_annotation_list)
    all_wrist_tiers = sorted(all_wrist_tiers)
    all_finger_tiers = sorted(all_finger_tiers)
    all_eyelid_tiers = sorted(all_eyelid_tiers)
    return all_wrist_tiers + all_finger_tiers + all_eyelid_tiers

def load_annotations(timeseries_home, video_id, annotation_fn=""):
    if annotation_fn == "":
        annotation_fn =  video_id + "_annotation"
    annotations_fp = path.join(
        timeseries_home, annotation_fn + ".npz"
        )
    annotations = np.load(annotations_fp, allow_pickle=True)
    return annotations

def load_data(timeseries_home, video_id):
    data_npz_path = path.join(timeseries_home, video_id + ".npz")
    data = np.load(data_npz_path)
    return data

def save_timeseries_to_csv(timeseries, csvs_folder, eaf_fn):
    df = pd.DataFrame.from_dict(timeseries)
    csv_fn = ".".join(eaf_fn.split(".")[:-1])
    csv_fp = path.join(csvs_folder, csv_fn + "_timeseries.csv")
    df.to_csv(csv_fp, index=False, na_rep="N/A")

def handle_eaf_fn(eaf_fn, video_id, start_frame, end_frame, video_fps):
    if eaf_fn == "":
        start_sec = round(start_frame / video_fps, 1)
        end_sec = round(end_frame / video_fps, 1)
        eaf_fn = video_id + "-" + str(start_sec) + "-" + str(end_sec) + ".eaf"
    return eaf_fn

def create_working_folders(args):
    video_id = args.video_id
    save_to_fd = args.save_to
    save_to_fd = path.join(save_to_fd, video_id)
    elans_folder = path.join(save_to_fd, "auto_annotation")
    csvs_folder = path.join(elans_folder, "csvs")
    if not path.exists(save_to_fd):
        print("Creating a folder: ", save_to_fd)
        Path(save_to_fd).mkdir(parents=True, exist_ok=True)
    if not path.exists(elans_folder):
        print("Creating a folder: ", elans_folder)
        Path(elans_folder).mkdir(parents=True, exist_ok=True)
    if not path.exists(csvs_folder):
        print("Creating a folder: ", csvs_folder)
        Path(csvs_folder).mkdir(parents=True, exist_ok=True)
    return save_to_fd, elans_folder, csvs_folder