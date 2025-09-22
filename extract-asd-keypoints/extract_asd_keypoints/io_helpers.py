from os import path, listdir as ls
import json
from typing import Tuple, Optional

import numpy as np
from natsort import natsorted
from scenedetect.frame_timecode import FrameTimecode

def load_scores_tracks(case_home: str) -> Tuple[list, list]:
    """Load scores and tracks from the storage.
    
    Args:
        case_home: Path to the folder with pickle files from the ASD pipeline.

    Returns:
        Tuple of lists of loaded values.
    """
    tracks_path = path.join(case_home, "tracks.pckl")
    scores_path = path.join(case_home, "scores.pckl")
    tracks = np.load(tracks_path, allow_pickle=True)
    scores = np.load(scores_path, allow_pickle=True)
    return scores, tracks

def load_all_jsons(jsons_home: str) -> list:
    """Load all the json files from the storage.

    Args:
        jsons_home: Path to folder where jsons live.

    Returns:
        all_data_from_jsons: list of metadata from json files.
    """
    all_data_from_jsons = list()
    all_jsons = natsorted(ls(jsons_home))
    all_jsons = [path.join(jsons_home, fn) for fn in all_jsons]
    for fp in all_jsons:
        json_data = json.load(open(fp))
        all_data_from_jsons.append(json_data)
    return all_data_from_jsons

def correct_timecode(timecode: str) -> str:
    H, M, S_MS = timecode.split(":")
    S, MS = S_MS.split(".")
    if S == "60":
        S = "00"
        M = str(int(M) + 1).zfill(2)
        timecode = H + ":" + M + ":" + S + "." + MS
    return timecode

def load_scenes(path_to_scenes: Optional[str], fps: float) -> Optional[list]:
    if path_to_scenes is None:
        return
    with open(path_to_scenes, "r") as file:
        all_lines = file.readlines()
    all_scenes = list()
    for line in all_lines[2:]:
        splitted_line = line.split(',')
        _, _, start_timecode, _, _, end_timecode, *_ = splitted_line
        start_timecode = correct_timecode(start_timecode)
        end_timecode = correct_timecode(end_timecode)
        start_frame = FrameTimecode(start_timecode, fps)
        end_frame = FrameTimecode(end_timecode, fps)
        all_scenes.append((start_frame, end_frame))
    return all_scenes
    