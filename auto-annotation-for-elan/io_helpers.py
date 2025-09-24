from os import path
from argparse import Namespace
from pathlib import Path
from itertools import product
from typing import Optional, List

from omegaconf import OmegaConf
import numpy as np
import pandas as pd

from datatypes import (
    FilePath, FolderPath, NumpyArray, Timeseries, SceneList, Number, Annotations
)


def correct_timecode(timecode: str) -> str:
    """
    Correct timecode format by handling seconds overflow (60 seconds -> next minute).
    
    Args:
        timecode (str): Timecode in format "HH:MM:SS.MS"
    
    Returns:
        str: Corrected timecode with proper minute increment if seconds was 60.
    """
    H, M, S_MS = timecode.split(":")
    S, MS = S_MS.split(".")
    if S == "60":
        S = "00"
        M = str(int(M) + 1).zfill(2)
        timecode = H + ":" + M + ":" + S + "." + MS
    return timecode

def load_scenes_from_npz(path_to_scenes: FilePath) -> NumpyArray:
    """
    Load scene data from NPZ file.
    
    Args:
        path_to_scenes (FilePath): Path to NPZ file containing scene data.
    
    Returns:
        NumpyArray: Array of scenes from the "all_scenes" key.
    """
    return np.load(path_to_scenes, allow_pickle=True)["all_scenes"]

def load_scenes_from_csv(path_to_scenes: str) -> List:
    """
    Load scene data from CSV file, extracting start and end frames.
    
    Args:
        path_to_scenes (str): Path to CSV file with scene boundaries.
    
    Returns:
        List: List of tuples containing (start_frame, end_frame) pairs.
    """
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

def load_scenes(path_to_scenes: str, from_boundaries: bool = False, boundaries: Optional[List] = None, fps: Number=25.) -> Optional[SceneList]:
    """
    Load scene data with preprocessing, trying NPZ first then CSV format.
    
    Args:
        path_to_scenes (str): Path to scene file (NPZ or CSV).
        from_boundaries (bool): If True, use provided boundaries instead of file.
        boundaries (Optional[List]): Scene boundaries to use when from_boundaries=True.
        fps (Number): Frame rate for FPS conversion. Defaults to 25.
    
    Returns:
        Optional[SceneList]: Scene data or None if path_to_scenes is None.
    
    Raises:
        ValueError: If from_boundaries=True but boundaries is None.
    """
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
        return load_scenes_from_csv(path_to_scenes)

def read_pseudo_labels_config(path_to_config: Optional[FilePath]=None):
    """
    Load pseudo labels configuration from YAML file.
    
    Args:
        path_to_config (Optional[FilePath]): Path to config file. 
            Defaults to "pseudolabels_config.yaml" if None.
    
    Returns:
        OmegaConf object with loaded configuration.
    """
    if path_to_config is not None:
        return OmegaConf.load(path_to_config)
    return OmegaConf.load("pseudolabels_config.yaml")

def parse_pseudo_labels_config(path_to_config: FilePath) -> List:
    """
    Parse pseudo labels config and generate tier names for wrists, fingers, and eyelids.
    
    Args:
        path_to_config (FilePath): Path to pseudo labels configuration file.
    
    Returns:
        List: Combined sorted list of all wrist, finger, and eyelid tier names.
    """
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

def load_annotations(timeseries_home: FolderPath, video_id: str, annotation_fn: str) -> Annotations:
    """
    Load annotation data from NPZ file.
    
    Args:
        timeseries_home (FolderPath): Directory containing annotation files.
        video_id (str): Video identifier used as fallback filename.
        annotation_fn (str): Annotation filename (without extension). 
            Uses video_id + "_annotation" if empty.
    
    Returns:
        Annotations: Loaded annotation data from NPZ file.
    """
    if annotation_fn == "":
        annotation_fn =  video_id + "_annotation"
    annotations_fp = path.join(
        timeseries_home, annotation_fn + ".npz"
        )
    annotations = np.load(annotations_fp, allow_pickle=True)
    return annotations

def load_data(timeseries_home: FolderPath, video_id: str) -> NumpyArray:
    """
    Load timeseries data from NPZ file.
    
    Args:
        timeseries_home (FolderPath): Directory containing data files.
        video_id (str): Video identifier used as filename.
    
    Returns:
        NumpyArray: Loaded data from NPZ file.
    """
    data_npz_path = path.join(timeseries_home, video_id + ".npz")
    data = np.load(data_npz_path)
    return data

def save_timeseries_to_csv(timeseries: Timeseries, csvs_folder: FolderPath, eaf_fn: FilePath) -> None:
    """
    Save timeseries data to CSV file.
    
    Args:
        timeseries (Timeseries): Timeseries data to save.
        csvs_folder (FolderPath): Directory to save CSV file.
        eaf_fn (FilePath): EAF filename used to generate CSV filename.
    
    Returns:
        None
    """
    df = pd.DataFrame.from_dict(timeseries)
    csv_fn = ".".join(eaf_fn.split(".")[:-1])
    csv_fp = path.join(csvs_folder, csv_fn + "_timeseries.csv")
    df.to_csv(csv_fp, index=False, na_rep="N/A")

def handle_eaf_fn(eaf_fn: FilePath, video_id: str, start_frame: int, end_frame: int, video_fps: Number) -> FilePath:
    """
    Generate EAF filename if not provided, using video ID and time range.
    
    Args:
        eaf_fn (FilePath): EAF filename. If empty, generates new filename.
        video_id (str): Video identifier for filename generation.
        start_frame (int): Starting frame number.
        end_frame (int): Ending frame number.
        video_fps (Number): Video frames per second for time conversion.
    
    Returns:
        FilePath: EAF filename, either provided or generated.
    """
    if eaf_fn == "":
        start_sec = round(start_frame / video_fps, 1)
        end_sec = round(end_frame / video_fps, 1)
        eaf_fn = video_id + "-" + str(start_sec) + "-" + str(end_sec) + ".eaf"
    return eaf_fn

def create_working_folders(args: Namespace) -> None:
    """
    Create necessary working folders for video processing and annotation.
    
    Args:
        args (Namespace): Command line arguments containing video_id and save_to path.
    
    Returns:
        tuple: Paths to (save_to_fd, elans_folder, csvs_folder).
    """
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