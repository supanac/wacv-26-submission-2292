from typing import Dict

import numpy as np

from datatypes import NumpyArray, SceneList, TierAnnotation
from elan.finger_indices import get_finger_indices
from .speed_helpers import calculate_speed

def calculate_directions(
        annotation_tier: str, 
        axis: str,
        data: NumpyArray,
        scenes: SceneList,
        threshold: float=0) -> Dict:
    """
    Calculate movement direction annotations based on speed data and keypoint positions.
    
    Computes speed annotations first, then processes them with keypoint data to determine
    movement directions. Identifies appropriate keypoint indices based on body part type
    and delegates direction extraction to extract_directions_annotation.
    
    Args:
        annotation_tier (str): Annotation tier name identifying body part and side.
        axis (str): Axis for direction calculation ('x' or 'y').
        data (NumpyArray): Keypoint data array.
        scenes (SceneList): List of scene frame ranges.
        threshold (float, optional): Speed threshold for direction detection. Defaults to 0.
    
    Returns:
        Dict: Dictionary containing direction annotations under "annotation" key.
    
    Raises:
        ValueError: If keypoint name format is invalid for finger/thumb tiers.
    """
    speed_annotation = calculate_speed(annotation_tier, axis, data, scenes)
    speed = speed_annotation["annotation"]
    speed = np.nan_to_num(speed)
    if "wrist" in annotation_tier:
        keypoint_raw_data = data["pose_keypoints_2d"]
        keypoint_ind = 4 if "right" in annotation_tier else 7
    elif "finger" in annotation_tier or "thumb" in annotation_tier:
        keypoint_ind = get_finger_indices(annotation_tier)[1]
        if "left" in annotation_tier:
            keypoint_raw_data = data["hand_left_keypoints_2d"]
        elif "right" in annotation_tier:
            keypoint_raw_data = data["hand_right_keypoints_2d"]
        else:
            raise ValueError("Invalid keypoint name.")
    directions = extract_directions_annotation(
        speed, axis, scenes, keypoint_raw_data, keypoint_ind, threshold
        )
    return {"annotation": directions}

def extract_directions_annotation(
        speed: NumpyArray,
        axis: str,
        scenes: SceneList,
        raw_data: NumpyArray,
        keypoint_ind: int,
        threshold: float = 0
        ) -> TierAnnotation:
    """
    Extract directional movement annotations from speed data within scene boundaries.
    
    Processes speed data to identify consistent directional movements lasting at least 5 frames
    above the threshold. Detects direction changes by finding sign reversals in speed values.
    Includes nested get_label function for axis-specific directional labeling and adds data
    quality indicators for segments with significant missing keypoint data.
    
    Args:
        speed (NumpyArray): Speed values for each frame from calculate_speed.
        axis (str): Axis identifier ('x' for lateral, 'y' for vertical movement).
        scenes (SceneList): List of scene frame ranges for processing.
        raw_data (NumpyArray): Raw keypoint data for missing value analysis.
        keypoint_ind (int): Index of the keypoint to analyze for data quality.
        threshold (float, optional): Minimum speed threshold for direction detection. Defaults to 0.
    
    Returns:
        TierAnnotation: Three lists containing (start_times, end_times, labels) for
            directional movement segments. Labels are:
            - x-axis: "Left"/"Right" based on speed sign
            - y-axis: "Up"/"Down" based on speed sign
            Quality suffixes added: "50%_missing" or "90%_missing" for poor data segments.
    """
    def get_label(axis, value):
        if axis == "x":
            if value >= 0:
                return "Left"
            else:
                return "Right"
        else:
            if value >= 0:
                return "Down"
            else:
                return "Up"
            
    start_time_list = list()
    end_time_list = list()
    label_list = list()
    num_frames = len(raw_data)
    for start_frame, end_frame in scenes:
        if end_frame - start_frame < 3:
            continue
        start_time, end_time = start_frame, end_frame
        if np.all(speed[start_time + 1 : end_time - 1] == (-1) * np.ones(end_time - start_time - 2)):
            continue
        while start_time < num_frames and speed[start_time] == 0:
            start_time += 1
        end_time = start_time + 1
        while end_time < end_frame:
            if speed[end_time] * speed[start_time] < 0:
                if (
                    (end_time - start_time > 5)
                    and
                    (max(np.absolute(speed[start_time:end_time])) > threshold)
                ):
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    missed_values = (
                        (raw_data[start_time:end_time, keypoint_ind, :] == 0).sum()
                        +
                        np.isnan(raw_data[start_time:end_time, keypoint_ind, :]).sum()
                    )
                    number_of_elems = raw_data[start_time:end_time, keypoint_ind, :].size
                    missed_percentage = missed_values / number_of_elems
                    if missed_percentage >= 0.9:
                        label = get_label(axis, speed[start_time]) + " 90%_missing"
                    elif missed_percentage >= 0.5:
                        label = get_label(axis, speed[start_time]) + " 50%_missing" 
                    else:
                        label = get_label(axis, speed[start_time])
                    label_list.append(label)
                start_time = end_time
            end_time += 1
    return start_time_list, end_time_list, label_list