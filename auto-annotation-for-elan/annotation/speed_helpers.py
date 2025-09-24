from typing import Dict

import numpy as np

from datatypes import NumpyArray, SceneList
from elan.finger_indices import get_finger_indices

def calculate_speed(annotation_tier: str, axis: str, data: NumpyArray, scenes: SceneList) -> Dict:
    """
    Calculate speed annotation for wrist or finger keypoints along specified axis.
    
    Args:
        annotation_tier (str): Annotation tier name identifying body part and side.
        axis (str): Axis for speed calculation ('x' or 'y').
        data (NumpyArray): Keypoint data array.
        scenes (SceneList): List of scene frame ranges.
    
    Returns:
        Dict: Dictionary containing speed annotation data.
    
    Raises:
        ValueError: If keypoint name format is invalid.
    """
    if "wrist" in annotation_tier:
        keypoint_data = data["pose_keypoints_2d_norm"]
        keypoint_ind = 4 if "right" in annotation_tier else 7
    elif "finger" in annotation_tier or "thumb" in annotation_tier:
        keypoint_ind = get_finger_indices(annotation_tier)[1]
        if "left" in annotation_tier:
            keypoint_data = data["hand_left_keypoints_2d_norm"]
        elif "right" in annotation_tier:
            keypoint_data = data["hand_right_keypoints_2d_norm"]
        else:
            raise ValueError("Invalid keypoint name.")
    speed = calculate_axis_speed(
        axis, keypoint_data, keypoint_ind, scenes
        )
    return {"annotation": speed}

def calculate_axis_speed(axis: str, data: NumpyArray, joint_ind: int, scenes: SceneList) -> NumpyArray:
    """
    Calculate movement speed for a specific joint along given axis across scenes.
    
    Args:
        axis (str): Axis for speed calculation ('x', 'y', or other for magnitude).
        data (NumpyArray): Keypoint coordinate data.
        joint_ind (int): Index of the joint to calculate speed for.
        scenes (SceneList): List of scene frame ranges.
    
    Returns:
        NumpyArray: Speed values for each frame, zero outside scenes.
    """
    speed = np.zeros(len(data))
    if data.shape[-1] == 3:
        coords = data[:, joint_ind, :-1]
    elif data.shape[-1] == 2:
        coords = data[:, joint_ind]
    for start_frame, end_frame in scenes:
        scene_coords = coords[start_frame:end_frame]
        if scene_coords.size == 2:
            continue
        diff = np.roll(scene_coords, -1, axis=0) - scene_coords
        if axis == "x":
            scene_speed = diff[..., 0]
        elif axis == "y":
            scene_speed = diff[..., 1]
        else:
            scene_speed = np.linalg.norm(diff, axis=-1)
        scene_speed[-1] = scene_speed[-2]
        scene_speed[(np.roll(scene_coords, -1, axis=0) * scene_coords).sum(axis=1) == 0] = 0
        speed[start_frame:end_frame] = np.round(scene_speed, 6)
    return speed