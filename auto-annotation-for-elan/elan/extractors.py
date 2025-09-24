from collections import OrderedDict
from typing import Tuple, List

import numpy as np

from datatypes import NumpyArray, Number, Annotations, SceneList, EafFile, Timeseries


def extract_timeseries(
        data: NumpyArray, all_timeseries: List, start_frame: int, end_frame: int, video_fps: Number, *args, **kwargs
        ) -> Timeseries:
    """
    Extract multiple keypoint timeseries from data and add time column.
    
    Args:
        data (NumpyArray): Input keypoints data array.
        all_timeseries (List): List of timeseries names to extract.
        start_frame (int): Starting frame index.
        end_frame (int): Ending frame index.
        video_fps (Number): Video frames per second for time calculation.
        *args, **kwargs: Additional arguments passed to extraction function.
    
    Returns:
        Timeseries: Ordered dictionary with extracted timeseries and time column.
    """
    ret_dict = OrderedDict()
    for ts_name in all_timeseries:
        timeseries = extract_keypoints_timeseries(
            data, ts_name, start_frame, end_frame, *args, **kwargs
            )
        ret_dict[ts_name] = timeseries
    ret_dict["time"] = np.arange(len(timeseries)) / video_fps
    ret_dict.move_to_end("time", last=False)
    return ret_dict

def add_annotation(eaf: EafFile, tier_name: str, data: NumpyArray, start_frame: int, end_frame: int, video_fps: int) -> None:
    """
    Add parsed annotations from data to a specific tier in EAF file.
    
    Args:
        eaf (EafFile): ELAN EAF object to modify.
        tier_name (str): Name of the tier to add annotations to.
        data (NumpyArray): Annotation data array.
        start_frame (int): Starting frame index.
        end_frame (int): Ending frame index.
        video_fps (int): Video frames per second for time conversion.
    
    Returns:
        None
    """
    start_time_list, end_time_list, label_list = parse_data(
        tier_name, data, start_frame, end_frame
        )
    for start_time, end_time, label in zip(
        start_time_list, end_time_list, label_list
        ):
        if start_time == end_time:
            continue
        start_time_ms = int(1000 * start_time / video_fps)
        end_time_ms = int(1000 * end_time / video_fps)
        eaf.add_annotation(tier_name, start_time_ms, end_time_ms, label)
    return None

def parse_data(tier_name: str, annotations: Annotations, start_frame: int, end_frame: int) -> Tuple[List, List, List]:
    """
    Parse and filter annotation data within specified frame range.
    
    Args:
        tier_name (str): Name of annotation tier to extract.
        annotations (Annotations): Annotation data dictionary.
        start_frame (int): Starting frame for filtering.
        end_frame (int): Ending frame for filtering.
    
    Returns:
        Tuple[List, List, List]: Start times, end times, and labels within frame range.
    """
    annotation = extract_frame_annotation(tier_name, annotations)
    start_time_list = list()
    end_time_list = list()
    annotation_list = list()
    for start_time, end_time, label in zip(*annotation):
        if end_time < start_frame:
            continue
        if start_time > end_frame:
            break
        start_time_list.append(start_time)
        end_time_list.append(min(end_time, end_frame))
        annotation_list.append(label)
    return start_time_list, end_time_list, annotation_list

def extract_frame_annotation(tier_name: str, annotations: Annotations) -> NumpyArray:
    """
    Extract specific annotation data based on tier name.
    
    Args:
        tier_name (str): Name of the annotation tier to extract.
        annotations (Annotations): Annotation data container.
    
    Returns:
        NumpyArray: Annotation data array for the specified tier.
    """
    if "Left Wrist Vertical Direction Threshold" == tier_name:
        return annotations["left_wrist_direction_y_thr"][()]["annotation"]
    elif "Left Wrist Lateral Direction Threshold" == tier_name:
        return annotations["left_wrist_direction_x_thr"][()]["annotation"]
    elif "Right Wrist Vertical Direction Threshold" == tier_name:
        return annotations["right_wrist_direction_y_thr"][()]["annotation"]
    elif "Right Wrist Lateral Direction Threshold" == tier_name:
        return annotations["right_wrist_direction_x_thr"][()]["annotation"]
    elif "Left Wrist Vertical Zone" == tier_name:
        return annotations["left_wrist_annotation_y"][()]["annotation"]
    elif "Left Wrist Lateral Zone" == tier_name:
        return annotations["left_wrist_annotation_x"][()]["annotation"]
    elif "Right Wrist Vertical Zone" == tier_name:
        return annotations["right_wrist_annotation_y"][()]["annotation"]
    elif "Right Wrist Lateral Zone" == tier_name:
        return annotations["right_wrist_annotation_x"][()]["annotation"]
    elif "Left Wrist Zone" == tier_name:
        return annotations["left_wrist_zones"][()]["annotation"]
    elif "Right Wrist Zone" == tier_name:
        return annotations["right_wrist_zones"][()]["annotation"]
    elif "Left Wrist Distance" == tier_name:
        return annotations["left_wrist_distance"][()]["annotation"]
    elif "Left Wrist Distance Lateral" == tier_name:
        return annotations["left_wrist_distance_x"][()]["annotation"]
    elif "Left Wrist Distance Vertical" == tier_name:
        return annotations["left_wrist_distance_y"][()]["annotation"]
    elif "Right Wrist Distance" == tier_name:
        return annotations["right_wrist_distance"][()]["annotation"]
    elif "Right Wrist Distance Lateral" == tier_name:
        return annotations["right_wrist_distance_x"][()]["annotation"]
    elif "Right Wrist Distance Vertical" == tier_name:
        return annotations["right_wrist_distance_y"][()]["annotation"]
    elif "Left Index Finger Direction Lateral Auto" == tier_name:
        return annotations["left_index_finger_direction_x"][()]["annotation"]
    elif "Left Index Finger Direction Vertical Auto" == tier_name:
        return annotations["left_index_finger_direction_y"][()]["annotation"]
    elif "Left Middle Finger Direction Lateral Auto" == tier_name:
        return annotations["left_middle_finger_direction_x"][()]["annotation"]
    elif "Left Middle Finger Direction Vertical Auto" == tier_name:
        return annotations["left_middle_finger_direction_y"][()]["annotation"]
    elif "Left Ring Finger Direction Lateral Auto" == tier_name:
        return annotations["left_ring_finger_direction_x"][()]["annotation"]
    elif "Left Ring Finger Direction Vertical Auto" == tier_name:
        return annotations["left_ring_finger_direction_y"][()]["annotation"]
    elif "Left Little Finger Direction Lateral Auto" == tier_name:
        return annotations["left_little_finger_direction_x"][()]["annotation"]
    elif "Left Little Finger Direction Vertical Auto" == tier_name:
        return annotations["left_little_finger_direction_x"][()]["annotation"]
    elif "Left Thumb Direction Lateral Auto" == tier_name:
        return annotations["left_thumb_finger_direction_x"][()]["annotation"]
    elif "Left Thumb Direction Vertical Auto" == tier_name:
        return annotations["left_thumb_finger_direction_y"][()]["annotation"]
    elif "Right Index Finger Direction Lateral Auto" == tier_name:
        return annotations["right_index_finger_direction_x"][()]["annotation"]
    elif "Right Index Finger Direction Vertical Auto" == tier_name:
        return annotations["right_index_finger_direction_y"][()]["annotation"]
    elif "Right Middle Finger Direction Lateral Auto" == tier_name:
        return annotations["right_middle_finger_direction_x"][()]["annotation"]
    elif "Right Middle Finger Direction Vertical Auto" == tier_name:
        return annotations["right_middle_finger_direction_y"][()]["annotation"]
    elif "Right Ring Finger Direction Lateral Auto" == tier_name:
        return annotations["right_ring_finger_direction_x"][()]["annotation"]
    elif "Right Ring Finger Direction Vertical Auto" == tier_name:
        return annotations["right_ring_finger_direction_y"][()]["annotation"]
    elif "Right Little Finger Direction Lateral Auto" == tier_name:
        return annotations["right_little_finger_direction_x"][()]["annotation"]
    elif "Right Little Finger Direction Vertical Auto" == tier_name:
        return annotations["right_little_finger_direction_x"][()]["annotation"]
    elif "Right Thumb Direction Lateral Auto" == tier_name:
        return annotations["right_thumb_finger_direction_x"][()]["annotation"]
    elif "Right Thumb Direction Vertical Auto" == tier_name:
        return annotations["right_thumb_finger_direction_y"][()]["annotation"]
    elif "Left Index Finger Vertical Auto" == tier_name:
        return annotations["left_index_finger_annotation_y"][()]["annotation"]
    elif "Left Index Finger Lateral Auto" == tier_name:
        return annotations["left_index_finger_annotation_x"][()]["annotation"]
    elif "Left Middle Finger Vertical Auto" == tier_name:
        return annotations["left_middle_finger_annotation_y"][()]["annotation"]
    elif "Left Middle Finger Lateral Auto" == tier_name:
        return annotations["left_middle_finger_annotation_x"][()]["annotation"]
    elif "Left Ring Finger Vertical Auto" == tier_name:
        return annotations["left_ring_finger_annotation_y"][()]["annotation"]
    elif "Left Ring Finger Lateral Auto" == tier_name:
        return annotations["left_ring_finger_annotation_x"][()]["annotation"]
    elif "Left Little Finger Vertical Auto" == tier_name:
        return annotations["left_little_finger_annotation_y"][()]["annotation"]
    elif "Left Little Finger Lateral Auto" == tier_name:
        return annotations["left_little_finger_annotation_x"][()]["annotation"]
    elif "Left Thumb Vertical Auto" == tier_name:
        return annotations["left_thumb_finger_annotation_y"][()]["annotation"]
    elif "Left Thumb Lateral Auto" == tier_name:
        return annotations["left_thumb_finger_annotation_x"][()]["annotation"]
    elif "Right Index Finger Vertical Auto" == tier_name:
        return annotations["right_index_finger_annotation_y"][()]["annotation"]
    elif "Right Index Finger Lateral Auto" == tier_name:
        return annotations["right_index_finger_annotation_x"][()]["annotation"]
    elif "Right Middle Finger Vertical Auto" == tier_name:
        return annotations["right_middle_finger_annotation_y"][()]["annotation"]
    elif "Right Middle Finger Lateral Auto" == tier_name:
        return annotations["right_middle_finger_annotation_x"][()]["annotation"]
    elif "Right Ring Finger Vertical Auto" == tier_name:
        return annotations["right_ring_finger_annotation_y"][()]["annotation"]
    elif "Right Ring Finger Lateral Auto" == tier_name:
        return annotations["right_ring_finger_annotation_x"][()]["annotation"]
    elif "Right Little Finger Vertical Auto" == tier_name:
        return annotations["right_little_finger_annotation_y"][()]["annotation"]
    elif "Right Little Finger Lateral Auto" == tier_name:
        return annotations["right_little_finger_annotation_x"][()]["annotation"]
    elif "Right Thumb Vertical Auto" == tier_name:
        return annotations["right_thumb_finger_annotation_y"][()]["annotation"]
    elif "Right Thumb Lateral Auto" == tier_name:
        return annotations["right_thumb_finger_annotation_x"][()]["annotation"]
    elif "Right Eye Blinking" == tier_name:
        return annotations["right_eyelid_annotation_y"][()]["annotation"]
    elif "Left Eye Blinking" == tier_name:
        return annotations["left_eyelid_annotation_y"][()]["annotation"]

def extract_keypoints_timeseries(
        data: NumpyArray, keypoint_name: str, start_frame: int, end_frame: int, *args, **kwargs
        ) -> NumpyArray:
    """
    Extract specific keypoint timeseries data based on keypoint name and frame range.
    
    Args:
        data (NumpyArray): Input keypoints data array.
        keypoint_name (str): Name of the keypoint to extract.
        start_frame (int): Starting frame index.
        end_frame (int): Ending frame index.
        *args, **kwargs: Additional arguments including 'all_scenes' or 'annotations'.
    
    Returns:
        NumpyArray: Extracted keypoint timeseries with data only in specified frame range.
    
    Raises:
        ValueError: If keypoint name is not recognized.
    """
    if "all_scenes" in kwargs:
        all_scenes = kwargs["all_scenes"]
    else:
        all_scenes = kwargs["annotations"]["all_scenes"][()]
    if "Right Wrist Velocity" == keypoint_name:
        keypoints = data["pose_keypoints_2d_norm"][:, 4]
        if keypoints.shape[-1] == 3:
            keypoints = keypoints[..., :-1]
        diff = np.roll(keypoints, -1, axis=0) - keypoints
        ret_data = np.linalg.norm(diff, axis=-1)
        ret_data[(np.roll(keypoints, -1, axis=0) * keypoints).sum(axis=1) == 0] = 0
    elif "Left Wrist Velocity" == keypoint_name:
        keypoints = data["pose_keypoints_2d_norm"][:, 7]
        if keypoints.shape[-1] == 3:
            keypoints = keypoints[..., :-1]
        diff = np.roll(keypoints, -1, axis=0) - keypoints
        ret_data = np.linalg.norm(diff, axis=-1)
        ret_data[(np.roll(keypoints, -1, axis=0) * keypoints).sum(axis=1) == 0] = 0
    elif "Right Wrist Lateral Position" == keypoint_name:
        ret_data = data["pose_keypoints_2d_norm"][:, 4, 0]
    elif "Right Wrist Vertical Position" == keypoint_name:
        ret_data = data["pose_keypoints_2d_norm"][:, 4, 1]
    elif "Left Wrist Lateral Position" == keypoint_name:
        ret_data = data["pose_keypoints_2d_norm"][:, 7, 0]
    elif "Left Wrist Vertical Position" == keypoint_name:
        ret_data = data["pose_keypoints_2d_norm"][:, 7, 1]
    elif "Right Index Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 8, 0] \
                    - data["hand_right_keypoints_2d_norm"][:, 6, 0]
    elif "Right Index Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 8, 1] \
                    - data["hand_right_keypoints_2d_norm"][:, 6, 1]
    elif "Right Middle Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 12, 0] \
                    - data["hand_right_keypoints_2d_norm"][:, 10, 0]
    elif "Right Middle Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 12, 1] \
                    - data["hand_right_keypoints_2d_norm"][:, 10, 1]
    elif "Right Ring Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 16, 0] \
                    - data["hand_right_keypoints_2d_norm"][:, 14, 0]
    elif "Right Ring Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 16, 1] \
                    - data["hand_right_keypoints_2d_norm"][:, 14, 1]
    elif "Right Little Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 20, 0] \
                    - data["hand_right_keypoints_2d_norm"][:, 18, 0]
    elif "Right Little Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 20, 1] \
                    - data["hand_right_keypoints_2d_norm"][:, 18, 1]
    elif "Right Thumb Lateral Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 4, 0] \
                    - data["hand_right_keypoints_2d_norm"][:, 3, 0]
    elif "Right Thumb Vertical Position" == keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 4, 1] \
                    - data["hand_right_keypoints_2d_norm"][:, 3, 1]
    elif "Left Index Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 8, 0] \
                    - data["hand_left_keypoints_2d_norm"][:, 6, 0]
    elif "Left Index Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 8, 1] \
                    - data["hand_left_keypoints_2d_norm"][:, 6, 1]
    elif "Left Middle Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 12, 0] \
                    - data["hand_left_keypoints_2d_norm"][:, 10, 0]
    elif "Left Middle Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 12, 1] \
                    - data["hand_left_keypoints_2d_norm"][:, 10, 1]
    elif "Left Ring Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 16, 0] \
                    - data["hand_left_keypoints_2d_norm"][:, 14, 0]
    elif "Left Ring Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 16, 1] \
                    - data["hand_left_keypoints_2d_norm"][:, 14, 1]
    elif "Left Little Finger Lateral Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 20, 0] \
                    - data["hand_left_keypoints_2d_norm"][:, 18, 0]
    elif "Left Little Finger Vertical Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 20, 1] \
                    - data["hand_left_keypoints_2d_norm"][:, 18, 1]
    elif "Left Thumb Lateral Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 4, 0] \
                    - data["hand_left_keypoints_2d_norm"][:, 3, 0]
    elif "Left Thumb Vertical Position" == keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 4, 1] \
                    - data["hand_left_keypoints_2d_norm"][:, 3, 1]
    elif "Right Eyebrow Lateral Position" == keypoint_name:
        ret_data = extract_eyebrows_coords(keypoint_name, data, all_scenes)[:, 0]
    elif "Right Eyebrow Vertical Position" == keypoint_name:
        ret_data = extract_eyebrows_coords(keypoint_name, data, all_scenes)[:, 1]
    elif "Left Eyebrow Lateral Position" == keypoint_name:
        ret_data = extract_eyebrows_coords(keypoint_name, data, all_scenes)[:, 0]
    elif "Left Eyebrow Vertical Position" == keypoint_name:
        ret_data = extract_eyebrows_coords(keypoint_name, data, all_scenes)[:, 1]
    elif "Right Eyelid Distance Vertical" == keypoint_name:
        ret_data = data["face_keypoints_2d"][:, 37, 1]
    elif "Left Eyelid Distance Vertical" == keypoint_name:
        ret_data = data["face_keypoints_2d"][:, 43, 1]
    else:
        raise ValueError("Keypoint name was not found.")
    augmented_ret_data = np.zeros_like(ret_data)
    augmented_ret_data[start_frame:end_frame] = ret_data[start_frame:end_frame]
    return augmented_ret_data

def extract_eyebrows_coords(eyebrow: str, data: NumpyArray, all_scenes: SceneList) -> NumpyArray:
    """
    Extract and normalize eyebrow coordinates relative to eyelid position across scenes.
    
    Args:
        eyebrow (str): Eyebrow identifier ("Right" or "Left").
        data (NumpyArray): Face keypoints data array.
        all_scenes (SceneList): List of scene frame ranges.
    
    Returns:
        NumpyArray: Normalized eyebrow coordinates averaged across keypoints.
    """
    if "Right" in eyebrow:
        eyebrow_start_ind, eyebrow_end_ind = 17, 22
        eyelid_start_ind, eyelid_end_ind = 40, 42
    elif "Left" in eyebrow:
        eyebrow_start_ind, eyebrow_end_ind = 22, 27
        eyelid_start_ind, eyelid_end_ind = 46, 48
    face_data = data["face_keypoints_2d_norm"]
    for start_frame, end_frame in all_scenes:
        if face_data.shape[-1] == 3:
            scene_data = face_data[start_frame:end_frame, :, :-1]
        else:
            scene_data = face_data[start_frame:end_frame, ...]
        eyebrow_scene_data = scene_data[:, eyebrow_start_ind:eyebrow_end_ind, :].mean(axis=1)
        eye_scene_data = scene_data[:, eyelid_start_ind:eyelid_end_ind, :].mean(axis=1)
        nu = np.linalg.norm(eyebrow_scene_data-eye_scene_data, axis=1).mean()
        scene_data = (scene_data - eye_scene_data[:, None, :])/(nu + 1e-4)
        if face_data.shape[-1] == 3:
            face_data[start_frame:end_frame, :, :-1] = scene_data
        else:
            face_data[start_frame:end_frame] = scene_data
    ret_data = face_data[:, eyebrow_start_ind:eyebrow_end_ind, :].mean(axis=1)
    return ret_data