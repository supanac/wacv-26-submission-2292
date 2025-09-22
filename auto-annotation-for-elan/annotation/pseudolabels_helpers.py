import numpy as np

from constants import left_hand_fingers, right_hand_fingers
from datatypes import NumpyArray
from elan.finger_indices import get_finger_indices
from .annotation_extractors import (
    extract_wrists_annotation, extract_wrists_xy_annotation,
    extract_fingers_annotation, extract_eyebrows_annotation, 
    extract_eyelid_annotation
)


def calculate_pseudolabels(
        annotation_tier: str, 
        axis: str,
        data: NumpyArray, 
        scenes: list) -> dict:
    """ Calculates pseudo-labels for a requested tier.

        Returns pseudo-annotation for a requested keypoint.
        If requested a spatial annotation for wrists, returns
        two keys, {frame_annotation, annotation_xy}, otherwise returns
        three keys {frame_annotation, annotation_x, annotation_y}.
        frame_annotation represent numerical values which are transformed
        to actual pseudo-labels through extract_[KEYPOINT]_annotation.
        After processing numerical values to pseudo-labels, 
        the latter is stored in \"annotation_xy\" key or
        in a pair of keys, \"annotation_x\", \"annotation_y\".

    Args:
        annotation_ties: The name of a tier.
        data: Coordinates of keypoints.
        scenes: All scenes for a given video.

    Returns:
        The keys in the dictionary depend on the annotation_tier:
            - If annotation_tier contains "wrist_xy", returns 
            {"frame_annotation": frame_annotation, "annotation_xy": annotation_xy}.
            - Otherwise returns 
            {"frame_annotation": frame_annotation, "annotation_x": annotation_x, "annotation_y": annotation_y}.

    Note:
        See extract_[KEYPOINT]_annotation for details on annotation 
        return type.
    """
    if axis == "xy":
        frame_annotation_x = handle_frame_annotation(annotation_tier, "x", data, scenes)
        frame_annotation_y = handle_frame_annotation(annotation_tier, "y", data, scenes)
    else:
        frame_annotation = handle_frame_annotation(annotation_tier, axis, data, scenes)
    if "zones" in annotation_tier:
        frame_annotation = (frame_annotation_x, frame_annotation_y)
        annotation_xy = extract_wrists_xy_annotation(frame_annotation, data, annotation_tier)
        return {
            "frame_annotation": frame_annotation,
            "annotation": annotation_xy
        }
    elif "wrist" in annotation_tier:
        annotation = extract_wrists_annotation(frame_annotation, axis)
    elif "finger" in annotation_tier or "thumb" in annotation_tier:
        annotation = extract_fingers_annotation(frame_annotation, data, annotation_tier, axis)
    elif "eyebrow" in annotation_tier:
        # Is not called at the moment since we do not have any eyebrows'
        # annotation.
        # We do have timeseries for eyebrows. See elan_helpers.py.
        annotation = extract_eyebrows_annotation(frame_annotation, axis)
    elif "eyelid" in annotation_tier:
        annotation = extract_eyelid_annotation(frame_annotation, axis)
    return {
        "frame_annotation": frame_annotation,
        "annotation": annotation
    }

def handle_frame_annotation(annotation_tier: str, axis: str, data: NumpyArray, scenes: list) -> tuple:
    if "left_wrist" in annotation_tier or "right_wrist" in annotation_tier:
        calculate_func = handle_wrists
        data = data["pose_keypoints_2d_norm"]
    elif any([elem in annotation_tier for elem in left_hand_fingers]):
        calculate_func = handle_angles
        data = data["hand_left_keypoints_2d_norm"]
    elif any([elem in annotation_tier for elem in right_hand_fingers]):
        calculate_func = handle_angles
        data = data["hand_right_keypoints_2d_norm"]
    elif "left_eyebrow" in annotation_tier or "right_eyebrow" in annotation_tier:
        calculate_func = handle_eyebrows
        data = data["face_keypoints_2d_norm"]
    elif "left_eyelid" in annotation_tier or "right_eyelid" in annotation_tier:
        calculate_func = handle_eyelid
        data = data["face_keypoints_2d"]
    return calculate_func(annotation_tier, axis, data, scenes)

def handle_wrists(
        annotation_tier: str, 
        axis: str, 
        data: NumpyArray, 
        scenes: list
        ) -> NumpyArray:
    joint_ind = 4 if "right" in annotation_tier else 7
    frames_annotation = np.ones(len(data)) * (-1)
    for start_frame, end_frame in scenes:
        wrist_coords = data[start_frame:end_frame, joint_ind]
        if not np.any(wrist_coords):
            continue
        rshoulder_non_nan_inds = ~np.isnan(data[start_frame:end_frame, 2, :].sum(axis=1))
        if data[start_frame:end_frame, 2, :][rshoulder_non_nan_inds].size == 0:
            rshoulder_x = np.nan
        else:    
            mean_rshoulder = data[start_frame:end_frame, 2, :][rshoulder_non_nan_inds].mean(axis=0)
            rshoulder_x = mean_rshoulder[0]
        lshoulder_non_nan_inds = ~np.isnan(data[start_frame:end_frame, 5, :].sum(axis=1))
        if data[start_frame:end_frame, 5, :][lshoulder_non_nan_inds].size == 0:
            lshoulder_x = np.nan
        else:
            mean_lshoulder = data[start_frame:end_frame, 5, :][lshoulder_non_nan_inds].mean(axis=0)
            lshoulder_x = mean_lshoulder[0]
        neck_non_nan_inds = ~np.isnan(data[start_frame:end_frame, 1, :].sum(axis=1))
        if data[start_frame:end_frame, 1, :][neck_non_nan_inds].size == 0:
            neck_x = np.nan
        else:
            mean_neck = data[start_frame:end_frame, 1, :][neck_non_nan_inds].mean(axis=0)
            neck_x = mean_neck[0]
        right_offset = (rshoulder_x + neck_x) / 2
        left_offset = (lshoulder_x + neck_x) / 2
        for frame_ind in range(start_frame, end_frame):
            curr_ind = frame_ind - start_frame
            axis_ind = 0 if axis == "x" else 1
            coord = wrist_coords[curr_ind, axis_ind]
            if np.isnan(coord) or np.isnan(left_offset) or np.isnan(right_offset) or np.isnan(neck_x):
                continue
            neck_coords = data[frame_ind, 1]
            frame_annotation = calculate_axiswise_label(
                axis, coord, neck_coords[0], right_offset, left_offset
                )
            frames_annotation[frame_ind] = frame_annotation
    return frames_annotation

def handle_angles(
        annotation_tier: str, 
        axis: str, 
        data: NumpyArray, 
        scenes=None
        ) -> NumpyArray:
    finger_ind = get_finger_indices(annotation_tier)
    base_joint_ind, tip_joint_ind = finger_ind
    angles = np.ones(len(data)) * -1
    finger_vectors = data[:, tip_joint_ind] - data[:, base_joint_ind]
    angles = np.arctan2(finger_vectors[:, 1], finger_vectors[:, 0])
    angles = np.rad2deg(
        angles,
        where=np.any(finger_vectors, axis=1),
        out=np.ones(len(data))*1000
        )
    return angles

def handle_eyebrows(
        annotation_tier: str, 
        axis: str, 
        data: NumpyArray, 
        scenes: list
        ) -> tuple:
    joint_ind = 19 if "right" in annotation_tier else 24
    return data[:, joint_ind, 1]

def handle_eyelid(
        annotation_tier: str,
        axis: str,
        data: NumpyArray,
        scenes: list
        ) -> tuple:  
    def classify_blink_alternative(frames_data, lid_data, low_lid_data):
        blink_status = [1] * len(frames_data)     
        for i in range(1, len(lid_data) - 1):
            prev_y = lid_data[i - 1]
            current_y = lid_data[i]
            next_y = lid_data[i + 1]
            if lid_data[i] == 0:
                blink_status[i] = -1
            else:
                if lid_data[i] - low_lid_data[i]>=4:
                    continue
                else:     
                    if current_y < prev_y and current_y < next_y and (prev_y - current_y > 1 or next_y - current_y > 1) :
                        blink_start = i
                        blink_end = i    
                        while blink_start > 0 and (lid_data[blink_start - 1] - lid_data[blink_start] >= 1 or 
                                                   (blink_start > 1 and lid_data[blink_start-2] -lid_data[blink_start] >= 3) or 
                                                   (blink_start > 2 and lid_data[blink_start-3]-lid_data[blink_start] >=3.5)):
                            
                            blink_start -= 1
                        while blink_end < len(lid_data) - 1 and (lid_data[blink_end + 1] - lid_data[blink_end] >= 1 or 
                                                                 (blink_end +2 < len(lid_data) and lid_data[blink_end+2] -lid_data[blink_end]>=3) or 
                                                                 (blink_end +3 < len(lid_data) and lid_data[blink_end+3]-lid_data[blink_end]>=3.5)):
                            blink_end += 1
                        if blink_start != i:
                            speed_dwnhill = ((lid_data[blink_start]-lid_data[i])/(i-blink_start))
                        else:
                            speed_dwnhill = float('inf')
                        if blink_end != i:
                            speed_uphill = ((lid_data[blink_end]-lid_data[i])/(blink_end-i))
                        else:
                            speed_uphill = float('inf')
                        if speed_dwnhill <2 and speed_uphill <2 and (lid_data[i] - low_lid_data[i] > 3) :
                            continue
                        else:
                            for j in range(blink_start, blink_end + 1):
                                blink_status[j] = 0
        for i in range(1,len(blink_status)-1):
            if blink_status[i] != 0 and blink_status[i]!= -1: #Assuming it is not the identified blinks and it is not frames with no openpose outputs 
                if lid_data[i]-low_lid_data[i] < 3:
                    blink_status[i] = 2
        return blink_status
    
    upper_ind = 37 if "right" in annotation_tier else 43
    lower_ind = 41 if "right" in annotation_tier else 47
    frame_annotation = np.ones(len(data)) * (-1)
    for start_frame, end_frame in scenes:
        lid_data = data[start_frame:end_frame, upper_ind,1]
        low_lid_data = data[start_frame:end_frame, lower_ind,1]
        frames_data = list(range(start_frame, end_frame))
        blink_stat = classify_blink_alternative(frames_data, lid_data, low_lid_data)
        frame_annotation[start_frame :end_frame] = blink_stat
    return frame_annotation


def calculate_axiswise_label(axis, coord, neck_coord, right_offset, left_offset):
    if axis == "x":
        if coord == 0:
            frame_annotation = -1
        elif coord < neck_coord - abs(3*right_offset):
            frame_annotation = 0
        elif coord < neck_coord - abs(1.5*right_offset):
            frame_annotation = 1
        elif coord < neck_coord + abs(1.5*left_offset):
            frame_annotation = 2
        elif coord < neck_coord + abs(3*left_offset):
            frame_annotation = 3
        else:
            frame_annotation = 4
    elif axis == "y":
        if coord == 0:
            frame_annotation = -1
        elif coord < -3.5:
            frame_annotation = 0
        elif coord < -2.:
            frame_annotation = 1
        elif coord < -0.5:
            frame_annotation = 2
        elif coord < -0.:
            frame_annotation = 3
        else:
            frame_annotation = 4
    return frame_annotation