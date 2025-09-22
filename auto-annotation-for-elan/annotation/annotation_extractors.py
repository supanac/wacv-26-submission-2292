import numpy as np

from datatypes import NumpyArray
from elan.merge_labels import merge_label
from elan.finger_indices import get_finger_indices


def extract_wrists_annotation(frame_annotation, axis):
    start_time_list = list()
    end_time_list = list()
    label_list = list()
    in_progress = False
    label = ""
    label_value = 0
    start_time, end_time = 0, 0
    for i, current_value in enumerate(frame_annotation):
        if current_value == -1:
            if in_progress:
                end_time = i
                in_progress = False
                if end_time - start_time >= 5:
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    label_list.append(label)
            continue
        if not in_progress:
            label_value = current_value
            in_progress = True
            start_time = i
            if axis == "y":
                if current_value == 0:
                    label = "Down"
                elif current_value == 1:
                    label = "Centre"
                else:
                    label = "Up"
            else:
                if current_value == 0:
                    label = "Right"
                elif current_value == 1:
                    label = "CentreRight"
                elif current_value == 2:
                    label = "CentreCentre"
                elif current_value == 3:
                    label = "CentreLeft"
                else:
                    label = "Left"
        else:
            if current_value != label_value:
                end_time = i
                in_progress = False
                if end_time - start_time >= 5:
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    label_list.append(label)
    return start_time_list, end_time_list, label_list

def extract_wrists_xy_annotation(frame_annotation_xy, data, annotation_tier):
    keypoint_ind = 4 if "right" in annotation_tier else 7
    start_time_list = list()
    end_time_list = list()
    label_list = list()
    in_progress = False
    label = ""
    label_value = (0., 0.)
    start_time, end_time = 0, 0
    raw_data = data["pose_keypoints_2d"]
    for i, current_value in enumerate(zip(*frame_annotation_xy)):
        if any([elem == -1 for elem in current_value]):
            if in_progress:
                end_time = i
                in_progress = False
                if end_time - start_time >= 0:
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    missed_values = (raw_data[start_time:end_time, keypoint_ind, :] == 0).sum()
                    number_of_elems = raw_data[start_time:end_time, keypoint_ind, :].size
                    missed_percentage = missed_values / number_of_elems
                    if missed_percentage >= 0.5:
                        label = merge_label(label) + " 50%_missing"
                    else:
                        label = merge_label(label)
                    label_list.append(label)
            continue
        if not in_progress:
            label_value = current_value
            value_x, value_y = current_value
            in_progress = True
            start_time = i
            if value_y == 0:
                label_y = "Down"
            elif value_y == 1:
                label_y = "CentreDown"
            elif value_y == 2:
                label_y = "CentreY"
            elif value_y == 3:
                label_y = "CentreUp"
            else:
                label_y = "Up"
            if value_x == 0:
                label_x = "Right"
            elif value_x == 1:
                label_x = "CentreRight"
            elif value_x == 2:
                label_x = "CentreX"
            elif value_x == 3:
                label_x = "CentreLeft"
            else:
                label_x = "Left"
            label = label_x + " " + label_y
        else:
            if current_value != label_value:
                end_time = i
                in_progress = False
                if end_time - start_time >= 0:
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    missed_values = (raw_data[start_time:end_time, keypoint_ind, :] == 0).sum()
                    number_of_elems = raw_data[start_time:end_time, keypoint_ind, :].size
                    missed_percentage = missed_values / number_of_elems
                    if missed_percentage >= 0.5:
                        label = merge_label(label) + " 50%_missing"
                    else:
                        label = merge_label(label)
                    label_list.append(label)
    return start_time_list, end_time_list, label_list

def extract_fingers_annotation(
        angles: NumpyArray, 
        data: NumpyArray, 
        annotation_tier: str,
        axis: str
        ):
    def angle2value(angle: float, axis: str):
        if angle > 360:
            return -1
        if axis == "x":
            if abs(angle) > 110:
                return 0
            if abs(angle) < 70:
                return 1
        else:
            if -160 < angle < -20:
                return 0
            if 20 < angle < 160:
                return 1
        return -1

    finger_ind = get_finger_indices(annotation_tier)
    finger_tip_ind = finger_ind[1]
    if "left" in annotation_tier:
        key = "hand_left_keypoints_2d"
    elif "right" in annotation_tier:
        key = "hand_right_keypoints_2d"
    raw_data = data[key]
    start_time_list = list()
    end_time_list = list()
    label_list = list()
    left = "Left" if axis == "x" else "Up"
    right = "Right" if axis == "x" else "Down"
    in_progress = False
    label = ""
    label_value = 0
    start_time, end_time = 0, 0
    for i, current_angle in enumerate(angles):
        current_value = angle2value(current_angle, axis)
        if current_value == -1:
            if in_progress:
                end_time = i
                in_progress = False
                if end_time - start_time >= 5:
                    missed_values = (
                        raw_data[start_time:end_time, finger_tip_ind, :] == 0
                        ).sum()
                    number_of_elems = raw_data[
                        start_time:end_time, finger_tip_ind, :
                        ].size
                    missed_percentage = missed_values / number_of_elems
                    if missed_percentage >= 0.9:
                        label += " 90%_missing"
                    elif missed_percentage >= 0.5:
                        label += " 50%_missing"
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    label_list.append(label)
            continue
        if not in_progress:
            label_value = current_value
            in_progress = True
            start_time = i
            if current_value == 0:
                label = right
            elif current_value == 1:
                label = left
        else:
            if current_value != label_value:
                end_time = i
                in_progress = False
                if end_time - start_time >= 5:
                    missed_values = (
                        raw_data[start_time:end_time, finger_tip_ind, :] == 0
                        ).sum()
                    number_of_elems = raw_data[
                        start_time:end_time, finger_tip_ind, :
                        ].size
                    missed_percentage = missed_values / number_of_elems
                    if missed_percentage >= 0.9:
                        label += " 90%_missing"
                    elif missed_percentage >= 0.5:
                        label += " 50%_missing"
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    label_list.append(label)
    return start_time_list, end_time_list, label_list

def extract_eyelid_annotation(frame_annotation, axis):
    start_time_list = list()
    end_time_list = list()
    label_list = list()  
    start_ind = 0
    
    for i in range(len(frame_annotation)):
        if frame_annotation[i] != frame_annotation[start_ind]:
            if frame_annotation[start_ind] == 0.0 and i - start_ind > 3:
                start_time_list.append(start_ind)
                end_time_list.append(i)
                label_list.append(frame_annotation[start_ind])
            start_ind = i
    return start_time_list, end_time_list, label_list

def extract_eyebrows_annotation(frame_annotation, axis):
    return [None], [None], [None]

