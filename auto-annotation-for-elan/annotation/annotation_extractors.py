from typing import Tuple

from datatypes import NumpyArray, TierAnnotation, Annotation
from elan.merge_labels import merge_label
from elan.finger_indices import get_finger_indices


def extract_wrists_annotation(frame_annotation: NumpyArray, axis: str) -> TierAnnotation:
    """
    Convert numerical wrist frame annotations to temporal segments with directional labels.
    
    Args:
        frame_annotation: Array of numerical frame annotations from handle_frame_annotation.
        axis (str): Axis identifier ('x' for horizontal, 'y' for vertical positioning).
    
    Returns:
        tuple: Three lists containing (start_times, end_times, labels) for segments
            lasting at least 5 frames. Labels depend on axis:
            - y-axis: "Down", "Centre", "Up" 
            - x-axis: "Right", "CentreRight", "CentreCentre", "CentreLeft", "Left"
    """
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

def extract_wrists_xy_annotation(frame_annotation_xy: Tuple[NumpyArray, NumpyArray], data: Annotation, annotation_tier: str) -> TierAnnotation:
    """
    Convert numerical wrist spatial frame annotations to temporal segments with spatial zone labels.
    
    Processes combined horizontal and vertical wrist position data to create spatial zone 
    annotations. Labels are merged from separate X and Y components and include data 
    quality indicators for segments with significant missing keypoint data.
    
    Args:
        frame_annotation_xy (Tuple[NumpyArray, NumpyArray]): Tuple of (x_annotations, y_annotations)
            from handle_frame_annotation processing both axes.
        data (Annotation): Raw keypoint data for missing value analysis.
        annotation_tier (str): Annotation tier name identifying wrist side.
    
    Returns:
        TierAnnotation: Three lists containing (start_times, end_times, labels) for all
            temporal segments. Labels are spatial combinations (e.g., "Right Up", "Centre Left")
            with "50%_missing" suffix added for segments with ≥50% missing keypoint data.
    """
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
        ) -> TierAnnotation:
    """
    Convert finger angle data to temporal segments with directional labels and data quality indicators.
    
    Processes finger angles to classify finger positions into directional categories.
    Includes nested angle2value function that converts angles to discrete position values
    based on axis-specific thresholds. Adds data quality suffixes for segments with
    significant missing keypoint data.
    
    Args:
        angles (NumpyArray): Array of finger angles in degrees from handle_angles.
        data (NumpyArray): Raw keypoint data for missing value analysis.
        annotation_tier (str): Annotation tier name identifying finger and hand side.
        axis (str): Axis identifier ('x' for lateral, 'y' for vertical movement).
    
    Returns:
        TierAnnotation: Three lists containing (start_times, end_times, labels) for
            segments lasting at least 5 frames. Labels are directional:
            - x-axis: "Left"/"Right" based on angle thresholds
            - y-axis: "Up"/"Down" based on angle thresholds
            Quality suffixes added: "50%_missing" or "90%_missing" for poor data segments.
    """
    def angle2value(angle: float, axis: str) -> int:
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

def extract_eyelid_annotation(frame_annotation: NumpyArray) -> TierAnnotation:
    """
    Convert numerical eyelid frame annotations to temporal segments identifying blink events.
    
    Processes eyelid state data to extract blink segments (value 0.0) that last longer
    than 3 frames. Uses a sliding window approach to detect state changes and filter
    for meaningful blink durations.
    
    Args:
        frame_annotation (NumpyArray): Array of numerical eyelid state values from handle_eyelid
            (0: blink, 1: open, 2: closed, -1: invalid).
    
    Returns:
        TierAnnotation: Three lists containing (start_times, end_times, labels) for
            blink segments lasting more than 3 frames. Only blink events (value 0.0)
            are extracted as annotations.
    """
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

def extract_eyebrows_annotation(frame_annotation: NumpyArray, axis: str) -> TierAnnotation:
    """
    Placeholder function for extracting eyebrow annotations from frame data.
    
    Args:
        frame_annotation (NumpyArray): Array of numerical eyebrow frame annotations 
            from handle_eyebrows.
        axis (str): Axis identifier for eyebrow movement analysis.
    
    Returns:
        TierAnnotation: Three lists of None values as placeholder. Function not implemented
            as eyebrow annotations are currently not used in the pipeline.
    """
    return [None], [None], [None]

