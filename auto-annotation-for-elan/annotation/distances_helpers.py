from typing import Dict

import numpy as np

from datatypes import NumpyArray, SceneList, LocalMinima, TierAnnotation
from .speed_helpers import calculate_speed

def calculate_distances(annotation_tier: str, axis: str, data: NumpyArray, scenes: SceneList) -> Dict:
    """
    Calculate distance-based annotations using movement intervals between local speed minima.
    
    Computes XY speed data to find local minima points, then calculates interval statistics
    between these points to generate distance annotations. Uses confidence scores to assess
    data quality during interval analysis.
    
    Args:
        annotation_tier (str): Annotation tier name identifying body part and side.
        axis (str): Axis parameter for interval statistics calculation.
        data (NumpyArray): Keypoint data array including confidence scores.
        scenes (SceneList): List of scene frame ranges for processing.
    
    Returns:
        Dict: Dictionary containing distance annotations under "annotation" key.
    """
    speed_annotation = calculate_speed(annotation_tier, "xy", data, scenes)
    speed = speed_annotation["annotation"]
    if "wrist" in annotation_tier:
        keypoint_data = data["pose_keypoints_2d_norm"]
        keypoint_ind = 4 if "right" in annotation_tier else 7
    confidences = data["confidence_scores"]
    local_minima = find_local_minima(speed, scenes)
    distances = calculate_interval_statistics(
        axis, keypoint_ind, keypoint_data, local_minima, speed, confidences
        )
    return {"annotation": distances}

def find_local_minima(speed: NumpyArray, scenes: SceneList) -> LocalMinima:
    """
    Identify local minima in speed data to define movement intervals within scenes.
    
    Finds points where speed reaches local minimum values using multiple detection criteria:
    strict local minima, transitions from movement to stationary periods, and transitions
    from stationary to movement periods. Filters out intervals that are too short (<5 frames)
    or contain only zero speed values.
    
    Args:
        speed (NumpyArray): Speed values for each frame.
        scenes (SceneList): List of scene frame ranges for processing.
    
    Returns:
        LocalMinima: List of tuples (start_frame, end_frame) representing valid movement intervals
            between local minima points, excluding intervals shorter than 5 frames or
            with zero total speed.
    """
    all_local_minima = list()
    for start_frame, end_frame in scenes:
        scene_speed = speed[start_frame:end_frame]
        scene_local_minima = [start_frame]
        for i in range(2, len(scene_speed)-2):
            if (
                (scene_speed[i-2] > scene_speed[i] < scene_speed[i+2]
                and
                scene_speed[i-1] > scene_speed[i] < scene_speed[i+1])
                or
                scene_speed[i-2] > scene_speed[i-1] > scene_speed[i] == scene_speed[i+1] == scene_speed[i+2] == 0
                or
                0 == scene_speed[i-2] == scene_speed[i-1] == scene_speed[i] < scene_speed[i+1] < scene_speed[i+2]
            ):
                scene_local_minima.append(start_frame + i)
        scene_local_minima.append(end_frame)
        for lb, rb in zip(scene_local_minima[:-1], scene_local_minima[1:]):
            if speed[lb:rb].sum() == 0 or rb - lb < 5:
                continue
            all_local_minima.append((lb,rb))
    return all_local_minima

def calculate_interval_statistics(
        axis: str,
        keypoint_ind: int,
        data: NumpyArray,
        local_minima: LocalMinima,
        speed: NumpyArray,
        confidences: NumpyArray
        ) -> TierAnnotation:
    """
    Calculate distance statistics for movement intervals between local minima points.
    
    Computes distances for each interval using different methods based on axis:
    cumulative speed sum for XY, coordinate displacement for single axes. Filters out
    intervals with low confidence scores or invalid coordinate data.
    
    Args:
        axis (str): Calculation method ('xy' for speed sum, 'x'/'y' for coordinate displacement).
        keypoint_ind (int): Index of keypoint to analyze.
        data (NumpyArray): Normalized keypoint coordinate data.
        local_minima (LocalMinima): List of (start_frame, end_frame) interval tuples.
        speed (NumpyArray): Speed values for each frame.
        confidences (NumpyArray): Confidence scores for keypoint detection quality.
    
    Returns:
        TierAnnotation: Three lists containing (start_times, end_times, distances) for
            valid intervals. Distances are rounded to 6 decimal places and converted to strings.
            Intervals with mean confidence <0.05 or NaN distances are excluded.
    """
    start_time_list = list()
    end_time_list = list()
    distances = list()
    coords = data[:, keypoint_ind, :-1]
    keypoint_conf = confidences[:, keypoint_ind]
    for i, (start_frame, end_frame) in enumerate(local_minima):
        if axis == "xy":
            dist = speed[start_frame:end_frame].sum()
        elif axis == "x":
            interval_coords = coords[start_frame:end_frame]
            if np.any(np.isnan(interval_coords[:, 0])):
                dist = np.nan
            else:
                dist = interval_coords[-1, 0] - interval_coords[0, 0]
        else:
            interval_coords = coords[start_frame:end_frame]
            if np.any(np.isnan(interval_coords[:, 0])):
                dist = np.nan
            else:
                dist = interval_coords[-1, 1] - interval_coords[0, 1]
        if keypoint_conf[start_frame:end_frame].mean() < 0.05:
            continue
        if np.isnan(dist):
            continue
        start_time_list.append(start_frame)
        end_time_list.append(end_frame)
        distances.append(str(np.round(dist, 6)))
    return start_time_list, end_time_list, distances