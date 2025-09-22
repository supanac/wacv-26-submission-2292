import numpy as np

from datatypes import NumpyArray
from .speed_helpers import calculate_speed

def calculate_distances(
        annotation_tier: str, 
        axis: str, 
        data: NumpyArray, 
        scenes: list) -> tuple:
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

def find_local_minima(speed, scenes):
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
        local_minima: list, 
        speed: NumpyArray,
        confidences: NumpyArray
        ) -> dict:
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