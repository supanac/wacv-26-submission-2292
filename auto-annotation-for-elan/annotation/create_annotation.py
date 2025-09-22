from datatypes import NumpyArray
from .directions_helpers import calculate_directions
from .distances_helpers import calculate_distances
from .pseudolabels_helpers import calculate_pseudolabels
from .speed_helpers import calculate_speed

def parse_tier_name(tier_name):
    splitted_tier_name = tier_name.split("_")
    axis = None
    threshold = 0.
    if "annotation" in splitted_tier_name or "zones" in splitted_tier_name:
        level = "pseudolabel"
    elif "direction" in splitted_tier_name:
        level = "direction"
    elif "speed" in splitted_tier_name:
        level = "speed"
    elif "distance" in splitted_tier_name:
        level = "distance"
    if "thr" in splitted_tier_name:
            threshold = 5e-3
    if "x" in splitted_tier_name:
        axis = "x"
    elif "y" in splitted_tier_name:
        axis = "y"
    else:
        axis = "xy"
    return level, axis, threshold

def create_annotation(
        annotation_tier: str,
        data: NumpyArray,
        scenes: list) -> dict:
    level, axis, threshold = parse_tier_name(annotation_tier)
    if level == "direction":
        return calculate_directions(annotation_tier, axis, data, scenes, threshold)
    elif level == "speed":
        return calculate_speed(annotation_tier, axis, data, scenes)
    elif level == "pseudolabel":
        return calculate_pseudolabels(annotation_tier, axis, data, scenes)
    elif level == "distance":
        return calculate_distances(annotation_tier, axis, data, scenes)
