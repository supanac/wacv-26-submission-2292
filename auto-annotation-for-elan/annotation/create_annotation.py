from typing import Tuple, Dict

from datatypes import NumpyArray, SceneList
from .directions_helpers import calculate_directions
from .distances_helpers import calculate_distances
from .pseudolabels_helpers import calculate_pseudolabels
from .speed_helpers import calculate_speed

def parse_tier_name(tier_name: str) -> Tuple[str, str, float]:
    """
    Parse tier name to extract processing level, axis, and threshold parameters.
    
    Args:
        tier_name (str): Tier name with underscore-separated components indicating
            processing type and parameters.
    
    Returns:
        tuple: Three values (level, axis, threshold) where:
            - level (str): Processing type ("pseudolabel" for annotation/zones, 
              "direction", "speed", or "distance")
            - axis (str): Spatial axis ("x", "y", or "xy" for combined)
            - threshold (float): Threshold value (5e-3 if "thr" in name, else 0.0)
    """
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
        scenes: SceneList) -> Dict:
    """
    Create annotations for a tier by routing to appropriate calculation function based on tier type.
    
    Parses the tier name to determine processing level and delegates to the corresponding
    calculation function. Acts as a dispatcher for different annotation types.
    
    Args:
        annotation_tier (str): Name of annotation tier indicating processing type and parameters.
        data (NumpyArray): Keypoint data array for processing.
        scenes (SceneList): List of scene frame ranges.
    
    Returns:
        Dict: Annotation dictionary with structure dependent on processing level:
            - "direction": Movement direction annotations
            - "speed": Velocity-based annotations  
            - "pseudolabel": Spatial/positional label annotations
            - "distance": Distance-based annotations
    """
    level, axis, threshold = parse_tier_name(annotation_tier)
    if level == "direction":
        return calculate_directions(annotation_tier, axis, data, scenes, threshold)
    elif level == "speed":
        return calculate_speed(annotation_tier, axis, data, scenes)
    elif level == "pseudolabel":
        return calculate_pseudolabels(annotation_tier, axis, data, scenes)
    elif level == "distance":
        return calculate_distances(annotation_tier, axis, data, scenes)
