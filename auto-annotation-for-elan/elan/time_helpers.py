from os import path
from typing import Tuple

from datatypes import Number, FilePath

def filename2timestamps(fn: FilePath) -> Tuple[float, float]:
    """
    Extract start and end timestamps from filename format.
    
    Args:
        fn (FilePath): Filename containing timestamps separated by hyphens.
    
    Returns:
        Tuple[float, float]: Start and end timestamps as floats.
    """
    fn_basename = path.basename(fn)
    splitted_basename = fn_basename.split("-")
    start_timestamp = splitted_basename[-2]
    end_timestamp = splitted_basename[-1]
    end_timestamp = ".".join(end_timestamp.split(".")[:-1])
    return float(start_timestamp), float(end_timestamp)

def timestamp2frame(timestamp: float, fps: Number):
    """
    Convert timestamp to frame number using frame rate.
    
    Args:
        timestamp (float): Time in seconds.
        fps (Number): Frames per second.
    
    Returns:
        Frame number rounded to nearest integer.
    """
    return round(timestamp * fps)

def filename2frames(fn: FilePath, fps: Number) -> Tuple[Number, Number]:
    """
    Extract start and end frame numbers from filename using frame rate.
    
    Args:
        fn (FilePath): Filename containing timestamps, or empty string for full range.
        fps (Number): Frames per second for timestamp conversion.
    
    Returns:
        Tuple[Number, Number]: Start and end frame numbers, or (0, None) if filename is empty.
    """
    if fn == "":
        start_frame = 0
        end_frame = None
    else:
        start_timestamp, end_timestamp = filename2timestamps(fn)
        start_frame = timestamp2frame(start_timestamp, fps)
        end_frame = timestamp2frame(end_timestamp, fps)
    return start_frame, end_frame