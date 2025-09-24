from os import path
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np

from datatypes import FilePath, XMLTree, XMLRoot, Timeseries, List, EafFile

def template2case_xml(src_xml_path: FilePath, timeseries, video_filename: str):
    """
    Convert XML template to case-specific configuration by updating source URL and min/max values.
    
    Args:
        src_xml_path: Path to source XML template file.
        timeseries: Timeseries data for min/max adjustment.
        video_filename: Video filename for source URL update.
    
    Returns:
        XML tree object with updated configuration.
    """
    tree, root = open_xml_file(src_xml_path)
    root = change_source_url(root, video_filename)
    root = adjust_min_max(root, timeseries) if timeseries else root
    return tree

def open_xml_file(src_xml_path: FilePath) -> Tuple[XMLTree, XMLRoot]:
    """
    Parse XML file and return tree and root elements.
    
    Args:
        src_xml_path (FilePath): Path to XML file to parse.
    
    Returns:
        Tuple[XMLTree, XMLRoot]: XML tree object and root element.
    """
    tree = ET.parse(src_xml_path)
    root = tree.getroot()
    return tree, root

def change_source_url(root: XMLRoot, video_filename: str) -> XMLRoot:
    """
    Update the source URL attribute in XML root element to point to timeseries CSV file.
    
    Args:
        root (XMLRoot): XML root element to modify.
        video_filename (str): Video filename for generating CSV path.
    
    Returns:
        ET.Element: Modified XML root element with updated source URL.
    """
    root[0].attrib["source-url"] = "file:///foo/" + video_filename + "_timeseries.csv"
    return root

def adjust_limits_for_wrists_position(root: XMLRoot, timeseries: Timeseries, timeseries_name: str) -> List[float, float]:
    """
    Calculate adjusted min/max limits for keypoint coordinates from timeseries.
    
    Args:
        root (XMLRoot): XML root element containing track information.
        timeseries (Timeseries): Timeseries data dictionary.
        timeseries_name (str): Name of timeseries to process.
    
    Returns:
        List[float, float]: Joint limits as [min_limit, max_limit] with 0.5 padding.
    """
    joint_limits = [0, 0]
    for elem in root[0]:
        if elem.tag == "track":
            series_name = elem.attrib["name"]
            if any(elem == series_name for elem in timeseries_name):
                ts = timeseries[series_name]
                ts = ts[(ts != 0) & (~np.isnan(ts))]
                if ts.size > 0:
                    ts_min, ts_max = ts.min() - 0.5, ts.max() + 0.5
                    if ts_min < joint_limits[0]:
                        joint_limits[0] = ts_min
                    if ts_max > joint_limits[1]:
                        joint_limits[1] = ts_max
    return joint_limits

def adjust_min_max(root: XMLRoot, timeseries: Timeseries) -> XMLRoot:
    """
    Adjust min/max attributes for track elements based on timeseries data and track type.
    
    Args:
        root (XMLRoot): XML root element containing track elements.
        timeseries (Timeseries): Timeseries data dictionary.
    
    Returns:
        XMLRoot: Modified XML root with updated min/max values for tracks.
    """
    for elem in root[0]:
        if elem.tag == "track":
            series_name = elem.attrib["name"]
            if series_name not in timeseries:
                continue
            if "eyebrow" in series_name.lower():
                ts_min = 0.75
                ts_max = 1.25
            elif "velocity" in series_name.lower():
                ts_min = .0
                ts_max = 0.19
            elif "wrist" in series_name.lower():
                if "lateral" in series_name.lower():
                    lateral_timeseries_names = [
                        "Right Wrist Lateral Position",
                        "Left Wrist Lateral Position"
                    ]
                    joint_lateral_limits = adjust_limits_for_wrists_position(
                        root, timeseries, lateral_timeseries_names
                        )
                    ts_min = joint_lateral_limits[0]
                    ts_max = joint_lateral_limits[1]
                elif "vertical" in series_name.lower():
                    vertical_timeseries_names = [
                        "Right Wrist Vertical Position",
                        "Left Wrist Vertical Position"
                    ]
                    joint_vertical_limits = adjust_limits_for_wrists_position(
                        root, timeseries, vertical_timeseries_names
                        )
                    ts_min = joint_vertical_limits[0]
                    ts_max = joint_vertical_limits[1]
            elem[4].attrib["min"] = str(ts_min)
            elem[4].attrib["max"] = str(ts_max)
    return root

def handle_eaf_file(eaf_file: EafFile, template_eaf: EafFile, eaf_filename: str, video_filename: str) -> XMLTree:
    """
    Process EAF file by applying template modifications and updating media descriptors.
    
    Args:
        eaf_file (EafFile): Path to EAF file to modify.
        template_eaf (EafFile): Path to template EAF file.
        eaf_filename (str): EAF filename for template processing.
        video_filename (str): Video filename for media descriptor.
    
    Returns:
        XMLTree: Modified XML tree with updated header and media descriptors.
    """
    modified_template = modify_templtae_eaf(template_eaf, eaf_filename, video_filename)
    modified_header = modified_template[0]
    tree, root = open_xml_file(eaf_file)
    header = root[0]
    for elem in modified_header:
        if elem.tag == "MEDIA_DESCRIPTOR":
            header[0].attrib = elem.attrib
        elif elem.tag == "LINKED_FILE_DESCRIPTOR":
            header.append(elem)
    return tree

def modify_templtae_eaf(template_eaf: FilePath, eaf_filename: str, video_filename: str) -> XMLRoot:
    """
    Modify EAF template by updating media descriptors and linked file paths.
    
    Args:
        template_eaf (FilePath): Path to template EAF file.
        eaf_filename (str): Target EAF filename for path generation.
        video_filename (str): Video filename for media URL updates.
    
    Returns:
        XMLRoot: Modified XML root with updated media and file descriptors.
    """
    video_filename = ".".join(path.basename(video_filename).split(".")[:-1])
    _, root = open_xml_file(template_eaf)
    header = root[0]
    for elem in header:
        if elem.tag == "MEDIA_DESCRIPTOR":
            src_rel_media_url = elem.attrib["RELATIVE_MEDIA_URL"]
            dst_media_url = "file:///foo/" + video_filename + ".mp4"
            dst_rel_media_url = \
                change_basename(src_rel_media_url, video_filename + ".mp4")
            elem.attrib["MEDIA_URL"] = dst_media_url
            elem.attrib["RELATIVE_MEDIA_URL"] = dst_rel_media_url
        elif elem.tag == "LINKED_FILE_DESCRIPTOR":
            if elem.attrib["MIME_TYPE"] == "text/xml":
                elem.attrib["LINK_URL"] = "file:///foo/" + eaf_filename + "_tsconf.xml"
                elem.attrib["RELATIVE_LINK_URL"] = "./" + eaf_filename + "_tsconf.xml"
            elif elem.attrib["MIME_TYPE"] == "unknown":
                csv_rel_link = elem.attrib["RELATIVE_LINK_URL"]
                elem.attrib["ASSOCIATED_WITH"] = "file:///foo/" + eaf_filename + ".mp4"
                elem.attrib["LINK_URL"] = "file:///foo/" + eaf_filename + "_timeseries.csv"
                elem.attrib["RELATIVE_LINK_URL"] = \
                    change_basename(csv_rel_link, eaf_filename + "_timeseries.csv")
    return root

def change_basename(src_path: FilePath, target_name: str) -> FilePath:
    """
    Replace the basename (filename) in a file path while keeping the directory structure.
    
    Args:
        src_path (FilePath): Original file path.
        target_name (str): New filename to replace the basename.
    
    Returns:
        FilePath: Modified file path with new basename.
    """
    return "/".join(src_path.split("/")[:-1] + [target_name])
