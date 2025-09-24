from os import path
from shutil import copyfile
from typing import List

import pympi

from constants import template_home, template_id
from datatypes import FilePath, FolderPath, EafFile, Annotations, Timeseries
from .extractors import add_annotation
from .xml_helpers import template2case_xml, handle_eaf_file


def create_elan(eaf_fn: FilePath) -> EafFile:
    """
    Create or load an ELAN Eaf object.
    
    Args:
        eaf_fn (FilePath): Path to Eaf file. Loads existing file or creates new Eaf object.
    
    Returns:
        EafFile: ELAN Eaf object, either loaded from file or newly created.
    """
    if path.isfile(eaf_fn):
        eaf = pympi.Elan.Eaf(eaf_fn)
    else:
        eaf = pympi.Elan.Eaf()
    return eaf

def write_tiers_to_eaf(eaf: EafFile, tiers: List, annotations: Annotations, start_frame: int, end_frame: int, video_fps: int) -> EafFile:
    """
    Add tiers and their annotations to an EAF object.
    
    Args:
        eaf (EafFile): ELAN EAF object to modify.
        tiers (List): List of tier names to create.
        annotations (Annotations): Annotation data to add to the tiers.
        start_frame (int): Starting frame number.
        end_frame (int): Ending frame number.
        video_fps (int): Video frames per second.
    
    Returns:
        EafFile: Modified EAF object with added tiers and annotations.
    """
    if tiers:
        for tier_name in tiers:
            eaf.add_tier(tier_name)
            add_annotation(eaf, tier_name, annotations, start_frame, end_frame, video_fps)
    return eaf

def write_eaf_file(elans_folder: FolderPath, eaf: EafFile, video_id: str, eaf_fn: FilePath) -> None:
    """
    Write EAF file to disk and apply template processing.
    
    Args:
        elans_folder (FolderPath): Directory to save the EAF file.
        eaf (EafFile): ELAN EAF object to write.
        video_id (str): Video identifier for filename generation.
        eaf_fn (FilePath): EAF filename to save as.
    
    Returns:
        None
    """
    extension = eaf_fn.split(".")[-1]
    eaf_fn_wo_extension = ".".join(eaf_fn.split(".")[:-1])
    video_name = video_id + "." + extension
    dst_eaf = path.join(elans_folder, eaf_fn)
    eaf.to_file(dst_eaf)
    template_eaf = path.join(template_home, "elans", template_id + ".eaf")
    xml_tree = handle_eaf_file(dst_eaf, template_eaf, eaf_fn_wo_extension, video_name)
    xml_tree.write(dst_eaf, xml_declaration=True)

def write_psfx_file(elans_folder: FolderPath, eaf_fn: FilePath) -> None:
    """
    Copy pfsx template file to ELAN folder with matching eaf filename.
    
    Args:
        elans_folder (FolderPath): Directory to save the PFSX file.
        eaf_fn (FilePath): EAF filename to generate corresponding PFSX filename.
    
    Returns:
        None
    """
    eaf_fn = ".".join(eaf_fn.split(".")[:-1])
    src_psfx = path.join(template_home, "elans", template_id + ".pfsx")
    dst_psfx = path.join(elans_folder, eaf_fn + ".pfsx")
    copyfile(src_psfx, dst_psfx)

def write_tsconf_file(elans_folder: FolderPath, eaf_fn: FilePath, timeseries: Timeseries) -> None:
    """
    Generate and write timeseries configuration XML file from template.
    
    Args:
        elans_folder (FolderPath): Directory to save the configuration file.
        eaf_fn (FilePath): EAF filename to generate corresponding tsconf filename.
        timeseries (Timeseries): Timeseries data for template processing.
    
    Returns:
        None
    """
    eaf_fn = ".".join(eaf_fn.split(".")[:-1])
    src_xml = path.join(template_home, "xmls", template_id + "_tsconf.xml")
    dst_xml = path.join(elans_folder, eaf_fn + "_tsconf.xml")
    tree = template2case_xml(src_xml, timeseries, eaf_fn)
    tree.write(dst_xml, xml_declaration=True)