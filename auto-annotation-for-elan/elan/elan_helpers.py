from os import path
from shutil import copyfile

import pympi

from constants import template_home, template_id
from .extractors import add_annotation
from .xml_helpers import template2case_xml, handle_eaf_file

def create_elan(eaf_fn):
    if path.isfile(eaf_fn):
        eaf = pympi.Elan.Eaf(eaf_fn)
    else:
        eaf = pympi.Elan.Eaf()
    return eaf

def write_tiers_to_eaf(eaf, tiers, annotations, start_frame, end_frame, video_fps):
    if tiers:
        for tier_name in tiers:
            eaf.add_tier(tier_name)
            add_annotation(eaf, tier_name, annotations, start_frame, end_frame, video_fps)
    return eaf

def write_eaf_file(elans_folder, eaf, video_id, eaf_fn):
    video_id = video_id
    extension = eaf_fn.split(".")[-1]
    eaf_fn_wo_extension = ".".join(eaf_fn.split(".")[:-1])
    video_name = video_id + "." + extension
    dst_eaf = path.join(elans_folder, eaf_fn)
    eaf.to_file(dst_eaf)
    template_eaf = path.join(template_home, "elans", template_id + ".eaf")
    xml_tree = handle_eaf_file(dst_eaf, template_eaf, eaf_fn_wo_extension, video_name)
    xml_tree.write(dst_eaf, xml_declaration=True)

def write_psfx_file(elans_folder, eaf_fn):
    eaf_fn = ".".join(eaf_fn.split(".")[:-1])
    src_psfx = path.join(template_home, "elans", template_id + ".pfsx")
    dst_psfx = path.join(elans_folder, eaf_fn + ".pfsx")
    copyfile(src_psfx, dst_psfx)

def write_tsconf_file(elans_folder, eaf_fn, timeseries):
    eaf_fn = ".".join(eaf_fn.split(".")[:-1])
    src_xml = path.join(template_home, "xmls", template_id + "_tsconf.xml")
    dst_xml = path.join(elans_folder, eaf_fn + "_tsconf.xml")
    tree = template2case_xml(src_xml, timeseries, eaf_fn)
    tree.write(dst_xml, xml_declaration=True)