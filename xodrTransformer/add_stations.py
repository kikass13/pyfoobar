#!/usr/bin/env python3
import argparse
import os
import yaml
from lxml import etree as ET
from pathlib import Path

def load_stations(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)["stations"]

def add_stations_to_xodr(xodr_path, stations, out_path=None):
    tree = ET.parse(str(xodr_path))
    root = tree.getroot()  # <OpenDRIVE>

    print(f"Adding {len(stations)} stations to map: {xodr_path.name}")
    for station in stations:
        st_elem = ET.SubElement(root, "station")
        st_elem.set("id", station["id"])
        st_elem.set("name", station["name"])
        print(f"  - Added station '{station['name']}' [{station['id']}]")
        for pf in station.get("platforms", []):
            pf_elem = ET.SubElement(st_elem, "platform")
            pf_elem.set("id", pf["id"])
            pf_elem.set("name", pf["name"])
            print(f"      * platform '{pf['name']}' [{pf['id']}]")
            for i, seg in enumerate(pf.get("segments", [])):
                seg_elem = ET.SubElement(pf_elem, "segment")
                seg_elem.set("roadId", str(seg.get("roadId", 0)))
                seg_elem.set("sStart", str(seg.get("sStart", 0.0)))
                seg_elem.set("sEnd", str(seg.get("sEnd", 0.0)))
                seg_elem.set("side", seg.get("side", "right"))
                print(f"         > segment '{i}' "
                      f"roadId={seg.get('roadId',0)} sStart={seg.get('sStart',0.0)} "
                      f"sEnd={seg.get('sEnd',0.0)} side={seg.get('side','right')}")
    indent(root)
    if out_path is None:
        out_path = xodr_path.stem + "_with_stations.xodr"
    tree.write(str(out_path), pretty_print=True, encoding="utf-8", xml_declaration=True)
    print(f"Finished! Updated map written to: {out_path}")

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=Path)
    parser.add_argument("yaml_file", type=Path)
    parser.add_argument("--out", dest="out_path", default="", help="Output .xodr file")
    args = parser.parse_args()

    if args.out_path == "":
        fname, ext = os.path.splitext(args.in_path)
        args.out_path = f"{fname}_stations{ext}"

    stations = load_stations(args.yaml_file)
    add_stations_to_xodr(args.in_path, stations, args.out_path)
