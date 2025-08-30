#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
import re
import requests

import pyproj
from lxml import etree as ET

# ----------------------------
# Math helpers
# ----------------------------
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
            
def wrap_angle_rad(a):
    while a <= -math.pi: a += 2*math.pi
    while a > math.pi: a -= 2*math.pi
    return a

class Rigid2D:
    def __init__(self, x0, y0, yaw_deg):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.theta = math.radians(float(yaw_deg))
        self.ct = math.cos(self.theta)
        self.st = math.sin(self.theta)

    def transform_xy(self, x, y):
        dx = float(x) - self.x0
        dy = float(y) - self.y0
        xr = dx*self.ct - dy*self.st
        yr = dx*self.st + dy*self.ct
        return xr, yr

    def transform_hdg(self, hdg_rad):
        return wrap_angle_rad(float(hdg_rad) + self.theta)

# ----------------------------
# CRS / transformer utilities
# ----------------------------

def parse_crs_from_header(root):
    geo_elem = root.find(".//geoReference")
    if geo_elem is None or not geo_elem.text.strip():
        raise RuntimeError("No <geoReference> found in header.")

    proj_str = geo_elem.text.strip()
    proj_str = proj_str.replace("<![CDATA[", "").replace("]]>", "")
    crs_map = pyproj.CRS.from_proj4(proj_str)

    # Extract false easting/northing and central meridian
    x0_orig = float(re.search(r"\+x_0=([0-9\.\-e]+)", proj_str).group(1))
    y0_orig = float(re.search(r"\+y_0=([0-9\.\-e]+)", proj_str).group(1))
    lon0 = float(re.search(r"\+lon_0=([0-9\.\-e]+)", proj_str).group(1))
    return crs_map, x0_orig, y0_orig, lon0

def getHeader(root):
    header = root.find("./header")
    return header.attrib


def get_transformer_from_header(root):
    crs_map, _, _, _ = parse_crs_from_header(root)
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    return pyproj.Transformer.from_crs(crs_wgs84, crs_map, always_xy=True)

# ----------------------------
# Header update
# ----------------------------
def update_all_georeferences(root, x0_map, y0_map, yaw_deg):
    """
    Find all <geoReference> elements anywhere under <header> and overwrite them
    with offsets relative to original CRS. Preserve pipeline/geoid if present.
    """
    crs_map, x0_orig, y0_orig, lon0 = parse_crs_from_header(root)
    x0_offset = x0_orig - x0_map
    y0_offset = y0_orig - y0_map

    header = root.find(".//header")
    if header is None:
        raise RuntimeError("No <header> found in XODR")

    # Compute main replacement proj string
    proj_str_main = f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=0.9996 +x_0={x0_offset:.3f} +y_0={y0_offset:.3f} +datum=WGS84 +units=m +no_defs"

    # Find all geoReference elements under header (recursive)
    geo_elements = header.xpath(".//geoReference")
    for geo_elem in geo_elements:
        old_text = geo_elem.text.strip()
        if old_text.startswith("+proj=pipeline"):
            # Replace only x_0 and y_0 in pipeline string
            new_text = re.sub(r"\+x_0=[0-9\.\-e]+", f"+x_0={x0_offset:.3f}", old_text)
            new_text = re.sub(r"\+y_0=[0-9\.\-e]+", f"+y_0={y0_offset:.3f}", new_text)
            geo_elem.text = ET.CDATA(new_text)
        else:
            geo_elem.text = ET.CDATA(proj_str_main)

    # Add or update applied yaw in header
    attr = header.find("./attribute[@name='yaw_offset_deg']")
    if attr is None:
        attr = ET.SubElement(header, "attribute")
        attr.set("name", "yaw_offset_deg")
    attr.set("value", f"{yaw_deg:.3f}")

# ----------------------------
# Declination
# ----------------------------

def get_declination_wmm(lat, lon, alt_m=0, date=None):
    import wmm2020
    if date is None:
        date = datetime.now()
    model = wmm2020.WMM()
    return model.calc(lat, lon, alt_m, date)['decl']

def get_declination_noaa(lat, lon, apikey='zNEw7', date=datetime.now()):
    ### Parameter 	Required 	Default 	Description
    ### key	            yes		            To get the API key, register at https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
    ### lat1	        yes		            decimal degrees or degrees minutes seconds: -90.0 to 90.0
    ### lon1	        yes		            decimal degrees or degrees minutes seconds: -180.0 to 180.0
    ### model		               WMM	    which magnetic reference model to use: 'WMM', 'WMMHR', or 'IGRF'
    ###                                     For 'EMM' use base url: https://emmcalc.geomag.info with paramater magneticComponent=d
    ### startYear		        current year	year part of calculation date; WMM: 2024-2029, WMMHR: 2024-2029, IGRF: 1590-2029
    ### startMonth		        current month	month part of calculation date: 1 - 12
    ### startDay		        current day	    day part of calculation date: 1 - 31
    ### resultFormat	        html	        format of calculation results: 'html', 'csv', 'xml', 'json'
    url = "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination"
    params = {
        "key": apikey,

        "lat1": lat,
        "lon1": lon,

        "resultFormat": "json",
        "startYear": date.year,
        "startMonth": date.month,
        "startDay": date.day,    

        # "model": "WMM"
        "model": "IGRF"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["result"][0]["declination"])

# ----------------------------
# Transform map geometries
# ----------------------------

def transform_opendrive(tree, transformer, origin_lat, origin_lon, yaw_deg, update_georef=False, update_extents=False):
    root = tree.getroot()
    x0_map, y0_map = transformer.transform(origin_lon, origin_lat)
    tf = Rigid2D(x0_map, y0_map, yaw_deg)

    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    def update_bbox(x, y):
        nonlocal minx, miny, maxx, maxy
        minx = min(minx, x); maxx = max(maxx, x)
        miny = min(miny, y); maxy = max(maxy, y)

    for elem in root.iter():
        tag = elem.tag.split('}')[-1].lower()

        # Skip local corners
        if tag == "cornerlocal":
            continue

        if 'x' in elem.attrib and 'y' in elem.attrib:
            x = float(elem.attrib['x'])
            y = float(elem.attrib['y'])
            xr, yr = tf.transform_xy(x, y)
            elem.attrib['x'] = f"{xr:.10f}"
            elem.attrib['y'] = f"{yr:.10f}"
            update_bbox(xr, yr)

            if 'hdg' in elem.attrib:
                hdg = float(elem.attrib['hdg'])
                elem.attrib['hdg'] = f"{tf.transform_hdg(hdg):.10f}"

    if update_extents and minx < float("inf"):
        header = root.find(".//header")
        if header is not None:
            header.set("west", f"{minx:.10f}")
            header.set("east", f"{maxx:.10f}")
            header.set("south", f"{miny:.10f}")
            header.set("north", f"{maxy:.10f}")

    if update_georef:
        update_all_georeferences(root, x0_map, y0_map, yaw_deg)

    return tree

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Reframe OpenDRIVE map to a new origin + yaw.")
    parser.add_argument("in_path", help="Input .xodr file (global UTM 32N).")
    parser.add_argument("--out", dest="out_path", default="", help="Output .xodr file (local frame).")
    parser.add_argument("--origin-lat", type=float, required=True, help="New origin latitude (WGS84).")
    parser.add_argument("--origin-lon", type=float, required=True, help="New origin longitude (WGS84).")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Yaw offset in degrees (CCW positive).")
    parser.add_argument("--update-georef", action="store_true", default=True)
    parser.add_argument("--update-extents", action="store_true", default=True)
    parser.add_argument("--declination-source", choices=["none","wmm","online"], default="none")
    parser.add_argument("--declination-date", default=None)

    args = parser.parse_args()
    if args.out_path == "":
        fname, ext = os.path.splitext(args.in_path)
        args.out_path = f"{fname}_reprojected{ext}"

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Parse map
    try:
        tree = ET.parse(str(in_path))
        header = getHeader(tree)
    except ET.XMLSyntaxError as e:
        print(f"XML parse error: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply magnetic declination
    if args.declination_source != "none":
        ### define date to use
        date = args.declination_date
        if date == None:
            ## take time from header

            date = datetime.strptime(header["date"], "%d-%m-%y")
        print(f"Magnetic declination modeled at time {date}")
        decl = 0.0
        if args.declination_source == "wmm":
            decl = get_declination_wmm(args.origin_lat, args.origin_lon, date=date)
            print(f"Magnetic declination (WMM) = {decl:.2f}°")
        elif args.declination_source == "online":
            decl = get_declination_noaa(args.origin_lat, args.origin_lon, date=date)
            print(f"Magnetic declination (NOAA online) = {decl:.2f}°")
    args.yaw_deg += decl
    print(f"Total yaw offset = {args.yaw_deg:.2f}°")

    root = tree.getroot()
    transformer = get_transformer_from_header(root)

    # Transform geometries
    tree = transform_opendrive(
        tree,
        transformer,
        args.origin_lat,
        args.origin_lon,
        args.yaw_deg,
        update_georef=args.update_georef,
        update_extents=args.update_extents
    )

    # Write output
    indent(tree.getroot())
    tree.write(str(args.out_path), pretty_print=True, encoding="utf-8", xml_declaration=True)
    print(f"Written: {args.out_path}")

if __name__ == "__main__":
    main()
