"""
Build src/ps315_profile.json from DICOM PS3.15 Table E.1-1
(Application Level Confidentiality Profile Attributes).

The table lists, for every attribute that may carry identifying information,
the action of the Basic Application Level Confidentiality Profile (D, Z, X, K,
C, U and combinations) and of each profile option. Generating the profile from
the standard's own DocBook source keeps the de-identifier traceable to PS3.15
instead of a hand-maintained tag list.

Usage:
    python scripts/build_ps315_profile.py                       # downloads part15.xml
    python scripts/build_ps315_profile.py --xml path/to/part15.xml
"""

import os
import sys
import json
import argparse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

PART15_URL = "https://dicom.nema.org/medical/dicom/current/source/docbook/part15/part15.xml"
DB = "{http://docbook.org/ns/docbook}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

# Table column header -> profile key
COLUMNS = {
    "Basic Prof.": "basic",
    "Rtn. Safe Priv. Opt.": "retain_safe_private",
    "Rtn. UIDs Opt.": "retain_uids",
    "Rtn. Dev. Id. Opt.": "retain_device_identity",
    "Rtn. Inst. Id. Opt.": "retain_institution_identity",
    "Rtn. Pat. Chars. Opt.": "retain_patient_characteristics",
    "Rtn. Long. Full Dates Opt.": "retain_longitudinal_full_dates",
    "Rtn. Long. Modif. Dates Opt.": "retain_longitudinal_modified_dates",
    "Clean Desc. Opt.": "clean_descriptors",
    "Clean Struct. Cont. Opt.": "clean_structured_content",
    "Clean Graph. Opt.": "clean_graphics",
}


def cell_text(element):
    return " ".join("".join(element.itertext()).split())


def parse_table(xml_path):
    root = ET.parse(xml_path).getroot()
    table = next(t for t in root.iter(f"{DB}table") if t.get(XML_ID) == "table_E.1-1")
    headers = [cell_text(th) for th in table.iter(f"{DB}th")]

    rows = []
    for tr in table.iter(f"{DB}tr"):
        cells = [cell_text(td) for td in tr.findall(f"{DB}td")]
        if len(cells) != len(headers):
            continue
        record = dict(zip(headers, cells))
        tag = record["Tag"].strip("()").upper()          # e.g. "0008,0050" or "50XX,XXXX"
        actions = {key: record[col] for col, key in COLUMNS.items() if record.get(col)}
        rows.append({"name": record["Attribute Name"], "tag": tag, "actions": actions})
    return rows


def parse_safe_private(xml_path):
    """Table E.3.10-1 Safe Private Attributes: (group, element low byte, creator)."""
    root = ET.parse(xml_path).getroot()
    table = next(t for t in root.iter(f"{DB}table") if t.get(XML_ID) == "table_E.3.10-1")
    safe = []
    for tr in table.iter(f"{DB}tr"):
        cells = [cell_text(td) for td in tr.findall(f"{DB}td")]
        if len(cells) < 2 or not cells[0].startswith("("):
            continue
        group, element = cells[0].strip("()").upper().split(",")
        safe.append({"group": group, "element": element[-2:], "creator": cells[1]})
    return safe


def main():
    parser = argparse.ArgumentParser(description="Generate the PS3.15 de-identification profile")
    parser.add_argument("--xml", default=None, help="Local part15.xml (downloaded if omitted)")
    parser.add_argument("--output", default=os.path.join(config.PROJECT_ROOT, "src", "ps315_profile.json"))
    args = parser.parse_args()

    xml_path = args.xml
    if xml_path is None:
        xml_path = os.path.join(config.DATA_DIR, "part15.xml")
        if not os.path.exists(xml_path):
            print(f"Downloading {PART15_URL}")
            urllib.request.urlretrieve(PART15_URL, xml_path)

    rows = parse_table(xml_path)
    safe_private = parse_safe_private(xml_path)
    profile = {
        "source": PART15_URL,
        "table": "PS3.15 Table E.1-1 Application Level Confidentiality Profile Attributes",
        "attributes": rows,
        "safe_private_table": "PS3.15 Table E.3.10-1 Safe Private Attributes",
        "safe_private": safe_private,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=1)
    print(f"Wrote {len(rows)} attributes and {len(safe_private)} safe private attributes to {args.output}")


if __name__ == "__main__":
    main()
