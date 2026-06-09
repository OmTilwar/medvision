"""
Export segmentation masks and detection records to structured JSON / GeoJSON.

Useful for downstream mapping, analytics, and GIS-style ingestion pipelines.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def mask_to_bbox(mask: np.ndarray, class_id: int = 0, min_area: int = 1) -> dict[str, Any] | None:
    """
    Compute an axis-aligned bounding box for a binary mask.

    Returns None if the mask has no foreground pixels above min_area.
    """
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None

    area = int(binary.sum())
    if area < min_area:
        return None

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    return {
        "class_id": class_id,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": xmax - xmin + 1,
        "height": ymax - ymin + 1,
        "area": area,
        "confidence": None,
    }


def bbox_to_geojson_feature(bbox: dict[str, Any], properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap a bbox dict as a GeoJSON Feature with a rectangular Polygon."""
    xmin, ymin = bbox["xmin"], bbox["ymin"]
    xmax, ymax = bbox["xmax"], bbox["ymax"]
    props = {"class_id": bbox["class_id"], "area": bbox["area"]}
    if properties:
        props.update(properties)
    if bbox.get("confidence") is not None:
        props["confidence"] = bbox["confidence"]

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin],
            ]],
        },
        "properties": props,
    }


def masks_to_geojson(masks: list[np.ndarray], class_ids: list[int] | None = None) -> dict[str, Any]:
    """Convert a list of binary masks to a GeoJSON FeatureCollection."""
    class_ids = class_ids or list(range(len(masks)))
    features = []
    for mask, class_id in zip(masks, class_ids):
        bbox = mask_to_bbox(mask, class_id=class_id)
        if bbox is not None:
            features.append(bbox_to_geojson_feature(bbox))

    return {"type": "FeatureCollection", "features": features}


def detections_to_json(detections: list[dict[str, Any]], indent: int | None = 2) -> str:
    """Serialize a list of detection dicts (bbox + class + confidence) to JSON."""
    return json.dumps({"detections": detections}, indent=indent)


def masks_to_detection_json(masks: list[np.ndarray], class_ids: list[int] | None = None) -> str:
    """Serialize mask bounding boxes to a simple JSON detection list."""
    class_ids = class_ids or list(range(len(masks)))
    detections = []
    for mask, class_id in zip(masks, class_ids):
        bbox = mask_to_bbox(mask, class_id=class_id)
        if bbox is not None:
            detections.append(bbox)
    return detections_to_json(detections)
