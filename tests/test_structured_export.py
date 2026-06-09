"""Tests for structured_export helpers."""

import json

import numpy as np

from src.structured_export import (
    bbox_to_geojson_feature,
    detections_to_json,
    mask_to_bbox,
    masks_to_geojson,
    masks_to_detection_json,
)


def test_mask_to_bbox_returns_none_for_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert mask_to_bbox(mask) is None


def test_mask_to_bbox_computes_coordinates():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 8:22] = 1
    bbox = mask_to_bbox(mask, class_id=2)
    assert bbox is not None
    assert bbox["class_id"] == 2
    assert bbox["xmin"] == 8
    assert bbox["ymin"] == 5
    assert bbox["xmax"] == 21
    assert bbox["ymax"] == 14
    assert bbox["area"] == 10 * 14


def test_bbox_to_geojson_feature_polygon():
    bbox = {"class_id": 1, "xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1, "area": 6}
    feature = bbox_to_geojson_feature(bbox, properties={"label": "region"})
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["label"] == "region"


def test_masks_to_geojson_feature_collection():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m1[1:4, 1:4] = 1
    collection = masks_to_geojson([m1], class_ids=[0])
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 1


def test_detections_to_json_roundtrip():
    payload = detections_to_json([{"class_id": 0, "xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4, "area": 4}])
    data = json.loads(payload)
    assert "detections" in data
    assert len(data["detections"]) == 1


def test_masks_to_detection_json():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    payload = masks_to_detection_json([mask], class_ids=[3])
    data = json.loads(payload)
    assert data["detections"][0]["class_id"] == 3
