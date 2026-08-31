"""Image preprocessing components."""

from __future__ import annotations

import re
import subprocess
import tempfile
from math import log1p
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import cv2
import numpy as np


def detect_orientation(image_path: str | Path) -> dict[str, Any]:
    """Return Tesseract OSD's clockwise correction angle and confidence."""
    command = ["tesseract", str(image_path), "stdout", "--psm", "0", "-l", "osd"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout
    rotation = re.search(r"Rotate:\s*(0|90|180|270)", output)
    confidence = re.search(r"Orientation confidence:\s*([\d.]+)", output)
    return {
        "detected_rotation": int(rotation.group(1)) if rotation else 0,
        "orientation_confidence": float(confidence.group(1)) if confidence else 0.0,
        "osd_succeeded": result.returncode == 0 and rotation is not None,
    }


def detect_orientation_by_ocr(image_path: str | Path) -> dict[str, Any]:
    """Fallback: choose the right-angle rotation with the strongest OCR evidence."""
    from receipt_ocr.ocr import run_tesseract

    scores: dict[int, float] = {}
    with Image.open(image_path) as opened, tempfile.TemporaryDirectory() as temp_dir:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.width > 1000:
            height = round(image.height * 1000 / image.width)
            image = image.resize((1000, height), Image.Resampling.LANCZOS)
        for rotation in (0, 90, 180, 270):
            candidate = image.rotate(-rotation, expand=True, fillcolor="white")
            candidate_path = Path(temp_dir) / f"rotation_{rotation}.jpg"
            candidate.save(candidate_path, quality=90)
            result = run_tesseract(candidate_path, language="vie+eng", psm=6)
            confidences = [float(word["confidence"]) for word in result["words"]]
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            scores[rotation] = mean_confidence * log1p(len(confidences))
    best_rotation = max(scores, key=scores.get)
    return {"fallback_rotation": best_rotation, "fallback_scores": scores}


def orientation_resize(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1600,
    max_upscale: float = 4.0,
    min_orientation_confidence: float = 5.0,
    use_ocr_fallback: bool = True,
) -> dict[str, Any]:
    """Correct confident right-angle rotation, resize, and save a copy."""
    source = Path(image_path)
    destination = Path(output_path)
    orientation = detect_orientation(source)
    with Image.open(source) as size_probe:
        source_is_landscape = size_probe.width > size_probe.height
    osd_confident = (
        orientation["osd_succeeded"]
        and orientation["orientation_confidence"] >= min_orientation_confidence
    )
    fallback: dict[str, Any] = {}
    if osd_confident:
        applied_rotation = orientation["detected_rotation"]
        orientation_method = "osd"
    elif use_ocr_fallback and not orientation["osd_succeeded"] and source_is_landscape:
        fallback = detect_orientation_by_ocr(source)
        applied_rotation = fallback["fallback_rotation"]
        orientation_method = "ocr_score_fallback"
    else:
        applied_rotation = 0
        orientation_method = "unchanged"

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        original_size = image.size
        if applied_rotation:
            image = image.rotate(-applied_rotation, expand=True, fillcolor="white")
        scale = 1.0
        if target_width > 0 and image.width < target_width:
            scale = min(target_width / image.width, max_upscale)
        if scale > 1.0:
            width = round(image.width * scale)
            height = round(image.height * scale)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        final_size = image.size
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_options = {"quality": 95} if destination.suffix.lower() in {".jpg", ".jpeg"} else {}
        image.save(destination, **save_options)

    return {
        **orientation,
        **fallback,
        "applied_rotation": applied_rotation,
        "orientation_method": orientation_method,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "output_width": final_size[0],
        "output_height": final_size[1],
        "resize_scale": scale,
    }


def orientation_resize_grayscale_clahe(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1600,
    max_upscale: float = 4.0,
    min_orientation_confidence: float = 5.0,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
) -> dict[str, Any]:
    """Apply Step 1, then grayscale and local contrast enhancement."""
    metadata = orientation_resize(
        image_path,
        output_path,
        target_width=target_width,
        max_upscale=max_upscale,
        min_orientation_confidence=min_orientation_confidence,
    )
    destination = Path(output_path)
    grayscale = cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise ValueError(f"Unable to read preprocessed image: {destination}")
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_grid_size, clahe_grid_size),
    )
    enhanced = clahe.apply(grayscale)
    if not cv2.imwrite(str(destination), enhanced):
        raise OSError(f"Unable to save preprocessed image: {destination}")
    return {
        **metadata,
        "grayscale_applied": True,
        "contrast_method": "clahe",
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_grid_size": clahe_grid_size,
    }


def orientation_resize_grayscale(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1600,
    max_upscale: float = 4.0,
    min_orientation_confidence: float = 5.0,
) -> dict[str, Any]:
    """Apply Step 1 and convert the result to grayscale without contrast changes."""
    metadata = orientation_resize(
        image_path,
        output_path,
        target_width=target_width,
        max_upscale=max_upscale,
        min_orientation_confidence=min_orientation_confidence,
    )
    destination = Path(output_path)
    grayscale = cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise ValueError(f"Unable to read preprocessed image: {destination}")
    if not cv2.imwrite(str(destination), grayscale):
        raise OSError(f"Unable to save preprocessed image: {destination}")
    return {**metadata, "grayscale_applied": True, "contrast_method": "none"}


def orientation_resize_crop(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1600,
    max_upscale: float = 4.0,
    min_orientation_confidence: float = 5.0,
    crop_padding: float = 0.03,
    min_receipt_area: float = 0.20,
) -> dict[str, Any]:
    """Apply Step 1, crop a confidently detected receipt, then normalize width."""
    metadata = orientation_resize(
        image_path,
        output_path,
        target_width=target_width,
        max_upscale=max_upscale,
        min_orientation_confidence=min_orientation_confidence,
    )
    destination = Path(output_path)
    image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read preprocessed image: {destination}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    image_area = width * height
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        area_ratio = contour_area / image_area
        rectangularity = contour_area / box_area if box_area else 0.0
        mask = cv2.drawContours(
            np.zeros((height, width), dtype="uint8"), [contour], -1, 255, -1
        )
        _, mean_saturation, mean_brightness, _ = cv2.mean(hsv, mask=mask)
        paper_like = mean_saturation <= 75 and mean_brightness >= 120
        if (
            min_receipt_area <= area_ratio <= 0.95
            and rectangularity >= 0.45
            and paper_like
        ):
            candidates.append((contour_area, (x, y, box_width, box_height)))

    crop_applied = False
    crop_box = (0, 0, width, height)
    if candidates:
        _, (x, y, box_width, box_height) = max(candidates, key=lambda item: item[0])
        pad_x = round(box_width * crop_padding)
        pad_y = round(box_height * crop_padding)
        left, top = max(0, x - pad_x), max(0, y - pad_y)
        right = min(width, x + box_width + pad_x)
        bottom = min(height, y + box_height + pad_y)
        cropped = image[top:bottom, left:right]
        crop_box = (left, top, right, bottom)
        crop_applied = True
        if cropped.shape[1] != target_width:
            scale = target_width / cropped.shape[1]
            cropped = cv2.resize(
                cropped,
                (target_width, round(cropped.shape[0] * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        image = cropped

    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Unable to save preprocessed image: {destination}")
    return {
        **metadata,
        "crop_applied": crop_applied,
        "crop_box": crop_box,
        "crop_padding": crop_padding,
        "crop_output_width": image.shape[1],
        "crop_output_height": image.shape[0],
    }


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _rotate_clockwise(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _resize_for_ocr(
    image: np.ndarray, target_width: int, max_upscale: float
) -> tuple[np.ndarray, float]:
    scale = 1.0
    if target_width > 0 and image.shape[1] < target_width:
        scale = min(target_width / image.shape[1], max_upscale)
    if scale > 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return image, scale


def _high_recall_masks(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.morphologyEx(
        cv2.Canny(blurred, 30, 120),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    paper = cv2.inRange(hsv, np.array((0, 0, 75)), np.array((179, 145, 255)))
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=2,
    )
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 9
    )
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=2,
    )
    return [canny, paper, adaptive]


def _high_recall_candidate_score(
    contour: np.ndarray,
    polygon: np.ndarray,
    image: np.ndarray,
    min_area_ratio: float,
) -> float | None:
    height, width = image.shape[:2]
    image_area = height * width
    area_ratio = cv2.contourArea(polygon) / image_area
    if not min_area_ratio <= area_ratio <= 0.99:
        return None
    ordered = _order_corners(polygon)
    candidate_width = max(
        np.linalg.norm(ordered[2] - ordered[3]), np.linalg.norm(ordered[1] - ordered[0])
    )
    candidate_height = max(
        np.linalg.norm(ordered[1] - ordered[2]), np.linalg.norm(ordered[0] - ordered[3])
    )
    if candidate_width < 0.18 * width or candidate_height < 0.18 * height:
        return None
    aspect = candidate_width / max(candidate_height, 1)
    if not 0.18 <= aspect <= 5.5:
        return None
    rectangle = cv2.minAreaRect(contour)
    rectangle_area = rectangle[1][0] * rectangle[1][1]
    rectangularity = min(1.0, cv2.contourArea(contour) / rectangle_area) if rectangle_area else 0.0
    mask = cv2.drawContours(
        np.zeros((height, width), dtype="uint8"), [polygon.astype("int32")], -1, 255, -1
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, brightness, _ = cv2.mean(hsv, mask=mask)
    if brightness < 65 or saturation > 155:
        return None
    paper_score = max(0.0, 1.0 - saturation / 155) * min(1.0, brightness / 170)
    center = polygon.reshape(4, 2).mean(axis=0)
    image_center = np.array((width / 2, height / 2))
    center_distance = np.linalg.norm(center - image_center) / np.linalg.norm(image_center)
    center_score = max(0.0, 1.0 - center_distance)
    return 0.40 * area_ratio + 0.25 * rectangularity + 0.20 * paper_score + 0.15 * center_score


def detect_receipt_high_recall(
    image: np.ndarray, min_area_ratio: float = 0.08, min_score: float = 0.38
) -> tuple[np.ndarray | None, float, str]:
    """Detect a receipt using the multi-mask/fallback strategy from processing_3."""
    scale = min(1.0, 1200 / max(image.shape[:2]))
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image.copy()
    )
    contours: list[np.ndarray] = []
    for mask in _high_recall_masks(working):
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(sorted(found, key=cv2.contourArea, reverse=True)[:40])
    candidates: list[tuple[float, np.ndarray, str]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        x, y, width, height = cv2.boundingRect(contour)
        signature = (round(x / 10), round(y / 10), round(width / 10), round(height / 10))
        if signature in seen:
            continue
        seen.add(signature)
        perimeter = cv2.arcLength(contour, True)
        polygon = None
        method = "four_point"
        for epsilon in (0.012, 0.018, 0.025, 0.035, 0.05, 0.07):
            approximation = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                polygon = approximation.reshape(4, 2)
                break
        if polygon is None:
            area_ratio = cv2.contourArea(contour) / (working.shape[0] * working.shape[1])
            rectangle = cv2.minAreaRect(contour)
            rectangle_area = rectangle[1][0] * rectangle[1][1]
            rectangularity = cv2.contourArea(contour) / rectangle_area if rectangle_area else 0.0
            if area_ratio >= max(0.12, min_area_ratio) and rectangularity >= 0.55:
                polygon = cv2.boxPoints(rectangle)
                method = "min_area_rect_fallback"
        if polygon is None:
            continue
        score = _high_recall_candidate_score(contour, polygon, working, min_area_ratio)
        if score is not None and score >= min_score:
            candidates.append((score, polygon / scale, method))
    if not candidates:
        return None, 0.0, "not_detected"
    score, polygon, method = max(candidates, key=lambda item: item[0])
    return polygon.astype("float32"), score, method


def _perspective_crop(image: np.ndarray, points: np.ndarray, padding: float) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_corners(points)
    width = round(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = round(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"), target
    )
    output = cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255))
    pad_x, pad_y = round(width * padding), round(height * padding)
    return cv2.copyMakeBorder(
        output, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def orientation_crop_resize(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1600,
    max_upscale: float = 4.0,
    min_orientation_confidence: float = 5.0,
    crop_padding: float = 0.01,
    min_receipt_area: float = 0.08,
    min_crop_score: float = 0.38,
    debug_path: str | Path | None = None,
) -> dict[str, Any]:
    """Canonical processing_3 logic: EXIF, orient, crop, pad, resize, save."""
    source, destination = Path(image_path), Path(output_path)
    with Image.open(source) as opened:
        rgb = np.array(ImageOps.exif_transpose(opened).convert("RGB"))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    original_height, original_width = image.shape[:2]
    if original_height >= original_width:
        rotation, orientation_method, orientation_confidence = 0, "portrait_unchanged", 0.0
    else:
        orientation = detect_orientation(source)
        orientation_confidence = float(orientation["orientation_confidence"])
        if orientation["osd_succeeded"] and orientation_confidence >= min_orientation_confidence:
            rotation, orientation_method = int(orientation["detected_rotation"]), "osd"
        else:
            fallback = detect_orientation_by_ocr(source)
            rotation, orientation_method = int(fallback["fallback_rotation"]), "ocr_score_fallback"
    oriented = _rotate_clockwise(image, rotation)
    polygon, crop_score, crop_method = detect_receipt_high_recall(
        oriented, min_receipt_area, min_crop_score
    )
    output = oriented if polygon is None else _perspective_crop(oriented, polygon, crop_padding)
    before_resize_height, before_resize_width = output.shape[:2]
    output, resize_scale = _resize_for_ocr(output, target_width, max_upscale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output):
        raise OSError(f"Unable to save preprocessed image: {destination}")
    if debug_path is not None:
        debug = oriented.copy()
        if polygon is not None:
            cv2.polylines(debug, [polygon.astype("int32")], True, (0, 255, 0), max(3, oriented.shape[1] // 250))
        debug_destination = Path(debug_path)
        debug_destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_destination), debug)
    return {
        "applied_rotation": rotation,
        "orientation_method": orientation_method,
        "orientation_confidence": orientation_confidence,
        "crop_applied": polygon is not None,
        "crop_method": crop_method,
        "crop_score": crop_score,
        "crop_corners": polygon.tolist() if polygon is not None else [],
        "original_width": original_width,
        "original_height": original_height,
        "before_resize_width": before_resize_width,
        "before_resize_height": before_resize_height,
        "output_width": output.shape[1],
        "output_height": output.shape[0],
        "resize_scale": resize_scale,
    }


def orientation_resize_perspective_crop(
    image_path: str | Path,
    output_path: str | Path,
    target_width: int = 1600,
    max_upscale: float = 4.0,
    min_orientation_confidence: float = 5.0,
    crop_padding: float = 0.01,
    min_receipt_area: float = 0.20,
    debug_path: str | Path | None = None,
) -> dict[str, Any]:
    """Canonical pipeline: orient, crop/rectify, pad, then normalize width."""
    metadata = orientation_resize(
        image_path,
        output_path,
        target_width=target_width,
        max_upscale=max_upscale,
        min_orientation_confidence=min_orientation_confidence,
    )
    destination = Path(output_path)
    image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read preprocessed image: {destination}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 40, 120)
    paper_mask = cv2.inRange(hsv, np.array((0, 0, 105)), np.array((179, 95, 255)))
    paper_edges = cv2.Canny(paper_mask, 30, 100)
    edges = cv2.bitwise_or(canny, paper_edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = width * height
    image_center = np.array((width / 2, height / 2))
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if not min_receipt_area <= area_ratio <= 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximations = [
            cv2.approxPolyDP(contour, epsilon * perimeter, True)
            for epsilon in (0.015, 0.02, 0.025, 0.03, 0.04)
        ]
        quadrilateral = next(
            (approximation for approximation in approximations if len(approximation) == 4 and cv2.isContourConvex(approximation)),
            None,
        )
        if quadrilateral is None:
            continue
        rectangle = cv2.minAreaRect(contour)
        rectangle_area = rectangle[1][0] * rectangle[1][1]
        rectangularity = area / rectangle_area if rectangle_area else 0.0
        mask = cv2.drawContours(np.zeros((height, width), dtype="uint8"), [quadrilateral], -1, 255, -1)
        _, saturation, brightness, _ = cv2.mean(hsv, mask=mask)
        if saturation > 90 or brightness < 105:
            continue
        center = quadrilateral.reshape(4, 2).mean(axis=0)
        center_distance = np.linalg.norm(center - image_center) / np.linalg.norm(image_center)
        center_score = max(0.0, 1.0 - center_distance)
        paper_score = max(0.0, 1.0 - saturation / 90) * min(1.0, brightness / 180)
        score = 0.40 * area_ratio + 0.25 * rectangularity + 0.20 * paper_score + 0.15 * center_score
        candidates.append((score, quadrilateral))

    crop_applied = False
    corners: list[list[float]] = []
    score = 0.0
    output = image
    selected = None
    if candidates:
        score, selected = max(candidates, key=lambda item: item[0])
        ordered = _order_corners(selected)
        top_left, top_right, bottom_right, bottom_left = ordered
        output_width = round(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
        output_height = round(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
        if output_width >= 200 and output_height >= 200:
            target = np.array(
                [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(ordered, target)
            output = cv2.warpPerspective(image, matrix, (output_width, output_height), borderValue=(255, 255, 255))
            pad_x, pad_y = round(output_width * crop_padding), round(output_height * crop_padding)
            output = cv2.copyMakeBorder(output, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            scale = target_width / output.shape[1]
            output = cv2.resize(output, (target_width, round(output.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)
            crop_applied = True
            corners = ordered.tolist()

    if debug_path is not None:
        debug = image.copy()
        if selected is not None:
            cv2.polylines(debug, [selected.astype("int32")], True, (0, 255, 0), 8)
        debug_destination = Path(debug_path)
        debug_destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_destination), debug)
    if not cv2.imwrite(str(destination), output):
        raise OSError(f"Unable to save preprocessed image: {destination}")
    return {
        **metadata,
        "crop_applied": crop_applied,
        "crop_method": "four_point_perspective",
        "crop_score": score,
        "crop_corners": corners,
        "crop_padding": crop_padding,
        "crop_output_width": output.shape[1],
        "crop_output_height": output.shape[0],
    }
