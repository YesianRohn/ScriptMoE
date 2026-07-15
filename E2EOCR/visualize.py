"""Box cropping and visualisation helpers (self-contained)."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# Crop helpers — used to extract per-box images for the recogniser
# -----------------------------------------------------------------------------
def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert len(points) == 4, 'points must be 4×2'
    crop_w = int(max(np.linalg.norm(points[0] - points[1]),
                     np.linalg.norm(points[2] - points[3])))
    crop_h = int(max(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [crop_w, 0],
                          [crop_w, crop_h], [0, crop_h]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(img, M, (crop_w, crop_h),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / max(dst.shape[1], 1) >= 1.5:
        dst = np.rot90(dst)
    return dst


def get_minarea_rect_crop(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    bbox = cv2.minAreaRect(np.array(points).astype(np.int32))
    pts = sorted(list(cv2.boxPoints(bbox)), key=lambda x: x[0])
    a, d = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    b, c = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    box = np.array([pts[a], pts[b], pts[c], pts[d]])
    return get_rotate_crop_image(img, box)


# -----------------------------------------------------------------------------
# Visualisation — side-by-side input / rendered text panel
# -----------------------------------------------------------------------------
def _create_font(txt, sz, font_path):
    font_size = max(int(sz[1] * 0.99), 6)
    font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
    try:
        if int(PIL.__version__.split('.')[0]) < 10:
            length = font.getsize(txt)[0]
        else:
            length = font.getlength(txt)
    except Exception:
        length = sz[0]
    if length > sz[0] and sz[0] > 0:
        font_size = max(int(font_size * sz[0] / length), 6)
        font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
    return font


def _draw_box_txt_fine(img_size, box, txt, font_path):
    bh = int(math.sqrt((box[0][0] - box[3][0]) ** 2 +
                       (box[0][1] - box[3][1]) ** 2))
    bw = int(math.sqrt((box[0][0] - box[1][0]) ** 2 +
                       (box[0][1] - box[1][1]) ** 2))

    if bh > 2 * bw and bh > 30:
        img_text = Image.new('RGB', (bh, bw), (255, 255, 255))
        if txt:
            font = _create_font(txt, (bh, bw), font_path)
            ImageDraw.Draw(img_text).text([0, 0], txt, fill=(0, 0, 0),
                                          font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new('RGB', (bw, bh), (255, 255, 255))
        if txt:
            font = _create_font(txt, (bw, bh), font_path)
            ImageDraw.Draw(img_text).text([0, 0], txt, fill=(0, 0, 0),
                                          font=font)

    pts1 = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_text = np.array(img_text, dtype=np.uint8)
    return cv2.warpPerspective(img_text, M, img_size,
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))


def draw_ocr_box_txt(image, boxes, txts=None, scores=None,
                     drop_score=0.5, font_path='./Arial_Unicode.ttf'):
    """Render a (input | reconstructed-text) side-by-side preview."""
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    rng = random.Random(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)

    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if isinstance(box[0], list):
            box = list(map(tuple, box))
        draw_left.polygon(box, fill=color)

        right_text = _draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, right_text)

    img_left = Image.blend(image, img_left, 0.5)
    canvas = Image.new('RGB', (w * 2, h), (255, 255, 255))
    canvas.paste(img_left, (0, 0, w, h))
    canvas.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(canvas)
