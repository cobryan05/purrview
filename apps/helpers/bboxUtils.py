from __future__ import annotations

import math

import numpy as np


class BBox:
    EPSILON_DIST = 0.01
    EPSILON_SIZE = 0.10

    def __init__(self, coords: tuple(float, float, float, float)):
        """
        Parameters:
        coords (tuple[float, float, float, float]) - relative coordinates, (x1, y1, w, h)
        """
        # Ensure positive w,h
        x1, y1, w, h = coords
        if w < 0:
            x1 += w
            w = -w
        if h < 0:
            y1 += h
            h = -h
        self.bbox = np.array((x1, y1, w, h), dtype=float)  # Relative, x1y1, w,h

    def __eq__(self, other):
        return self.similar(other)

    def __repr__(self):
        return f"TrackerTools.BBox: {self.bbox}"

    def copy(self) -> BBox:
        return BBox(self.bbox)

    def dist(self, other: BBox) -> float:
        """returns the distance between the center of this bbox and another"""
        pt1 = self.bbox[:2]
        pt2 = other.bbox[:2]
        return math.dist(pt1, pt2)

    def similar(
        self,
        other: BBox,
        distEpsilon: float = EPSILON_DIST,
        sizeEpsilon: float = EPSILON_SIZE,
    ) -> bool:
        """returns if this bounding box is 'similar' to another
        Parameters:
        other: (BBox) the other bounding box to compare against
        distEpsilon (float, optional): maximum distance between to be considered similar bboxes
        sizeEpsilon (float, optional): maximum area difference to be considered similar bboxes

        """
        if self.bbox is other.bbox:
            return True
        absAreaDiff = abs(self.area - other.area)
        areaRatio = min(self.area, other.area) / max(self.area, other.area)
        if absAreaDiff > distEpsilon and areaRatio > sizeEpsilon:
            return False
        return self.dist(other) < distEpsilon

    @property
    def area(self) -> float:
        """area of the bounding box"""
        return self.bbox[2] * self.bbox[3]

    @staticmethod
    def fromX1Y1X2Y2(x1, y1, x2, y2, imgX, imgY) -> BBox:
        """create a new bounding box from 4 coordinates in image coordinates"""
        return BBox((x1 / imgX, y1 / imgY, (x2 - x1) / imgX, (y2 - y1) / imgY))

    @staticmethod
    def fromX1Y1WH(x1, y1, w, h, imgX, imgY) -> BBox:
        """create a new bounding box from top-left coords and width height in image coordinates"""
        return BBox((x1 / imgX, y1 / imgY, w / imgX, h / imgY))

    @staticmethod
    def fromYolo(cX, cY, w, h) -> BBox:
        """createa a new bounding box from yolo format (relative center coords, width and height)"""
        return BBox((cX - w / 2, cY - h / 2, w, h))

    @staticmethod
    def fromRX1Y1WH(x1, y1, w, h) -> BBox:
        """creates a new bounding box from relative top-left coords, width and height"""
        return BBox((x1, y1, w, h))

    @staticmethod
    def fromRX1Y1X2Y2(x1, y1, x2, y2) -> BBox:
        """createa a new bounding box from relative 4 coordinates"""
        return BBox((x1, y1, x2 - x1, y2 - y1))

    def asX1Y1X2Y2(self, imgW, imgH) -> tuple[int, int, int, int]:
        """returns bbox coords as 4 corners in image coordinates"""
        x1, y1, w, h = self.asX1Y1WH(imgW, imgH)
        return (x1, y1, x1 + w, y1 + h)

    def asX1Y1WH(self, imgW, imgH) -> tuple[int, int, int, int]:
        """returns bbox coords as top-left coords, width and height in image coordinates"""
        x1 = round(self.bbox[0] * imgW)
        y1 = round(self.bbox[1] * imgH)
        w = round(self.bbox[2] * imgW)
        h = round(self.bbox[3] * imgH)
        return (x1, y1, w, h)

    def asYolo(self) -> tuple[float, float, float, float]:
        """returns bbox coords as relative center coords, width and height"""
        rX, rY, rW, rH = self.bbox
        cX, cY = rX + rW / 2, rY + rH / 2
        return (cX, cY, rW, rH)

    def asRX1Y1WH(self) -> tuple[float, float, float, float]:
        """returns bbox coords as relative top-left coords, width and height"""
        return tuple(self.bbox)

    def asRX1Y1X2Y2(self) -> tuple[float, float, float, float]:
        """returns bbox coords as relative 4 coordinates"""
        x1, y1, w, h = self.asRX1Y1WH()
        return (x1, y1, x1 + w, y1 + h)
