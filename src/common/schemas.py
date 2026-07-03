"""
Central Data Schema Module

This module defines the standardized data objects exchanged between all
modules of the Threat Assessment Framework. It serves as the single source
of truth for internal data contracts, ensuring type safety, immutability,
and maintainability.

All schemas are frozen dataclasses, making them hashable and safe to use
as keys or pass through multi-threaded pipelines without unintended mutations.

Architecture Constraint:
    This module sits at the very bottom of the dependency graph.
    It depends on nothing else in the project, and every other module depends on it.

Dependencies:
    - Python Standard Library only (dataclasses, typing, enum, pathlib)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from pathlib import Path


# -----------------------------------------------------------------------------
# Type Aliases (For cleaner signatures and future scalability)
# -----------------------------------------------------------------------------

DetectionTuple = Tuple["Detection", ...]
"""
Immutable tuple type for multiple detections.

This alias improves readability across the framework and provides a
centralized location to adjust the collection type if requirements change.
"""


# -----------------------------------------------------------------------------
# Enumerations (Standardized Vocabularies)
# -----------------------------------------------------------------------------

class ObjectType(str, Enum):
    """
    Standardized semantic types for all detectable entities.

    Expanded to include common COCO classes (BACKPACK, PHONE, CHAIR, DOG, BICYCLE)
    even though we may not use them in threat assessment. By including them now,
    we gracefully handle any detection YOLO returns without raising validation errors.

    Using granular types (e.g., KNIFE vs GUN) instead of a generic WEAPON
    preserves crucial semantic information for the Evidence Graph and
    downstream Threat Reasoning Engine.

    The naming convention (*Type) is used consistently across the framework
    (e.g., NodeType, RelationshipType, ThreatLevel) for easier navigation.
    """
    PERSON = "person"
    KNIFE = "knife"
    GUN = "gun"
    BAT = "bat"
    AXE = "axe"
    HAMMER = "hammer"
    BOTTLE = "bottle"
    BACKPACK = "backpack"
    PHONE = "phone"
    CHAIR = "chair"
    DOG = "dog"
    BICYCLE = "bicycle"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Core Data Models
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageInfo:
    """
    Immutable metadata descriptor for an input image or video frame.

    This object captures everything needed to identify and contextualize
    a single frame, independent of its pixel data.

    Attributes:
        image_id: Unique identifier (e.g., frame number, UUID, or filename stem).
        width: Image width in pixels.
        height: Image height in pixels.
        file_path: Optional filesystem path to the source image (for reproducibility).
        timestamp: Acquisition time in seconds since epoch (or video frame index).
        camera_id: Optional source camera identifier for multi-camera setups.
    """
    image_id: str
    width: int
    height: int
    file_path: Optional[Path] = None
    timestamp: float = 0.0
    camera_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate that image dimensions are physically plausible."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Image dimensions must be positive: {self.width}x{self.height}"
            )


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned rectangular bounding box in absolute pixel coordinates.

    This is a pure geometric container, completely decoupled from any
    specific object or detection confidence.

    Attributes:
        x_min: Top-left x-coordinate (inclusive).
        y_min: Top-left y-coordinate (inclusive).
        width: Box width in pixels (must be > 0).
        height: Box height in pixels (must be > 0).

    Properties (computed):
        x_max, y_max, area, center_x, center_y, aspect_ratio
    """
    x_min: float
    y_min: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """
        Validate that dimensions are strictly positive.

        A bounding box with zero width or height provides no useful spatial
        information and will break downstream geometric calculations (IoU,
        centering, area). Therefore, we reject them at the schema level.
        """
        if self.width <= 0:
            raise ValueError(f"Bounding box width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Bounding box height must be positive, got {self.height}")

    @property
    def x_max(self) -> float:
        """Right edge coordinate."""
        return self.x_min + self.width

    @property
    def y_max(self) -> float:
        """Bottom edge coordinate."""
        return self.y_min + self.height

    @property
    def area(self) -> float:
        """Surface area of the bounding box."""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """X-coordinate of the geometric center."""
        return self.x_min + self.width / 2.0

    @property
    def center_y(self) -> float:
        """Y-coordinate of the geometric center."""
        return self.y_min + self.height / 2.0

    @property
    def aspect_ratio(self) -> float:
        """
        Width-to-height ratio of the bounding box.

        Useful for distinguishing between standing persons (~0.4), vehicles (~2.0),
        or elongated weapons. Returns 0.0 if height is zero (though validation
        prevents this, the guard remains for safety).
        """
        if self.height == 0:
            return 0.0
        return self.width / self.height


@dataclass(frozen=True)
class ValidationResult:
    """
    Outcome of validating an input image or frame before processing.

    TODO (Future Enhancement): Currently, `is_valid=False` implies the image
    is unusable. In practice, a frame might have mild blur or low contrast but
    still be processable. A future iteration should restructure this to:
        passed: bool          # True if the image is usable
        warnings: Tuple[str]  # e.g., ("LOW_CONTRAST", "MOTION_BLUR")
        errors: Tuple[str]    # e.g., ("CORRUPTED_IMAGE",)

    For Sprint 0, we keep it simple but structure the warnings as a tuple
    to support multiple simultaneous issues.

    Attributes:
        is_valid: True if the image passes all pre-processing checks.
        warnings: Tuple of standardized warning codes (e.g., "LOW_BRIGHTNESS",
                  "MOTION_BLUR", "EXCESSIVE_NOISE", "OVEREXPOSED").
    """
    is_valid: bool
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Ensure warnings are non-empty only when is_valid is False."""
        if self.is_valid and self.warnings:
            raise ValueError(
                "A valid image should not have warnings. "
                "If there are warnings, is_valid must be False."
            )


@dataclass(frozen=True)
class Detection:
    """
    A single detected entity within an image frame.

    This object couples geometric (bbox) and semantic (type, confidence)
    information about one discrete object. The mandatory `detection_id`
    ensures the Evidence Graph can reference specific nodes unambiguously
    (e.g., "P_001" -> holds -> "W_001").

    Attributes:
        detection_id: Unique identifier for this detection instance (e.g., "P_001").
        bbox: Spatial location of the detection.
        confidence: Detector's confidence score, normalized to [0.0, 1.0].
        detector_class_id: The numeric class ID from the original detector's taxonomy
                           (e.g., YOLO's 0 for person, RT-DETR's different mapping).
                           This field belongs to the detector, not the framework.
        object_type: Standardized semantic type from the ObjectType enum.
        track_id: Optional tracking ID to link detections across frames.
    """
    detection_id: str
    bbox: BoundingBox
    confidence: float
    detector_class_id: int
    object_type: ObjectType
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate core attributes."""
        # Confidence validation
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Basic detection_id validation (non-empty, no whitespace)
        # TODO: Enforce strict format (e.g., P001, K001, V001) once the
        #       Evidence Graph construction logic is fully defined.
        if not self.detection_id or not self.detection_id.strip():
            raise ValueError("detection_id must be a non-empty string.")
        if " " in self.detection_id:
            raise ValueError(f"detection_id must not contain whitespace: {self.detection_id}")


@dataclass(frozen=True)
class DetectionResult:
    """
    Complete output bundle from an object detection module.

    This schema encapsulates all detections for a single frame along with
    the frame's metadata and performance telemetry. Using `DetectionTuple`
    (an alias for `tuple[Detection, ...]`) combined with `frozen=True`
    guarantees true immutability.

    Attributes:
        image_info: Metadata of the processed image.
        detections: Immutable tuple of all objects detected in the image.
        detector_name: Name/version of the detector used (e.g., "YOLO11", "RT-DETR").
        inference_time: Optional inference duration in milliseconds (for profiling).

    Properties (derived):
        total_count: Number of detections.
    """
    image_info: ImageInfo
    detections: DetectionTuple
    detector_name: str
    inference_time: Optional[float] = None

    @property
    def total_count(self) -> int:
        """Total number of detections in this result."""
        return len(self.detections)


# -----------------------------------------------------------------------------
# Explicit Exports (for cleaner imports)
# -----------------------------------------------------------------------------

__all__ = [
    "ObjectType",
    "DetectionTuple",
    "ImageInfo",
    "BoundingBox",
    "ValidationResult",
    "Detection",
    "DetectionResult",
]