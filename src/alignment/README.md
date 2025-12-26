# Module 4: ROI Rectification & Fine-Grained Quality Assessment

**Status:** ✅ **Production Ready**

## Overview

Module 4 transforms detected container ID regions from arbitrary quadrilaterals into rectangular top-down views suitable for OCR, while simultaneously filtering out false positives through geometric and quality validation.

### Pipeline Architecture

```
Input: (Original Image, 4 Keypoints from Module 3)
   ↓
┌─────────────────────────────────────┐
│ Stage 1: Geometric Validation       │
│ - Validate aspect ratio ranges      │
│ - Reject non-text shapes             │
└─────────────┬───────────────────────┘
              ↓ PASS
┌─────────────────────────────────────┐
│ Stage 2: Perspective Rectification  │
│ - Warp quadrilateral to rectangle   │
│ - Preserve all content               │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Stage 3: Resolution Check            │
│ - Validate height >= 25px            │
│ - Ensure OCR readability             │
└─────────────┬───────────────────────┘
              ↓ PASS
┌─────────────────────────────────────┐
│ Stage 4: Quality Assessment          │
│ - Measure contrast (threshold: 50)  │
│ - Measure sharpness (threshold: 100)│
└─────────────┬───────────────────────┘
              ↓ PASS
Output: (PASS/REJECT, Rectified Image, Quality Metrics)
```

## Quick Start

### Basic Usage

```python
from src.alignment import AlignmentProcessor
import cv2

# Initialize processor (loads default config)
processor = AlignmentProcessor()

# Load image and keypoints from Module 3
image = cv2.imread("container.jpg")
keypoints = [[100, 150], [450, 140], [460, 220], [90, 230]]  # TL, TR, BR, BL

# Process through pipeline
result = processor.process(image, keypoints)

# Check result
if result.is_pass():
    print("✓ Ready for OCR!")
    cv2.imwrite("rectified.jpg", result.rectified_image)
    print(f"Quality - Contrast: {result.metrics.contrast:.1f}, "
          f"Sharpness: {result.metrics.sharpness:.1f}")
else:
    print(f"✗ Rejected: {result.get_error_message()}")
```

### Convenience Function

```python
from src.alignment import process_alignment

result = process_alignment(image, keypoints)
```

## Configuration

Thresholds are managed in [`config.yaml`](config.yaml):

```yaml
geometric:
  # Multiple acceptable aspect ratio ranges
  # Each range is [min, max] - accepts if within ANY range
  aspect_ratio_ranges:
    - [1.5, 10.0]  # Default: single wide range
    # Alternative examples:
    # - [2.0, 3.0]   # For single-line IDs
    # - [5.0, 9.0]   # For very wide formats

quality:
  min_height_px: 25         # Minimum OCR-readable height
  contrast_threshold: 50    # Local contrast (P95-P5)
  sharpness_threshold: 100  # Laplacian variance
  sharpness_normalized_height: 64

processing:
  use_grayscale_for_quality: true
  warp_interpolation: "linear"  # or "cubic"
```

### Adjusting Thresholds

To tune thresholds for your dataset:

1. Edit `config.yaml` directly
2. Or load and modify programmatically:

```python
from src.alignment.config_loader import load_config

config = load_config()

# Adjust quality thresholds
config.quality.contrast_threshold = 40  # More lenient
config.quality.sharpness_threshold = 80

# Modify aspect ratio ranges (accepts if in ANY range)
config.geometric.aspect_ratio_ranges = [(2.0, 3.0), (5.0, 9.0)]

processor = AlignmentProcessor(config=config)
```

## Module Components

### Core Files

| File                     | Purpose                          |
| ------------------------ | -------------------------------- |
| `processor.py`           | Main pipeline orchestrator       |
| `geometric_validator.py` | Aspect ratio validation          |
| `quality_assessor.py`    | Contrast & sharpness measurement |
| `config_loader.py`       | Configuration management         |
| `types.py`               | Data classes & enums             |
| `config.yaml`            | Tunable thresholds               |

### Key Functions

#### `AlignmentProcessor.process()`
Main pipeline executor with fail-fast strategy.

**Returns:** `AlignmentResult` containing:
- `decision`: PASS or REJECT
- `rectified_image`: Warped ROI (if passed geometry check)
- `metrics`: Quality measurements (if passed resolution check)
- `rejection_reason`: Specific failure reason
- `predicted_width`, `predicted_height`, `aspect_ratio`: Diagnostic info

#### Quality Metrics

**Contrast (Local Contrast):**
- Robust range: `P95 - P5` on grayscale histogram
- Measures text-background separation
- Threshold: 50 (higher than general image quality)

**Sharpness (Stroke Sharpness):**
- Variance of Laplacian on normalized image (64px height)
- Measures edge crispness of characters
- Threshold: 100

## Testing

```bash
# Run all alignment tests
uv run pytest tests/alignment/ -v

# Run specific test file
uv run pytest tests/alignment/test_processor.py -v

# Run with coverage
uv run pytest tests/alignment/ --cov=src/alignment --cov-report=term
```

**Test Coverage:** 44 unit tests covering:
- Geometric validation (edge cases, invalid inputs)
- Quality assessment (contrast, sharpness, resolution)
- Configuration loading & validation
- Full pipeline integration (fail-fast behavior)

## Demo

Demo script has been removed to keep repository clean. See tests in `tests/alignment/` for usage examples.

## Technical Details

### Geometric Validation

**Aspect Ratio Formula:**
$$R = \frac{\max(W_{top}, W_{bottom})}{\max(H_{left}, H_{right})}$$

**Validation Logic:**
- Accepts if aspect ratio falls within **any** of the configured ranges
- Default range: $1.5 \leq R \leq 10.0$
- Supports multiple ranges for different text formats (e.g., single-line vs multi-line IDs)

**Common Rejection Reasons:**
- $R < 1.5$: Too square/vertical (logos, port markers, vertical text)
- $R > 10.0$: Too wide (scratches, container edges, horizontal lines)
- Between gaps: Falls outside all configured ranges

### Perspective Transformation

Uses OpenCV's `getPerspectiveTransform` and `warpPerspective`:

1. Order keypoints consistently (TL → TR → BR → BL)
2. Calculate destination rectangle size from edge lengths
3. Compute transformation matrix
4. Warp image with selected interpolation method

**Note:** Utilizes `extract_and_rectify_roi()` from `src.utils.image_rectification`.

### Quality Normalization

**Why normalize sharpness?**
- Laplacian variance depends on image resolution
- Resizing to fixed height (64px) ensures scale-independent measurement
- Allows consistent thresholds across different image sizes

## Integration Points

### Input from Module 3 (Localization)

Expected format:
```python
keypoints = np.array([
    [x1, y1],  # Top-Left
    [x2, y2],  # Top-Right
    [x3, y3],  # Bottom-Right
    [x4, y4]   # Bottom-Left
], dtype=np.float32)
```

**Order matters!** Points must be consistently ordered or use `order_points()` from `src.utils.image_rectification`.

### Output to Module 5 (OCR)

If `result.is_pass()`:
- Use `result.rectified_image` as OCR input
- Image is already perspective-corrected
- Quality-validated for optimal OCR performance

## Rejection Reasons

| Reason               | Stage | Description                | Solution                             |
| -------------------- | ----- | -------------------------- | ------------------------------------ |
| `INVALID_GEOMETRY`   | 1     | Aspect ratio out of bounds | Check localization accuracy          |
| `LOW_RESOLUTION`     | 3     | Character height < 25px    | Use higher resolution images         |
| `BAD_VISUAL_QUALITY` | 4     | Poor contrast or sharpness | Improve lighting, reduce motion blur |

## Performance Considerations

- **Fail-Fast Design:** Rejects early to avoid unnecessary computation
- **Typical Pass Rate:** 70-85% on well-captured images
- **Processing Time:** ~5-15ms per image (CPU)

## Future Enhancements

- [ ] Adaptive thresholding based on image statistics
- [ ] Multi-line text support (2-row container IDs)
- [ ] Rotation correction for upside-down text
- [ ] GPU acceleration for batch processing

## References

- Technical Specification: [`docs/modules/module-4-alignment/technical-specification.md`](../../../docs/modules/module-4-alignment/technical-specification.md)
- ISO 6346: Container ID format standard
- OpenCV Perspective Transform: [docs](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)

---

**Last Updated:** 2025-12-26  
**Version:** 1.0.0  
**Maintainer:** Container ID Research Team
