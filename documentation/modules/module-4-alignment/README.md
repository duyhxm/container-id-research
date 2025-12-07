# Module 4: Perspective Correction & Alignment

**Status**: 🔴 Not Yet Implemented  
**Priority**: Medium (depends on Module 3 completion)  
**Technology**: OpenCV

---

## Overview

This module applies perspective transformation to rectify the container ID region from the 4 keypoints predicted by Module 3, producing a frontal-view image suitable for OCR.

---

## Purpose

Transform the potentially skewed/angled ID region into a standardized, frontal-view rectangular image to maximize OCR accuracy.

---

## Technical Approach

### Perspective Transformation

Use OpenCV's `getPerspectiveTransform()` and `warpPerspective()`:

```python
import cv2
import numpy as np

def rectify_id_region(
    image: np.ndarray,
    keypoints: List[Tuple[float, float]],
    output_size: Tuple[int, int] = (400, 100)
) -> np.ndarray:
    """
    Rectify ID region using perspective transformation.
    
    Args:
        image: Source image
        keypoints: 4 keypoints [TL, TR, BR, BL]
        output_size: (width, height) of output image
    
    Returns:
        Rectified image
    """
    # Source points (from Module 3)
    src_pts = np.float32(keypoints)
    
    # Destination points (rectangle)
    w, h = output_size
    dst_pts = np.float32([
        [0, 0],           # Top-Left
        [w-1, 0],         # Top-Right
        [w-1, h-1],       # Bottom-Right
        [0, h-1]          # Bottom-Left
    ])
    
    # Compute homography matrix
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply perspective warp
    rectified = cv2.warpPerspective(image, H, output_size)
    
    return rectified
```

---

## Implementation Plan

### Phase 1: Core Functionality (Week 1)

- [ ] Implement perspective transformation function
- [ ] Handle edge cases (invalid keypoints, extreme angles)
- [ ] Add keypoint validation
- [ ] Unit tests

### Phase 2: Quality Assessment (Week 1)

- [ ] Implement quality checks on rectified image:
  - Blur detection
  - Contrast check
  - Aspect ratio validation
- [ ] Reject poor-quality rectifications

### Phase 3: Integration & Optimization (Week 1)

- [ ] Integrate with Module 3 output
- [ ] Integrate with Module 5 input
- [ ] Optimize for speed
- [ ] End-to-end testing

---

## Input/Output Specification

### Input

```json
{
  "source_image": "path/to/image.jpg",
  "keypoints": [
    [123.4, 56.7],  // Top-Left
    [345.6, 58.9],  // Top-Right
    [342.1, 89.0],  // Bottom-Right
    [125.8, 87.3]   // Bottom-Left
  ]
}
```

### Output

```json
{
  "rectified_image_path": "results/rectified_id.jpg",
  "output_size": [400, 100],
  "quality_score": 0.92,
  "quality_checks": {
    "blur_score": 0.88,
    "contrast_score": 0.95,
    "aspect_ratio_valid": true
  },
  "recommendation": "proceed_to_ocr"
}
```

---

## Configuration

### Output Size

Standard output dimensions:

```yaml
alignment:
  output_width: 400
  output_height: 100
```

**Rationale**:
- Container IDs are typically wider than tall
- 400x100 provides good resolution for OCR
- Aspect ratio ~4:1 matches typical ID plates

### Quality Thresholds

```yaml
alignment:
  quality_check:
    enabled: true
    min_blur_score: 50
    min_contrast_score: 30
    max_skew_angle: 5  # degrees
```

---

## Edge Cases & Handling

### Case 1: Invalid Keypoint Configuration

**Problem**: Keypoints form non-convex polygon or are in wrong order

**Solution**:
- Validate keypoint geometry before transformation
- Attempt to reorder keypoints if possible
- Reject if geometry is invalid

### Case 2: Extreme Perspective Angles

**Problem**: Very angled view leads to distorted output

**Solution**:
- Compute homography condition number
- Reject if perspective is too extreme
- Return warning to upstream modules

### Case 3: Output Quality Too Low

**Problem**: Rectified image is blurry or low contrast

**Solution**:
- Run quality checks on output
- Mark as "failed_quality_check"
- Don't pass to OCR module

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Processing Time | < 20ms per image |
| Success Rate | > 95% |
| Quality Score | > 0.80 (average) |

---

## Validation Strategy

### Synthetic Test Cases

Create synthetic test images with known transformations:

1. **Frontal View**: Should pass through with minimal change
2. **Mild Angle** (15°): Should rectify successfully
3. **Moderate Angle** (30°): Should rectify successfully
4. **Extreme Angle** (60°): May fail, should handle gracefully

### Real-World Validation

- Test on Module 3 predictions
- Measure OCR improvement with vs. without rectification
- Analyze failure cases

---

## Expected Impact

### Before Rectification
- OCR accuracy on angled images: ~60-70%

### After Rectification
- Expected OCR accuracy: ~90-95%

### Benefits
- Standardized input for OCR
- Reduced OCR errors
- Consistent image dimensions

---

## Dependencies

**Requires**:
- Module 3 (Container ID Localization) completed
- OpenCV installed

**Enables**:
- Module 5 (OCR Extraction) to work more reliably

---

## Next Steps

1. Wait for Module 3 completion
2. Implement core perspective transformation
3. Add quality checks
4. Integrate into pipeline
5. Measure end-to-end improvement

---

## References

- [OpenCV Perspective Transformation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Homography Estimation](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)

---

**Module Owner**: TBD  
**Estimated Start Date**: TBD (after Module 3)

