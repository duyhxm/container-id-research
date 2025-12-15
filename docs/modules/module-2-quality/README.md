# Module 2: Image Quality Assessment

**Status**: 🔴 Not Yet Implemented  
**Priority**: High  
**Estimated Effort**: 2-3 weeks

---

## Overview

This module assesses the quality of detected container door images to determine if they are suitable for container ID extraction.

## Purpose

Filter out low-quality images that would likely fail in downstream OCR processing, improving overall pipeline efficiency and accuracy.

---

## Planned Approach

### Quality Metrics

1. **Blur Detection**
   - Method: Laplacian variance
   - Threshold: TBD (to be determined through experimentation)

2. **Brightness Assessment**
   - Method: Histogram analysis
   - Range: Check if mean brightness is within acceptable range

3. **Contrast Evaluation**
   - Method: Standard deviation of pixel values
   - Threshold: TBD

4. **Size Adequacy**
   - Method: Check bounding box dimensions
   - Minimum size: TBD pixels

5. **Overall Quality Score**
   - Aggregation of above metrics
   - Weighted combination or ML classifier

---

## Implementation Options

### Option 1: Rule-Based System (Recommended for MVP)

**Pros**:
- Fast and lightweight
- Interpretable
- No training required

**Cons**:
- May require manual threshold tuning
- Less flexible

### Option 2: ML-Based Classifier

**Pros**:
- More adaptive
- Can learn complex patterns

**Cons**:
- Requires labeled quality data
- More complex to deploy

---

## Data Requirements

### For Rule-Based System
- Analysis of current dataset to determine thresholds
- No additional labeling required

### For ML Classifier
- Quality labels for training data:
  - `good_quality`: 1
  - `poor_quality`: 0
- Estimated: 300-500 labeled samples needed

---

## Technical Specification

### Input
- Cropped container door image (from Module 1)
- Image dimensions
- Detection confidence score

### Output
```json
{
  "quality_score": 0.85,
  "quality_gate": "pass",
  "metrics": {
    "blur_score": 0.90,
    "brightness_score": 0.85,
    "contrast_score": 0.80,
    "size_score": 1.0
  },
  "recommendation": "proceed_to_localization"
}
```

### Decision Logic
```
IF quality_score >= threshold (e.g., 0.7):
    quality_gate = "pass"
ELSE:
    quality_gate = "fail"
```

---

## Implementation Plan

### Phase 1: Analysis (Week 1)
- [ ] Analyze current dataset
- [ ] Compute blur scores for all images
- [ ] Compute brightness/contrast distributions
- [ ] Determine appropriate thresholds

### Phase 2: Implementation (Week 2)
- [ ] Implement quality metric functions
- [ ] Create quality assessment module
- [ ] Integrate with pipeline
- [ ] Unit tests

### Phase 3: Validation (Week 3)
- [ ] Validate on test set
- [ ] Measure impact on downstream OCR
- [ ] Fine-tune thresholds
- [ ] Documentation

---

## Success Criteria

1. **Precision**: > 90% of "pass" images should be OCR-successful
2. **Recall**: > 80% of OCR-successful images should pass quality gate
3. **Speed**: < 10ms per image
4. **False Negative Rate**: < 10% (rejecting good images)

---

## References

- [Blur Detection Methods](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
- [Image Quality Assessment Survey](https://ieeexplore.ieee.org/document/9184846)

---

## Next Steps

1. Review and approve this specification
2. Allocate resources for implementation
3. Begin Phase 1 analysis

---

**Document Owner**: TBD  
**Review Date**: TBD

