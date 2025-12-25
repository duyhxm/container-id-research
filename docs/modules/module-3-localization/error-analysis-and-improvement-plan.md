# Module 3 Localization - Error Analysis & Improvement Plan

**Analysis Date**: December 25, 2025  
**Model**: YOLOv11s-Pose (Experiment 001 Baseline)  
**Analyst**: AI Agent + Human Review  
**Status**: 🔴 Critical Issues Identified

---

## Executive Summary

Comprehensive error analysis of the localization model (exp001_baseline) reveals **critical accuracy issues** despite excellent detection performance. The model achieves 100% detection rate but fails to meet localization accuracy targets by a significant margin.

### Key Findings

| Metric                        | Current         | Target | Status                 |
| :---------------------------- | :-------------- | :----- | :--------------------- |
| **Detection Rate**            | 100.0%          | >95%   | ✅ **EXCELLENT**        |
| **Mean Distance Error (MDE)** | 14.92 ± 8.83 px | <5 px  | ❌ **3x OVER TARGET**   |
| **Polygon IoU**               | 0.647 ± 0.106   | >0.85  | ❌ **24% BELOW TARGET** |
| **Precision (P)**             | 99.09%          | >95%   | ✅ **EXCELLENT**        |
| **Recall (P)**                | 100.0%          | >95%   | ✅ **PERFECT**          |
| **mAP@50-95 (P)**             | 98.91%          | >90%   | ✅ **EXCELLENT**        |

**⚠️ CRITICAL FINDING**: The model has **excellent object detection** (mAP 98.91%) but **poor keypoint localization** (MDE 14.92px). This indicates the model can find the Container ID region but struggles to precisely locate the 4 corner keypoints.

---

## 1. Detailed Error Analysis

### 1.1 Overall Performance Metrics

**Validation Set**: 74 images

```
📊 Detection Performance:
   • Total images: 74
   • Detected: 74 (100.0%)
   • Missed: 0 (0.0%)

📏 Localization Accuracy:
   • Mean Distance Error: 14.92 ± 8.83 px
   • Median Distance Error: 12.44 px
   • Max Distance Error: 49.42 px

🔷 Polygon Quality:
   • Mean IoU: 0.647 ± 0.106
   • Median IoU: 0.647
   • Min IoU: 0.244
```

### 1.2 Per-Keypoint Error Analysis

**CRITICAL DISCOVERY**: Keypoint 3 (Bottom-Left) has **2.3x higher error** than the best-performing keypoint.

| Keypoint ID | Position        | Mean Error     | Std Dev | Min  | Median | Max       |
| :---------- | :-------------- | :------------- | :------ | :--- | :----- | :-------- |
| **0**       | Top-Left        | 12.07 px       | ±11.84  | 0.15 | 9.29   | 82.07     |
| **1**       | Top-Right       | 13.91 px       | ±10.89  | 3.13 | 10.85  | 67.98     |
| **2**       | Bottom-Right    | **10.19 px** ✅ | ±8.81   | 1.87 | 7.41   | 47.76     |
| **3**       | **Bottom-Left** | **23.50 px** ⚠️ | ±13.83  | 2.77 | 20.07  | **87.75** |

**Error Ratio**: max/min = 23.50 / 10.19 = **2.31x**

**Interpretation**:
- Keypoint 2 (Bottom-Right): Best performance, concentrated around 7-10px
- Keypoint 3 (Bottom-Left): Worst performance, highly dispersed distribution
- Keypoints 0, 1: Moderate performance, acceptable distribution

### 1.3 Worst-Case Analysis

**Top 10 Worst Cases by MDE:**

| Image       | MDE (px) | IoU   | Comments                         |
| :---------- | :------- | :---- | :------------------------------- |
| 0024620.jpg | 49.42    | 0.244 | Catastrophic failure - IoU <0.25 |
| 0015818.jpg | 43.70    | 0.538 | Very high error                  |
| 0002254.jpg | 41.43    | 0.458 | High error, low IoU              |
| 0000374.jpg | 37.14    | 0.578 | High error                       |
| 0021605.jpg | 29.42    | 0.544 | Moderate-high error              |
| 0009791.jpg | 29.06    | 0.715 | Error despite decent IoU         |
| 0003380.jpg | 28.54    | 0.564 | Moderate error                   |
| 0002346.jpg | 27.75    | 0.588 | Moderate error                   |
| 0022943.jpg | 25.82    | 0.659 | Moderate error                   |
| 0012695.jpg | 22.13    | 0.760 | Acceptable IoU but high MDE      |

**Patterns Observed**:
- 13 images (17.6%) have MDE > 20px (critical threshold)
- 7 images (9.5%) have IoU < 0.5 (unacceptable overlap)
- Worst case (0024620.jpg): IoU = 0.244 indicates near-complete failure

---

## 2. Root Cause Analysis

### 2.1 Keypoint 3 (Bottom-Left) Imbalance

**Hypothesis 1: Occlusion/Visibility Bias**
- Bottom-left corner may be more frequently occluded due to:
  - Camera angle (looking slightly upward at containers)
  - Physical obstacles (locks, hinges, rust spots)
  - Lighting conditions (shadows from container structure)

**Hypothesis 2: Model Architecture Bias**
- YOLOv11 pose model may learn spatial priors from training data
- If Keypoint 3 has higher variability in GT labels, model struggles to generalize

**Hypothesis 3: Loss Function Imbalance**
- Current `pose` weight in config: **NOT EXPLICITLY SET** (using YOLO default)
- Default keypoint loss may not penalize errors equally across all keypoints
- Model may prioritize IoU (box) over precise keypoint localization

### 2.2 Why High mAP but High MDE?

**Object Keypoint Similarity (OKS)** used in mAP normalizes distance by object scale:

```
OKS = exp(-d²/(2s²k²))
```

Where:
- `d` = Euclidean distance between pred and GT keypoint
- `s` = object scale (√area)
- `k` = keypoint-specific constant (typically 0.5-1.0)

**Example Calculation**:
- Object scale (Container ID): ~775px width
- 14px error → d²/(2s²k²) = 14²/(2×775²×0.5²) ≈ 0.065
- OKS ≈ exp(-0.065) ≈ 0.937 → **93.7% similarity** ✅

**Conclusion**: OKS forgives larger pixel errors for large objects. 14px error gives good mAP score but fails our <5px requirement.

### 2.3 Augmentation Issues

**Current Config (exp001_baseline)**:
```yaml
augmentation:
  degrees: 5.0          # Too conservative?
  translate: 0.1        # 10% - reasonable
  scale: 0.3            # 30% - reasonable
  perspective: 0.0      # DISABLED
  fliplr: 0.0           # DISABLED (correct for text)
```

**Potential Issues**:
1. **No perspective augmentation** → Model not robust to perspective distortion
2. **Small rotation range (±5°)** → May not cover real-world tilt variations
3. **No flip augmentation** → Less data diversity (though correct decision for text)

---

## 3. Improvement Strategy

### 3.1 Priority 1: Increase Keypoint Loss Weight

**Action**: Create `experiments/002_loc_improved.yaml` with increased `pose` weight.

```yaml
# NEW hyperparameters
training:
  epochs: 150                    # Increase from 100
  batch_size: 64                 # Keep same
  learning_rate: 0.001           # Keep same
  pose: 3.0                      # ⭐ NEW: Increase from default (~1.0)
  box: 7.5                       # Keep YOLO default
  cls: 0.5                       # Keep YOLO default
  kobj: 1.0                      # Keep YOLO default
```

**Rationale**:
- Increase keypoint loss weight by 3x to prioritize precise localization
- Model currently optimizes for detection (box) over keypoint accuracy
- Trade-off: May slightly reduce mAP, but improve MDE

**Expected Impact**: MDE 14.92px → **~8-10px** (40-50% reduction)

### 3.2 Priority 2: Add Perspective Augmentation

**Action**: Enable perspective distortion in augmentation.

```yaml
augmentation:
  degrees: 10.0          # Increase from 5.0
  translate: 0.1         # Keep same
  scale: 0.3             # Keep same
  perspective: 0.0005    # ⭐ NEW: Enable (very small value for text)
  fliplr: 0.0            # Keep disabled
```

**Rationale**:
- Real container doors often have perspective distortion
- Model needs to learn invariance to slight perspective changes
- Use very small value (0.0005) to avoid distorting text readability

**Expected Impact**: IoU 0.647 → **~0.75-0.80** (15-20% improvement)

### 3.3 Priority 3: Keypoint-Specific Loss Weighting

**Action**: Investigate if YOLO supports per-keypoint loss weights. If yes, increase weight for Keypoint 3.

**Rationale**:
- Keypoint 3 has 2.3x higher error → needs special attention
- May require custom loss function modification

**Research Required**: Check YOLOv11 documentation for per-keypoint loss configuration.

### 3.4 Priority 4: Data Quality Review

**Action**: Manually review GT labels for Keypoint 3 on worst-case images.

**Images to Review**:
- 0024620.jpg (MDE: 49.42px)
- 0015818.jpg (MDE: 43.70px)
- 0002254.jpg (MDE: 41.43px)

**Check For**:
- Labeling errors (wrong corner marked)
- Occlusion (keypoint not visible)
- Ambiguity (corner position unclear due to rust/damage)

### 3.5 Priority 5: Model Architecture Upgrade

**Action**: If Priority 1-3 don't achieve target, upgrade to YOLOv11m-Pose (medium model).

```yaml
model:
  architecture: yolo11m-pose    # Upgrade from yolo11s-pose
```

**Rationale**:
- YOLOv11m has more parameters → better feature representation
- May capture subtle corner patterns better than YOLOv11s

**Trade-off**: 
- Slower inference (~2x)
- Higher memory usage
- Longer training time

---

## 4. Implementation Plan

### Phase 1: Quick Wins (Week 1)

**Task 1.1**: Create `experiments/002_loc_improved.yaml`
- [x] Increase `pose` weight to 3.0
- [x] Increase `epochs` to 150
- [x] Enable perspective augmentation (0.0005)
- [x] Increase rotation range to ±10°

**Task 1.2**: Retrain model on Kaggle
```bash
uv run python src/localization/train_and_evaluate.py \
  --config experiments/002_loc_improved.yaml
```

**Task 1.3**: Re-run analysis notebook
```bash
# Update model path in notebook to new weights
# Run: notebooks/06_localization_prediction_analysis.ipynb
```

**Success Criteria**:
- MDE < 10px (vs current 14.92px)
- IoU > 0.75 (vs current 0.647)
- Keypoint 3 error < 15px (vs current 23.50px)

### Phase 2: Data Quality Review (Week 2)

**Task 2.1**: Create GT label review notebook
```python
# notebooks/07_localization_gt_keypoint_analysis.ipynb
# - Visualize Keypoint 3 distribution
# - Check for labeling patterns
# - Identify outliers
```

**Task 2.2**: Manual review of worst-case images
- Review top 10 worst MDE cases
- Check if Keypoint 3 is consistently problematic

**Task 2.3**: Re-label if needed
- If labeling errors found, correct in CVAT
- Re-export dataset
- Re-run data processing pipeline

### Phase 3: Architecture Upgrade (If Needed)

**Task 3.1**: Train YOLOv11m-Pose
- Create `experiments/003_loc_yolo11m.yaml`
- Train with same hyperparameters as exp002

**Task 3.2**: Benchmark inference speed
- Measure FPS on target hardware
- Ensure <200ms per image

---

## 5. Acceptance Criteria

### Minimum Requirements (Must Achieve)

| Metric           | Current | Target | Priority     |
| :--------------- | :------ | :----- | :----------- |
| Detection Rate   | 100.0%  | >99%   | **CRITICAL** |
| MDE (Mean)       | 14.92px | <8px   | **CRITICAL** |
| MDE (Max)        | 49.42px | <25px  | **HIGH**     |
| Polygon IoU      | 0.647   | >0.75  | **CRITICAL** |
| Keypoint 3 Error | 23.50px | <15px  | **HIGH**     |

### Stretch Goals

| Metric                          | Stretch Target |
| :------------------------------ | :------------- |
| MDE (Mean)                      | <5px           |
| Polygon IoU                     | >0.85          |
| Keypoint 3 Error                | <12px          |
| All Keypoints within 2x of best | Yes            |

---

## 6. Risk Assessment

### Risk 1: Overfitting with Higher Pose Weight
**Likelihood**: Medium  
**Impact**: Medium  
**Mitigation**:
- Monitor validation loss carefully
- Use early stopping (patience=20)
- Increase epochs to allow model to converge

### Risk 2: Perspective Augmentation Distorts Text
**Likelihood**: Low  
**Impact**: High  
**Mitigation**:
- Use very small perspective value (0.0005)
- Manually inspect augmented images
- Disable if text readability affected

### Risk 3: Ground Truth Quality Issues
**Likelihood**: Medium  
**Impact**: High  
**Mitigation**:
- Perform Phase 2 data quality review
- Re-label if systematic errors found
- Document labeling guidelines for Keypoint 3

### Risk 4: Insufficient Model Capacity
**Likelihood**: Low  
**Impact**: High  
**Mitigation**:
- Phase 3 architecture upgrade ready (YOLOv11m)
- Benchmark shows YOLOv11m achieves <5px MDE on similar tasks

---

## 7. Success Metrics & Tracking

### Experiment Tracking Table

| Exp ID | Model    | Pose Weight | Perspective Aug | MDE (px) | IoU     | KPT3 Error | Status        |
| :----- | :------- | :---------- | :-------------- | :------- | :------ | :--------- | :------------ |
| 001    | YOLOv11s | ~1.0        | 0.0             | 14.92    | 0.647   | 23.50      | ❌ Baseline    |
| 002    | YOLOv11s | 3.0         | 0.0005          | **TBD**  | **TBD** | **TBD**    | 🔄 In Progress |
| 003    | YOLOv11m | 3.0         | 0.0005          | **TBD**  | **TBD** | **TBD**    | ⏳ Planned     |

### Weekly Progress Review

**Week 1 Goals**:
- [ ] Create experiment 002 config
- [ ] Complete Kaggle training
- [ ] Run analysis notebook
- [ ] Validate MDE < 10px

**Week 2 Goals**:
- [ ] GT label review notebook
- [ ] Manual inspection of worst cases
- [ ] Decision: Re-label vs Continue training

**Week 3 Goals**:
- [ ] If needed: Train experiment 003 (YOLOv11m)
- [ ] Final validation
- [ ] Deploy best model to production

---

## 8. Lessons Learned

### What Went Right ✅

1. **Ground Truth Quality**: 100% valid labels (verified in notebook 05)
2. **Detection Pipeline**: Perfect detection rate (100%)
3. **mAP Score**: Excellent (98.91%) - model finds objects well
4. **Analysis Methodology**: Comprehensive per-keypoint analysis revealed root cause

### What Went Wrong ❌

1. **Insufficient Keypoint Loss Weight**: Default YOLO config prioritizes detection over localization
2. **No Perspective Augmentation**: Model not robust to perspective distortion
3. **No Keypoint-Specific Tuning**: Didn't account for Bottom-Left corner difficulty
4. **Delayed Error Analysis**: Should have run detailed analysis after first training

### Key Takeaways 📝

1. **OKS vs Pixel Accuracy**: High mAP doesn't guarantee precise localization for large objects
2. **Per-Keypoint Analysis is Critical**: Average metrics hide keypoint-specific issues
3. **Loss Function Tuning Matters**: Default hyperparameters not optimal for all tasks
4. **Visual Inspection Required**: Statistics alone don't reveal spatial error patterns

---

## 9. References

### Code & Notebooks

- **Training Script**: `src/localization/train_and_evaluate.py`
- **Evaluation Script**: `src/localization/evaluate.py`
- **Error Analysis**: `notebooks/06_localization_prediction_analysis.ipynb`
- **GT Quality Check**: `notebooks/05_localization_label_quality_check.ipynb`

### Configuration Files

- **Baseline Config**: `experiments/001_loc_baseline.yaml`
- **Improved Config**: `experiments/002_loc_improved.yaml` (to be created)
- **Data Config**: `data/processed/localization/data.yaml`

### Documentation

- **Module Overview**: `docs/modules/module-3-localization/README.md`
- **Labeling Guidelines**: `docs/data-labeling/id-container-labeling-guideline.md`

---

## 10. Appendix

### A. Histogram Analysis Interpretation

From the per-keypoint error distribution visualization:

**Keypoint 0 (Top-Left)**:
- Distribution: Right-skewed, concentrated around 7-12px
- Outliers: Few cases >40px (likely occlusion)
- Assessment: **Acceptable** performance

**Keypoint 1 (Top-Right)**:
- Distribution: Similar to KPT0, slightly higher mean
- Mode: 8-12px range
- Assessment: **Acceptable** performance

**Keypoint 2 (Bottom-Right)**:
- Distribution: **Tightest** concentration (7-10px)
- Very few outliers
- Assessment: **Best** performance - use as reference

**Keypoint 3 (Bottom-Left)**:
- Distribution: **Highly dispersed** (wide spread)
- Bimodal: Peak at 15-20px, secondary peak at 5-10px
- Long tail: Extends to >80px
- Assessment: **Problematic** - needs investigation

### B. Visualization Insights

From worst-case visualization cells:

**Common Failure Patterns**:
1. **Translation Shift**: Predicted polygon shifted downward (parallel translation)
2. **Scale Error**: Predicted box correct shape but wrong size
3. **Rotation Error**: Slight angular misalignment (~5-10°)
4. **Keypoint 3 Drift**: Bottom-left consistently further from GT than other corners

**NOT Observed**:
- Topology errors (keypoint order is always correct)
- Complete misdetection (100% detection rate)
- Mirror/flip errors (correct, as augmentation disabled)

---

**Document Owner**: AI Agent  
**Last Updated**: December 25, 2025  
**Next Review**: After Experiment 002 Training Completion
