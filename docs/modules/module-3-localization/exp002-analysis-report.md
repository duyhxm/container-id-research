# Module 3 Localization - Analysis Report (exp002)

**Date**: December 25, 2025  
**Analyst**: AI/ML Engineer  
**Model**: YOLOv11s-Pose (exp002_improved)

---

## 🎯 Executive Summary

**The "9 missed detections" issue was a configuration bug, NOT a model failure.**

The model **successfully detects all 74 validation images** when using the correct confidence threshold (`conf=0.25`). The notebook was incorrectly using `conf=0.6`, which is too high and filtered out 9 valid detections with confidence ~0.57-0.58.

**Model Performance**:
- ✅ Mean Distance Error (MDE): **9.95 ± 4.32 px** (Target: <10px)
- ✅ Polygon IoU: **0.832 ± 0.057** (Target: >0.75)
- ✅ Detection Rate: **100%** at `conf=0.25` (YOLO validation default)
- ⚠️ Keypoint Imbalance: Top-Left (KP0) has 1.92x higher error than Top-Right (KP1)

---

## 🔴 Root Cause Analysis

### Issue 1: Confidence Threshold Mismatch ✅ SOLVED

**Problem**: Notebook used `conf=0.6` during inference, while YOLO validation uses `conf=0.25` by default.

**Evidence**:
```
| Image        | conf=0.6 | conf=0.5   | conf=0.25  | Actual Confidence |
| ------------ | -------- | ---------- | ---------- | ----------------- |
| 0000384.jpg  | ❌ MISSED | ✅ DETECTED | ✅ DETECTED | 0.5783            |
| 0002989.jpg  | ❌ MISSED | ✅ DETECTED | ✅ DETECTED | 0.5747            |
| 0012124.jpg  | ❌ MISSED | ✅ DETECTED | ✅ DETECTED | 0.5739            |
| ... (6 more) | ❌ MISSED | ✅ DETECTED | ✅ DETECTED | ~0.57             |
```

**Characteristics of "missed" images**:
- All have **small bounding boxes** (0.94% - 2.73% of image area)
- All have **high aspect ratio** (5.5 - 7.4:1, very wide and thin)
- These are **valid Container ID regions**, not false positives
- Model correctly localized them, but confidence was below 0.6 threshold

**Fix**: Changed notebook inference from `conf=0.6` to `conf=0.25` to match validation settings.

---

### Issue 2: Keypoint Imbalance (Top-Left Corner) ⚠️ IN PROGRESS

**Problem**: Keypoint 0 (Top-Left) has significantly higher error than other keypoints.

**Per-Keypoint Error Statistics**:
```
| Keypoint         | Mean Error | Std Dev | Status            |
| ---------------- | ---------- | ------- | ----------------- |
| KP0 Top-Left     | 13.34 px   | ±3.99   | ⚠️ WORST (1.92x)   |
| KP1 Top-Right    | 6.96 px    | ±6.60   | ✅ BEST (high var) |
| KP2 Bottom-Right | 9.10 px    | ±4.67   | ✅ GOOD            |
| KP3 Bottom-Left  | 10.41 px   | ±6.09   | ⚠️ HIGH VARIANCE   |
```

**Possible Causes**:
1. **Labeling inconsistency**: Top-Left corner may be harder to annotate consistently
2. **Occlusion bias**: Top-Left may be occluded more often in training data
3. **Model architecture bias**: YOLO may prioritize bottom corners (closer to detection box)
4. **Insufficient pose weight**: Model may prioritize bounding box over keypoints

**Recommended Actions**:
- ✅ **Created exp003**: Increase pose weight from 15.0 to **20.0**
- 📋 **TODO**: Run [05_localization_label_quality_check.ipynb](../notebooks/05_localization_label_quality_check.ipynb) to verify Top-Left annotation quality
- 🔬 **TODO**: Visualize Top-Left errors on training set to identify patterns

---

## 📊 Comparison: exp001 vs exp002

| Metric                      | exp001 (Baseline) | exp002 (Improved) |     Change |   Status   |
| :-------------------------- | ----------------: | ----------------: | ---------: | :--------: |
| **MDE**                     |   14.92 ± 8.83 px |    9.95 ± 4.32 px | **-33%** ✅ | Target Met |
| **Polygon IoU**             |     0.647 ± 0.106 |     0.832 ± 0.057 | **+29%** ✅ | Target Met |
| **Detection Rate**          |              ~85% |             100%* | **+18%** ✅ | Excellent  |
| **KP0 Error (Top-Left)**    |            ~23 px |          13.34 px | **-42%** ⚠️ | Still High |
| **KP3 Error (Bottom-Left)** |          23.50 px |          10.41 px | **-56%** ✅ | Excellent  |

*At `conf=0.25` (YOLO validation default)

**Key Insights**:
- ✅ **Overall improvement**: exp002 successfully reduced MDE and improved IoU
- ✅ **Bottom-Left fixed**: KP3 error reduced from 23.50px to 10.41px (exp002 goal achieved)
- ⚠️ **Top-Left still problematic**: KP0 error at 13.34px (need further improvement)
- ✅ **More consistent**: Standard deviation reduced by 51% (8.83 → 4.32)

---

## 🛠️ Recommendations for exp003

### Priority 1: Fix Top-Left Keypoint Imbalance 🔴 HIGH

**Action**: Increase pose loss weight from 15.0 to **20.0** (33% increase)

**Justification**:
- Current imbalance ratio: 1.92x (Top-Left vs Top-Right)
- Target: <1.5x imbalance ratio
- Aggressive pose weight will force model to prioritize keypoint accuracy over bounding box

**Expected Result**: KP0 error from 13.34px → **<10px**

---

### Priority 2: Reduce Prediction Variance 🟡 MEDIUM

**Problem**: KP1 and KP3 have high standard deviation (6-7px) → inconsistent predictions

**Actions**:
1. **Increase batch size**: 64 → **96**
   - Larger batches = more stable gradients = lower variance
   - Still fits on Kaggle T4 x2 with AMP enabled
2. **Add cosine LR with warm restarts**:
   - Help escape training plateaus (observed around epoch 70-80)
   - `restart_epochs: 50` (restart every 50 epochs)

**Expected Result**: Std deviation from 6-7px → **<5px**

---

### Priority 3: Improve Geometric Robustness 🟢 LOW

**Action**: Increase perspective augmentation from 0.0005 to **0.001**

**Justification**:
- Container IDs have natural perspective distortion (camera angle)
- Current augmentation may be too conservative
- Higher perspective variance will improve real-world performance

---

### Priority 4: Inspect Labeling Quality 🔵 ANALYSIS

**Action**: Run [05_localization_label_quality_check.ipynb](../notebooks/05_localization_label_quality_check.ipynb)

**Focus Areas**:
- Check Top-Left (KP0) annotation consistency
- Verify no systematic labeling errors
- Cross-reference with worst-case predictions (MDE >20px)

---

## 🎯 Expected Results for exp003

| Metric                    |   exp002 | exp003 (Target) | Improvement |
| :------------------------ | -------: | --------------: | :---------: |
| **MDE**                   |  9.95 px |     **<8.0 px** |    -20%     |
| **Polygon IoU**           |    0.832 |       **>0.85** |     +2%     |
| **KP0 Error (Top-Left)**  | 13.34 px |    **<10.0 px** |    -25%     |
| **KP1 Std Dev**           |  6.60 px |     **<5.5 px** |    -17%     |
| **KP3 Std Dev**           |  6.09 px |     **<5.0 px** |    -18%     |
| **Detection Rate @ 0.25** |     100% |        **100%** |  Maintain   |

---

## 📋 Action Items

### Immediate (Today)
- [x] Fix notebook confidence threshold (`conf=0.6` → `conf=0.25`)
- [x] Create `experiments/003_loc_higher_pose_weight.yaml`
- [x] Document findings in this report

### Short-Term (Next Training Run)
- [ ] Train exp003 on Kaggle with new config
- [ ] Re-run [06_localization_prediction_analysis.ipynb](../notebooks/06_localization_prediction_analysis.ipynb) on exp003 results
- [ ] Compare exp002 vs exp003 metrics

### Medium-Term (After exp003)
- [ ] Run [05_localization_label_quality_check.ipynb](../notebooks/05_localization_label_quality_check.ipynb)
- [ ] Investigate Top-Left keypoint labeling consistency
- [ ] If imbalance persists, consider upgrading to **YOLOv11m** (medium)

---

## 📎 References

- **Experiment Configs**:
  - [experiments/001_loc_baseline.yaml](../experiments/001_loc_baseline.yaml)
  - [experiments/002_loc_improved.yaml](../experiments/002_loc_improved.yaml)
  - [experiments/003_loc_higher_pose_weight.yaml](../experiments/003_loc_higher_pose_weight.yaml)

- **Analysis Notebooks**:
  - [notebooks/05_localization_label_quality_check.ipynb](../notebooks/05_localization_label_quality_check.ipynb)
  - [notebooks/06_localization_prediction_analysis.ipynb](../notebooks/06_localization_prediction_analysis.ipynb)

- **Training Artifacts**:
  - [artifacts/localization/localization_exp001_yolo11s_pose_baseline/](../artifacts/localization/localization_exp001_yolo11s_pose_baseline/)
  - [artifacts/localization/localization_exp002_yolo11s_pose_improved/](../artifacts/localization/localization_exp002_yolo11s_pose_improved/)

---

**Report Generated**: December 25, 2025  
**Next Review**: After exp003 training completion
