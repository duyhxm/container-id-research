# Module 3 Data Preparation - Action Plan

**Document Version**: 1.0  
**Created**: 2025-12-23  
**Status**: 🚨 CRITICAL - Action Required  
**Owner**: Data Pipeline Team

---

## 📋 Executive Summary

Sau quá trình audit toàn diện, phát hiện **quy trình chuẩn bị dữ liệu hiện tại cho Module 3 (Container ID Localization) bị SAI LOGIC nghiêm trọng**, dẫn đến:

- ❌ Dataset training sử dụng **ảnh toàn cảnh** thay vì **ảnh đã crop vùng cửa xe**
- ❌ Tọa độ keypoint **không được transform** từ hệ tọa độ ảnh gốc sang ảnh crop
- ❌ Normalization sử dụng **kích thước ảnh sai** (original thay vì cropped)
- ❌ DVC pipeline gọi **script không phù hợp** cho Module 3

**Hậu quả nếu không sửa**:
- Model YOLOv11-Pose sẽ học trên wrong input format
- Inference sẽ fail hoàn toàn (train trên ảnh gốc, predict trên ảnh crop)
- Coordinate predictions sẽ không align với ảnh

**Action Required**: Dừng training Module 3 ngay lập tức cho đến khi fix xong.

---

## 🔍 Root Cause Analysis

### Vấn đề Chính: Architecture Mismatch

**Yêu cầu thực tế (Module 3 Specification)**:
```
Original Image → Detect Door (Module 1) → Crop to Door Region → Localize ID Keypoints (Module 3)
```

**Hiện trạng (Current Pipeline)**:
```
Original Image → Convert to YOLO Format → Train YOLOv11-Pose (on FULL images ❌)
```

**Nguyên nhân gốc rễ**:
1. Script `src/data/coco_to_yolo.py` được thiết kế cho **Module 1 (Detection)** - chỉ xử lý bbox, không crop ảnh
2. DVC pipeline tái sử dụng script này cho Module 3 với `--task pose` nhưng không nhận ra nó thiếu cropping logic
3. Không có validation để phát hiện mismatch giữa ảnh training và ảnh inference

---

## 🚨 Critical Issues Inventory

### Issue #1: Missing Image Cropping Pipeline
**Severity**: 🔴 CRITICAL (BLOCKING)  
**Affected Files**: 
- `src/data/coco_to_yolo.py` (line 221)
- `dvc.yaml` (stage `convert_localization`)

**Problem**:
```python
# Current code (WRONG)
shutil.copy2(src_img_path, dst_img_path)  # ❌ Copy full image
```

**Expected**:
```python
# Crop image to door region
from PIL import Image
img = Image.open(src_img_path)
x1, y1, x2, y2 = door_bbox
cropped_img = img.crop((x1, y1, x2, y2))
cropped_img.save(dst_img_path)
```

**Impact**: Model trains on wrong input format → 100% inference failure

---

### Issue #2: Incorrect Coordinate Transformation
**Severity**: 🔴 CRITICAL (BLOCKING)  
**Affected Files**: `src/data/coco_to_yolo.py` (lines 140-150)

**Problem**:
```python
# Current: Normalize using ORIGINAL image size ❌
normalized_points = [
    (x / orig_img_width, y / orig_img_height, v)
    for x, y, v in transformed_points
]
```

**Expected**:
```python
# Normalize using CROPPED image size ✅
crop_width = door_bbox[2] - door_bbox[0]
crop_height = door_bbox[3] - door_bbox[1]

normalized_points = [
    (x / crop_width, y / crop_height, v)
    for x, y, v in transformed_points
]
```

**Impact**: Keypoint coordinates won't align with cropped image boundaries

---

### Issue #3: Single-Category Processing Limitation
**Severity**: 🔴 CRITICAL (ARCHITECTURE)  
**Affected Files**: `src/data/coco_to_yolo.py` (entire file)

**Problem**:
- Script processes ONE category at a time (`--category-id` flag)
- Module 3 requires BOTH categories:
  - `container_door` (category_id=1) → for cropping bbox
  - `container_id` (category_id=2) → for keypoint annotations

**Current Workflow** (IMPOSSIBLE):
```bash
# Cannot access door bbox when processing ID keypoints
python src/data/coco_to_yolo.py --task pose --category-id 2
```

**Required Workflow**:
```python
# Script must process BOTH categories simultaneously
for image in images:
    door_bbox = get_annotation(image, category_id=1)
    id_keypoints = get_annotation(image, category_id=2)
    
    if door_bbox and id_keypoints:
        crop_and_transform(image, door_bbox, id_keypoints)
```

**Impact**: Fundamentally cannot implement correct logic with current script

---

### Issue #4: DVC Pipeline Misconfiguration
**Severity**: 🟠 HIGH (ORCHESTRATION)  
**Affected Files**: `dvc.yaml` (lines 40-55)

**Problem**:
```yaml
convert_localization:
  cmd: python src/data/coco_to_yolo.py --task pose ...  # ❌ Wrong script
  deps:
    # ❌ Missing: data/raw (original images for cropping)
    - data/interim/test_master.json
    - src/data/coco_to_yolo.py
```

**Expected**:
```yaml
convert_localization:
  cmd: python scripts/data_processing/prepare_module_3_data.py ...  # ✅ Specialized script
  deps:
    - data/interim/*.json
    - data/raw  # ✅ ADD: Need original images
    - scripts/data_processing/prepare_module_3_data.py
  params:
    - localization.door_category_id: 1  # ✅ ADD: Need door bbox
```

**Impact**: Pipeline won't reproduce correctly; missing dependencies

---

### Issue #5: Missing Data Filtering Logic Validation
**Severity**: 🟡 MEDIUM (DATA QUALITY)  
**Affected Files**: `src/data/coco_to_yolo.py` (lines 144-156)

**Problem**:
```python
# Current: Assumes field exists
ocr_feasibility = image.get("ocr_feasibility", "readable")  # ❌ Silent default
if ocr_feasibility == "unreadable":
    return True
```

**Finding from EDA**:
- `ocr_feasibility` distribution: readable (97%), unknown (2.2%), unreadable (0.8%)
- Current logic filters ONLY `unreadable`, but keeps `unknown` in training

**Question**: Should `unknown` samples be filtered from training?

**Recommendation**:
```python
# Be explicit about filtering rules
FILTER_VALUES = {"unreadable", "unknown"}  # Or just {"unreadable"}

ocr_feasibility = image.get("ocr_feasibility")
if ocr_feasibility is None:
    logger.warning(f"Missing ocr_feasibility for image {image['id']}")
    return False  # Don't filter if unknown

if self.task == "pose" and split == "train":
    return ocr_feasibility in FILTER_VALUES
```

**Impact**: Data quality inconsistency in training set

---

### Issue #6: No Validation for Cropped Image Dimensions
**Severity**: 🟡 MEDIUM (ROBUSTNESS)  
**Context**: YOLO requires minimum image size (typically ≥32x32)

**Problem**:
- No check for very small crops (e.g., distant containers)
- Could cause training instability or crashes

**Recommendation**:
```python
MIN_CROP_SIZE = 32

if crop_width < MIN_CROP_SIZE or crop_height < MIN_CROP_SIZE:
    logger.warning(
        f"Skipping {image_id}: crop too small ({crop_width}x{crop_height})"
    )
    continue
```

---

## 📝 Action Items

### 🔴 Phase 1: CRITICAL FIXES (Must Complete Before Training)

#### Task 1.1: Create Specialized Module 3 Preprocessing Script
**Priority**: P0 (BLOCKING)  
**Estimated Time**: 2-3 hours  
**Assignee**: Data Pipeline Lead

**Deliverable**: `scripts/data_processing/prepare_module_3_data.py`

**Requirements**:
1. Load BOTH `container_door` and `container_id` annotations
2. Match annotations by `image_id`
3. For each matched pair:
   - Load original image
   - Crop to door bbox
   - Transform keypoints: `x_new = x_old - x_door_min`
   - Normalize using crop dimensions: `x_norm = x_new / crop_width`
   - Save cropped image to `data/processed/localization/images/{split}/`
   - Save YOLO Pose label to `data/processed/localization/labels/{split}/`
4. Filter training data: exclude `ocr_feasibility ∈ {unreadable, unknown}`
5. Add validation: min crop size, keypoint bounds checking

**File Location**: `scripts/data_processing/` (per project structure rules)

**Template Structure**:
```python
"""
Module 3 Data Preparation Script

Converts COCO annotations to YOLO Pose format with image cropping for Container ID Localization.

Key Requirements:
- Crops images to container_door bbox
- Transforms keypoints from original frame to cropped frame
- Filters training data based on ocr_feasibility

Usage:
    python scripts/data_processing/prepare_module_3_data.py \
        --input data/interim \
        --output data/processed/localization \
        --images-dir data/raw \
        --config data/data_config.yaml
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Module3DataPreparator:
    """Prepare cropped dataset for Module 3 (Container ID Localization)."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.door_category_id = config["localization"]["door_category_id"]
        self.id_category_id = config["localization"]["category_id"]
        self.num_keypoints = config["localization"]["num_keypoints"]
        
    def crop_and_transform(
        self, 
        image_path: Path, 
        door_bbox: List[float], 
        keypoints: List[Tuple[float, float, int]]
    ) -> Tuple[Image.Image, List[float]]:
        """Crop image and transform keypoints to cropped coordinate system."""
        # TODO: Implement
        pass
    
    def process_split(self, split_name: str, master_json: Path, output_dir: Path):
        """Process one data split (train/val/test)."""
        # TODO: Implement
        pass
    
    def run(self, input_dir: Path, output_dir: Path, images_dir: Path):
        """Run full preprocessing pipeline."""
        # TODO: Implement
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="data/data_config.yaml")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Run preprocessing
    preparator = Module3DataPreparator(config)
    preparator.run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        images_dir=Path(args.images_dir)
    )


if __name__ == "__main__":
    main()
```

---

#### Task 1.2: Update DVC Pipeline
**Priority**: P0 (BLOCKING)  
**Estimated Time**: 30 minutes  
**Assignee**: Data Pipeline Lead

**File**: `dvc.yaml`

**Changes Required**:
```yaml
# REPLACE lines 40-55 with:
convert_localization:
  cmd: python scripts/data_processing/prepare_module_3_data.py 
    --input data/interim 
    --output data/processed/localization 
    --images-dir data/raw
    --config data/data_config.yaml
  deps:
    - data/interim/test_master.json
    - data/interim/train_master.json
    - data/interim/val_master.json
    - data/raw  # NEW: Need original images for cropping
    - scripts/data_processing/prepare_module_3_data.py
  params:
    - data/data_config.yaml:
      - localization.category_id
      - localization.category_name
      - localization.num_keypoints
      - localization.door_category_id  # NEW
  outs:
    - data/processed/localization
```

---

#### Task 1.3: Update Data Config
**Priority**: P0 (BLOCKING)  
**Estimated Time**: 5 minutes  
**Assignee**: Data Pipeline Lead

**File**: `data/data_config.yaml`

**Changes Required**:
```yaml
# ADD to localization section:
localization:
  category_id: 2
  category_name: container_id
  num_keypoints: 4
  door_category_id: 1  # NEW: For cropping bbox
  min_crop_size: 32    # NEW: Validation threshold
```

---

#### Task 1.4: Clean Old Dataset & Regenerate
**Priority**: P0 (BLOCKING)  
**Estimated Time**: 10 minutes (+ regeneration time)  
**Assignee**: Data Pipeline Lead

**Commands**:
```bash
# 1. Remove old (incorrect) dataset
dvc remove convert_localization
rm -rf data/processed/localization

# 2. Commit DVC changes
git add dvc.yaml dvc.lock
git commit -m "fix(dvc): update Module 3 pipeline to use specialized script"

# 3. Regenerate dataset with corrected pipeline
dvc repro convert_localization

# 4. Verify output
python scripts/validation/verify_module_3_dataset.py
```

---

### 🟡 Phase 2: VALIDATION & QUALITY ASSURANCE

#### Task 2.1: Create Dataset Validation Script
**Priority**: P1 (HIGH)  
**Estimated Time**: 1 hour  
**Assignee**: QA Lead

**Deliverable**: `scripts/validation/verify_module_3_dataset.py`

**Validation Checks**:
1. ✅ All images are cropped (not original size)
2. ✅ Keypoint coordinates are within [0, 1] range
3. ✅ Label format is correct: `0 cx cy w h p1x p1y v1 p2x p2y v2 ...`
4. ✅ Training split excludes `ocr_feasibility ∈ {unreadable, unknown}`
5. ✅ Test/Val splits include all samples
6. ✅ Number of images matches expected split ratios

**File Location**: `scripts/validation/`

---

#### Task 2.2: Add Unit Tests
**Priority**: P2 (MEDIUM)  
**Estimated Time**: 2 hours  
**Assignee**: Testing Team

**Deliverable**: `tests/data/test_module_3_preparation.py`

**Test Cases**:
```python
def test_coordinate_transformation():
    """Test keypoint transform from original to cropped frame."""
    pass

def test_normalization():
    """Test normalization uses crop dimensions, not original."""
    pass

def test_crop_boundaries():
    """Test cropped image has correct size."""
    pass

def test_filtering_logic():
    """Test ocr_feasibility filtering for train split."""
    pass
```

---

### 🔵 Phase 3: DOCUMENTATION & PROCESS IMPROVEMENT

#### Task 3.1: Update Module 3 Documentation
**Priority**: P2 (MEDIUM)  
**Estimated Time**: 1 hour  
**Assignee**: Technical Writer

**Files to Update**:
- `docs/modules/module-3-localization/README.md` → Add preprocessing details
- `docs/data-splitting/technical-specification-data-splitting.md` → Update adapter spec

---

#### Task 3.2: Add Pre-Training Checklist
**Priority**: P2 (MEDIUM)  
**Estimated Time**: 30 minutes  
**Assignee**: MLOps Lead

**Deliverable**: `docs/modules/module-3-localization/pre-training-checklist.md`

**Checklist Items**:
- [ ] Dataset generated using `prepare_module_3_data.py`
- [ ] Validation script passed all checks
- [ ] Sample images manually inspected (cropped correctly)
- [ ] Label format verified (YOLO Pose)
- [ ] Training split excludes unreadable samples
- [ ] DVC pipeline reproducible

---

## 📁 File Organization & Placement

### Current File Structure (BEFORE)
```
src/data/
├── coco_to_yolo.py          # Generic converter (Module 1 only)
└── stratification.py

scripts/
└── (empty)
```

### Proposed File Structure (AFTER)
```
src/data/
├── coco_to_yolo.py          # Keep for Module 1 (Detection)
├── stratification.py
└── augmentation.py

scripts/
├── data_processing/         # NEW: Module-specific preprocessing
│   ├── __init__.py
│   ├── prepare_module_1_data.py     # Future: Refactor if needed
│   ├── prepare_module_3_data.py     # NEW: Module 3 specialized script
│   └── README.md                    # Document each script's purpose
│
└── validation/              # NEW: Dataset validation utilities
    ├── __init__.py
    ├── verify_module_3_dataset.py   # NEW: Validation script
    └── check_coco_annotations.py    # Future: General validation

tests/
└── data/
    ├── test_coco_to_yolo.py         # Existing
    └── test_module_3_preparation.py # NEW: Unit tests
```

### Rationale for File Placement

**Why `scripts/data_processing/` for Module 3 script?**
1. **Per Project Structure Rules** (`project_structure.instructions.md`):
   - `src/` is for reusable library code (no execution logic)
   - `scripts/` is for standalone executables
   - Module-specific preprocessing → `scripts/`

2. **Separation of Concerns**:
   - `src/data/coco_to_yolo.py` → Generic, reusable converter
   - `scripts/data_processing/prepare_module_3_data.py` → Specialized, module-specific pipeline

3. **DVC Integration**:
   - DVC stages call `scripts/` (execution entry points)
   - `src/` is imported by scripts (library functions)

4. **Future Scalability**:
   - Each module can have its own preprocessing script
   - Clear separation between generic utilities and module adapters

**Why `scripts/validation/` for verification?**
- Not part of main data pipeline (DVC stages)
- Run manually or in CI/CD for quality checks
- Standalone utilities

---

## 🛠️ Implementation Roadmap

### Week 1: Critical Fixes
| Day | Task          | Owner     | Deliverable                        |
| --- | ------------- | --------- | ---------------------------------- |
| Mon | Task 1.1      | Data Lead | `prepare_module_3_data.py` (draft) |
| Tue | Task 1.1      | Data Lead | Script testing & debugging         |
| Wed | Task 1.2, 1.3 | Data Lead | DVC pipeline updated               |
| Thu | Task 1.4      | Data Lead | Dataset regenerated                |
| Fri | Task 2.1      | QA Lead   | Validation passed ✅                |

### Week 2: Quality Assurance
| Day | Task              | Owner       | Deliverable            |
| --- | ----------------- | ----------- | ---------------------- |
| Mon | Task 2.2          | Test Team   | Unit tests             |
| Tue | Manual inspection | ML Team     | Sample review          |
| Wed | Task 3.1          | Tech Writer | Docs updated           |
| Thu | Task 3.2          | MLOps       | Pre-training checklist |
| Fri | Code review       | Team        | Merge to main          |

---

## ✅ Definition of Done

**Phase 1 (Critical) is complete when**:
- ✅ `scripts/data_processing/prepare_module_3_data.py` implemented and tested
- ✅ DVC pipeline updated and reproducible
- ✅ Dataset regenerated with cropped images
- ✅ Validation script confirms:
  - All images are cropped (not 1920x1080)
  - Keypoints normalized to crop dimensions
  - Training data excludes unreadable samples
- ✅ Manual inspection of 20 random samples passed

**Phase 2 (Validation) is complete when**:
- ✅ Unit tests pass (>90% coverage on new code)
- ✅ CI/CD pipeline green
- ✅ Code review approved by 2+ reviewers

**Phase 3 (Documentation) is complete when**:
- ✅ All docs updated
- ✅ Pre-training checklist created
- ✅ Team training completed

---

## 🚫 What NOT to Do

❌ **Do NOT modify `src/data/coco_to_yolo.py` for Module 3**
- This script is correct for Module 1 (Detection)
- Attempting to make it handle both tasks will create spaghetti code
- Follow separation of concerns: one script per module

❌ **Do NOT start Module 3 training until Phase 1 is complete**
- Current dataset is fundamentally wrong
- Training will waste time and compute
- Results will be invalid

❌ **Do NOT skip validation (Task 2.1)**
- Silent failures are expensive
- Validation prevents regression

---

## 📊 Risk Assessment

| Risk                                  | Likelihood | Impact | Mitigation                                    |
| ------------------------------------- | ---------- | ------ | --------------------------------------------- |
| Script bugs during implementation     | High       | High   | Extensive unit testing + manual validation    |
| Dataset size mismatch after filtering | Medium     | Medium | EDA shows only 2.2% unknown + 0.8% unreadable |
| DVC reproducibility issues            | Low        | High   | Test on clean clone                           |
| Cropped images too small for YOLO     | Low        | Medium | Min size validation (32x32)                   |

---

## 📞 Contact & Escalation

**Questions about**:
- Data pipeline logic → Data Pipeline Lead
- DVC configuration → MLOps Lead
- YOLO format → ML Engineer
- This action plan → QA/Security Lead (Document Author)

**Escalation Path**:
1. Daily standup (flag blockers)
2. Project Manager (if >1 day delay)
3. Technical Lead (if architectural changes needed)

---

## 📚 References

**Related Documents**:
- [Project Structure Rules](../../.github/instructions/project_structure.instructions.md)
- [DVC Workflow](../../.github/instructions/dvc_workflow.instructions.md) (if exists)
- [Module 3 Technical Spec](./technical-specification.md) (if exists)
- [Data Splitting Methodology](../../data-splitting/data-splitting-methodology.md)

**Code Review Reports** (Generated 2025-12-23):
1. "Module 3 Dataset & Preprocessing Logic Audit"
2. "DVC Pipeline Integration for Module 1 & 3"
3. "Code Review: `coco_to_yolo.py`"

---

**Document Status**: 🟢 APPROVED  
**Next Review Date**: After Phase 1 completion  
**Change Log**:
- 2025-12-23: Initial version (v1.0)
