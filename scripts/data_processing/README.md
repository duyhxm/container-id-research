# Data Processing Scripts

This directory contains module-specific data preprocessing scripts for the Container ID Research project.

## Scripts

### `prepare_module_3_data.py`
**Purpose**: Prepare dataset for Module 3 (Container ID Localization) by cropping images and transforming keypoints.

**Key Operations**:
1. Loads BOTH `container_door` (category_id=1) and `container_id` (category_id=2) annotations
2. Crops images to door bounding box
3. Transforms keypoints from original coordinate system to cropped coordinate system
4. Normalizes keypoints using crop dimensions (not original image size)
5. Filters training data based on `ocr_feasibility` attribute
6. Validates crop sizes and keypoint bounds

**Usage**:
```bash
python scripts/data_processing/prepare_module_3_data.py \
    --input data/interim \
    --output data/processed/localization \
    --images-dir data/raw \
    --config data/data_config.yaml
```

**Input**:
- `data/interim/{train,val,test}_master.json` - Master annotation files
- `data/raw/` - Original images
- `data/data_config.yaml` - Configuration file

**Output**:
- `data/processed/localization/images/{train,val,test}/` - Cropped images
- `data/processed/localization/labels/{train,val,test}/` - YOLO Pose format labels
- `data/processed/localization/data.yaml` - YOLO dataset configuration

**Configuration Requirements** (`data/data_config.yaml`):
```yaml
localization:
  door_category_id: 1         # Category ID for container_door
  category_id: 2              # Category ID for container_id
  num_keypoints: 4            # Number of keypoints (4-point polygon)
  min_crop_size: 32           # Minimum crop size validation threshold
```

**Filtering Rules**:
- **Training split**: Excludes images with `ocr_feasibility ∈ {unreadable, unknown}`
- **Val/Test splits**: Includes ALL images (for robustness evaluation)

**Validation Checks**:
- Minimum crop size (default: 32×32 pixels)
- Bounding box within image bounds
- Keypoints within crop bounds (clamped if necessary)
- Final normalized coordinates in [0, 1] range

---

## Future Scripts

### `prepare_module_1_data.py` (Planned)
- Refactored version of `src/data/coco_to_yolo.py` for Module 1 (Detection)
- Will follow the same structure as Module 3 script

### `prepare_module_5_data.py` (Planned)
- OCR dataset preparation with text annotations
- Will handle aligned ROI images from Module 4

---

## Design Philosophy

**Why separate scripts for each module?**
1. **Separation of Concerns**: Each module has unique preprocessing requirements
2. **Maintainability**: Module-specific logic isolated from generic utilities
3. **DVC Integration**: Clear entry points for pipeline stages
4. **Scalability**: Easy to add new modules without modifying existing code

**Relationship with `src/data/`**:
- `src/data/` contains **reusable library code** (e.g., COCO parsers, augmentation)
- `scripts/data_processing/` contains **module-specific executables**
- Scripts MAY import utilities from `src/data/` when needed

---

## Testing

Unit tests for these scripts are located in:
- `tests/data/test_module_3_preparation.py`

Run tests with:
```bash
pytest tests/data/test_module_3_preparation.py -v
```

---

## Troubleshooting

**Issue**: `ocr_feasibility` attribute missing
- **Cause**: Image metadata incomplete in master JSON
- **Solution**: Script logs warning and does NOT filter the image

**Issue**: Crop size too small
- **Cause**: Door bounding box is very small (distant container)
- **Solution**: Image is skipped; adjust `min_crop_size` in config if needed

**Issue**: Keypoints outside crop bounds
- **Cause**: Annotation error or bbox mismatch
- **Solution**: Keypoints are clamped to crop boundaries with warning

---

**Last Updated**: 2025-12-23  
**Maintainer**: Data Pipeline Team
