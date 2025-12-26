# Migration Note: quality_lab → demos/quality

## Summary

The quality assessment demo has been **migrated and refactored**:
- **Old Location**: `scripts/quality_lab/` (1326 lines, standalone functions)
- **New Location**: `demos/quality/` (640 lines, uses `src/quality/` module)

## Key Changes

### ✅ Architecture Improvements
1. **Eliminated Code Duplication**: All quality logic now lives in `src/quality/`
2. **Single Source of Truth**: Demo imports from production module
3. **Type Safety**: Leverages dataclass configs (`QualityConfig`, `PhotometricConfig`, etc.)
4. **Maintainability**: Changes to quality logic automatically propagate to demo

### 📦 Files Created
- `demos/quality/__init__.py` - Module marker
- `demos/quality/app.py` - Refactored Streamlit app (640 lines)
- `demos/quality/launch.py` - Environment verification & launcher
- `demos/quality/README.md` - Comprehensive documentation

### 🗑️ Old Files (To Be Deprecated)
- `scripts/quality_lab/app.py` - Original standalone app (1326 lines)
- `scripts/quality_lab/app.py.backup` - Backup copy
- `scripts/quality_lab/README.md` - Original documentation

**Recommendation**: Keep old files for reference during transition, then remove after validation.

## Functional Comparison

### Retained Features
✅ Real-time quality assessment on uploaded images
✅ Interactive parameter tuning with immediate feedback  
✅ Image distortion controls (brightness, contrast, blur, noise)
✅ All 5 quality metrics (geometric, brightness, contrast, sharpness, naturalness)
✅ Configuration export

### Removed Features
❌ YOLO detection integration (bbox hardcoded as 10-90% of image)
❌ Zoom simulation (crop top-right / padding)
❌ Multi-tab UI (Live Analysis / Parameter Tuning / Documentation)
❌ Negative sample generation workflow

### New Features
✨ **YAML Export**: Production-ready config format (vs. JSON in old version)
✨ **Simpler UI**: Single-page layout focused on tuning workflow
✨ **Production Module Integration**: Uses `QualityAssessor` from `src/quality/`

## Technical Details

### Old Architecture (scripts/quality_lab/)
```python
# Standalone functions (duplicate logic)
def calculate_brightness_metric(img_gray):
    median = np.median(img_gray)
    return float(median)

def brightness_quality_gaussian(m_b, target, sigma):
    exponent = -((m_b - target) ** 2) / (2 * sigma**2)
    return float(np.exp(exponent))

# Manual metric calculation
metrics = calculate_metrics(
    img, brisque_obj,
    brightness_target=100.0,
    brightness_sigma=65.0,
    # ... many parameters
)
```

### New Architecture (demos/quality/)
```python
# Import from production module
from src.quality import QualityAssessor, QualityConfig, PhotometricConfig

# Create type-safe config
custom_config = QualityConfig(
    photometric=PhotometricConfig(
        brightness_target=brightness_target,
        brightness_sigma=brightness_sigma,
        brightness_threshold=brightness_threshold,
    ),
    # ... other configs
)

# Run assessment via production pipeline
assessor = QualityAssessor(config=custom_config)
result = assessor.assess(image, bbox)
```

## Migration Benefits

1. **Code Reuse**: 0% duplication (was ~400 lines of duplicate logic)
2. **Consistency**: Demo always uses production-quality algorithms
3. **Testing**: Single codebase to test and maintain
4. **Evolution**: Improvements to `src/quality/` automatically benefit demo

## Usage

### Old Way (Deprecated)
```bash
cd scripts/quality_lab
streamlit run app.py
```

### New Way
```bash
# From project root
uv run python demos/quality/launch.py

# Or directly
uv run streamlit run demos/quality/app.py
```

## Next Steps

1. ✅ **Validate**: Test new demo with real images
2. ⚠️ **Compare**: Ensure results match old app
3. 📝 **Document**: Update team wiki/docs with new location
4. 🗑️ **Deprecate**: Move old `scripts/quality_lab/` to archive after validation period

---

**Status**: ✅ Migration Complete | New demo ready for use
**Location**: `demos/quality/`
**Integration**: Uses production `src/quality/` module
