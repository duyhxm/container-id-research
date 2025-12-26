# Alignment Demo Changelog

## [2.0.0] - 2025-12-27

### Added - M1 Enhancement Integration

#### 🎛️ Sigmoid Quality Scoring UI
- **Advanced Sigmoid Controls**: Exposed all 6 sigmoid parameters through collapsible expanders
  - Contrast Sigmoid (Q_C): τ_C, α_C, threshold
  - Sharpness Sigmoid (Q_S): τ_S, α_S, threshold
- **Sigmoid Equation Display**: LaTeX formula shown in UI for educational clarity
- **Legacy Compatibility**: Original M_C and M_S thresholds retained in collapsed expander

#### 📊 Enhanced Quality Visualization
- **Real-time Parameter Display**: Quality scores now show active τ and α values
- **Dynamic Thresholds**: Pass/fail indicators use configurable thresholds (not hardcoded 0.5)
- **Status Indicators**: Clear ✅/⚠️ visual feedback for each quality metric

#### 📐 Bimodal Aspect Ratio Explanation
- **Educational Info Box**: Explains ISO 6346 dual-format nature
  - Mode 1: 2.5–4.5 (multi-line, 2 rows)
  - Mode 2: 5.0–9.0 (single-line, 1 row)
  - Gap: 4.5–5.0 (ambiguous, rejected)
- **Statistical Context**: Notes real-data basis for ranges

#### 🔧 Configuration Export/Import
- **Complete YAML Export**: Now includes all 10 quality parameters
  - 4 legacy: min_height_px, contrast_threshold, sharpness_threshold, sharpness_normalized_height
  - 6 sigmoid: contrast_tau, contrast_alpha, contrast_quality_threshold, sharpness_tau, sharpness_alpha, sharpness_quality_threshold
- **Full Config Pipeline**: create_custom_config() and export_config_to_yaml() updated

### Changed
- **Function Signatures**: create_custom_config() now accepts 10 quality parameters (was 4)
- **UI Layout**: Reorganized sidebar for better information hierarchy
  - Basic controls remain visible
  - Advanced sigmoid parameters in collapsible sections
  - Legacy thresholds collapsed by default

### Technical Details
- **Files Modified**: demos/align/app.py
- **Lines Changed**: ~150 LOC modified across 8 sections
- **Backward Compatible**: Existing config files still work with defaults

### UX/UI Design Principles Applied
1. **Progressive Disclosure**: Advanced controls hidden until needed
2. **Visual Hierarchy**: Most common parameters remain prominent
3. **Contextual Help**: Tooltips and info boxes explain technical concepts
4. **Immediate Feedback**: Live parameter values in result displays
5. **Clean Layout**: Expanders prevent UI clutter

---

## [1.0.0] - Previous Version
- Basic alignment parameter tuning
- Aspect ratio configuration
- Legacy quality thresholds only
