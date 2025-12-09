# DVC Pipeline vs Standalone Files - Explanation

**Date**: December 10, 2024  
**Topic**: Understanding DVC data management in container-id-research project

---

## 🎯 TL;DR

**Dataset `data/processed/detection` KHÔNG có file `.dvc` riêng!**

Nó được quản lý bởi **DVC pipeline** trong `dvc.yaml`, hash được lưu trong `dvc.lock`.

---

## 📚 DVC có 2 cách quản lý data:

### **Method 1: Standalone Files** (`.dvc` files)

**Cách dùng:**
```bash
dvc add data/raw/
dvc add data/annotations/
```

**Kết quả:**
```
data/
├── raw.dvc           ← File tracking metadata
├── annotations.dvc   ← File tracking metadata
├── raw/              (actual data, gitignored)
└── annotations/      (actual data, gitignored)
```

**Git tracks:**
- ✅ `data/raw.dvc`
- ✅ `data/annotations.dvc`
- ❌ NOT the actual data folders

**Push/Pull:**
```bash
dvc push data/raw.dvc
dvc pull data/raw.dvc
```

---

### **Method 2: Pipeline Outputs** (trong `dvc.yaml` + `dvc.lock`)

**Cách định nghĩa:**

```yaml
# dvc.yaml
stages:
  convert_detection:
    cmd: python src/data/coco_to_yolo.py ...
    deps:
      - data/interim/train_master.json
      - src/data/coco_to_yolo.py
    outs:
      - data/processed/detection  ← Pipeline output!
```

**Kết quả:**
```
data/processed/
└── detection/        (actual data, gitignored)

# NO detection.dvc file created!
```

**Hash được lưu trong `dvc.lock`:**
```yaml
# dvc.lock
stages:
  convert_detection:
    outs:
    - path: data/processed/detection
      hash: md5
      md5: 91b20250d3ea6dd41cca724079718820.dir
      size: 111035997
      nfiles: 1005
```

**Git tracks:**
- ✅ `dvc.yaml` (pipeline definition)
- ✅ `dvc.lock` (hash + metadata)
- ❌ NOT `data/processed/detection.dvc` (doesn't exist!)
- ❌ NOT the actual data folder

**Push/Pull:**
```bash
# Push ALL pipeline outputs
dvc push

# Pull ALL pipeline outputs  
dvc pull

# Or specific stage
dvc push -r storage convert_detection
dvc pull convert_detection
```

---

## 🔍 Our Project Structure

### **Standalone `.dvc` files:**
```
data/raw.dvc              → tracks data/raw/
data/annotations.dvc      → tracks data/annotations/
```

### **Pipeline outputs** (NO `.dvc` files):
```
data/interim/
├── augmented_images/     (from split_data stage)
├── train_master.json     (from split_data stage)
├── val_master.json       (from split_data stage)
└── test_master.json      (from split_data stage)

data/processed/
├── detection/            (from convert_detection stage)
└── localization/         (from convert_localization stage)
```

**All tracked in `dvc.lock`, NOT separate `.dvc` files!**

---

## 🚨 Common Mistake (What we fixed)

### **Wrong Code:**
```python
# Looking for detection.dvc (doesn't exist!)
if not os.path.exists("data/processed/detection.dvc"):
    print("❌ DVC tracking file not found!")
    sys.exit(1)

# Trying to pull with .dvc file (wrong!)
os.system("dvc pull data/processed/detection.dvc")
```

### **Correct Code:**
```python
# Check dvc.lock (pipeline metadata)
if not os.path.exists("dvc.lock"):
    print("❌ dvc.lock not found!")
    sys.exit(1)

# Pull all pipeline outputs
os.system("dvc pull")
```

---

## 📊 DVC Pipeline Visualization

```
data/annotations.dvc
        ↓
   [split_data]
        ↓
   data/interim/*
        ↓
 [convert_detection]
        ↓
data/processed/detection  ← NO .dvc file!
```

**See DAG:**
```bash
dvc dag
```

Output:
```
+----------------------+
| data\annotations.dvc |
+----------------------+
            *
            *
         +------------+
         | split_data |
         +------------+
        *             *
       *               *
+-------------------+  +----------------------+
| convert_detection |  | convert_localization |
+-------------------+  +----------------------+
```

---

## 🔧 How to Work with Pipeline Outputs

### **Check Status:**
```bash
# Local vs lock file
dvc status

# Local vs remote (Google Drive)
dvc status -c
```

### **Push to Remote:**
```bash
# Push all pipeline outputs
dvc push

# Push specific stage output
dvc push -r storage convert_detection
```

### **Pull from Remote:**
```bash
# Pull all pipeline outputs
dvc pull

# Pull specific stage
dvc pull convert_detection
```

### **Re-run Pipeline:**
```bash
# Run specific stage
dvc repro convert_detection

# Run entire pipeline
dvc repro
```

### **Commit Changes:**
```bash
# After pipeline run, commit changes to outputs
dvc commit convert_detection

# Then push to remote
dvc push
```

---

## 💡 Key Differences

| Aspect | Standalone `.dvc` | Pipeline Output |
|--------|-------------------|-----------------|
| **Created by** | `dvc add` | `dvc repro` |
| **Tracked in** | `.dvc` file | `dvc.lock` |
| **File exists** | ✅ Yes | ❌ No |
| **Git commits** | `.dvc` file | `dvc.lock` |
| **Push command** | `dvc push file.dvc` | `dvc push` |
| **Pull command** | `dvc pull file.dvc` | `dvc pull` |
| **Dependencies** | None | Defined in `dvc.yaml` |
| **Reproducible** | ❌ No | ✅ Yes (can `dvc repro`) |

---

## 🎓 When to Use Each Method

### **Use Standalone `.dvc` files when:**
- Raw data that never changes
- External datasets downloaded once
- No processing pipeline needed
- Example: `data/raw/`, `data/annotations/`

### **Use Pipeline Outputs when:**
- Data is generated by scripts
- Multiple processing stages
- Need reproducibility
- Want to track parameters
- Example: `data/processed/detection/`, `data/interim/*`

---

## ✅ Our Kaggle Training Fix

**Problem**: Code looked for `detection.dvc` (doesn't exist)

**Solution**: 
1. Check `dvc.lock` instead
2. Use `dvc pull` (all outputs) instead of `dvc pull detection.dvc`
3. Push all outputs: `dvc push` → 3 files pushed
4. Verify: `dvc status -c` → "Cache and remote in sync"

**Status**: ✅ Fixed in commit `ab18776`

---

## 🔗 References

- [DVC Pipeline Documentation](https://dvc.org/doc/user-guide/pipelines)
- [DVC Add vs Pipeline](https://dvc.org/doc/user-guide/pipelines/defining-pipelines#outputs-and-dependencies)
- Project files:
  - `dvc.yaml` - Pipeline definition
  - `dvc.lock` - Pipeline state (hashes, sizes)
  - `.dvc/config` - Remote configuration

---

## 📝 Summary for Kaggle Training

**On Local Machine:**
```bash
# 1. Make sure pipeline outputs are pushed
dvc status -c

# If not in sync:
dvc push

# Verify
dvc status -c  # Should show: "Cache and remote in sync"
```

**On Kaggle Notebook:**
```python
# 1. Clone repo (includes dvc.lock)
git clone https://github.com/duyhxm/container-id-research.git

# 2. Configure DVC with service account
# (done in Step 4 of training notebook)

# 3. Pull ALL pipeline outputs
dvc pull  # ← NOT dvc pull detection.dvc!

# ✅ data/processed/detection/ will be fetched
```

**Key Insight**: Pipeline outputs are tracked collectively in `dvc.lock`, not individual `.dvc` files!

---

**Last Updated**: December 10, 2024  
**Maintainer**: Module 1 Team  
**Status**: Production-ready ✅

