# Hướng Dẫn Training Trên Kaggle - Phiên Bản Cuối Cùng

## 📋 Tổng Quan

**Quyết định thiết kế:** Không dùng Poetry trên Kaggle, install dependencies trực tiếp vào system Python.

**Lý do:**
- ✅ Đơn giản hơn, ít lỗi hơn
- ✅ Kaggle environment ephemeral, không cần isolate
- ✅ Faster setup time
- ✅ Native Kaggle workflow

---

## 🚀 Workflow Hoàn Chỉnh

### **Phát triển Local (Máy của bạn)**
```bash
# Clone repo
git clone https://github.com/your-org/container-id-research.git
cd container-id-research

# Setup Poetry environment
poetry install

# Activate virtual environment
poetry shell

# Develop & test
python src/detection/train.py --config params.yaml --experiment test

# Commit & push
git add .
git commit -m "feat(detection): ..."
git push
```

### **Training Trên Kaggle**
```bash
# 1. Clone repo trong notebook
!git clone https://github.com/your-org/container-id-research.git
%cd /kaggle/working/container-id-research

# 2. Install dependencies (system Python)
!pip install -q ultralytics dvc[gdrive] wandb pyyaml pandas opencv-python

# 3. Configure DVC & WandB
# (See notebook cell below)

# 4. Train
!python src/detection/train.py --config params.yaml --experiment exp001
```

---

## 📝 Cách Sử Dụng

### **Bước 1: Mở Kaggle Notebook**

1. Vào notebook hiện tại của bạn (đang chạy SSH tunnel)
   **HOẶC**
2. Tạo notebook mới:
   - New Notebook → Settings → GPU T4 → Internet ON

### **Bước 2: Copy Training Cell**

Mở file **`kaggle_training_notebook.py`** trong repository.

**Copy toàn bộ nội dung** (Ctrl+A → Ctrl+C)

### **Bước 3: Paste Vào Kaggle Notebook**

1. Trong Kaggle notebook, click **"+ Code"**
2. Paste code vào cell (Ctrl+V)
3. Click **Run** (hoặc Shift+Enter)

### **Bước 4: Đợi Training Hoàn Thành**

- ⏱️ Thời gian: ~3-4 giờ (150 epochs, GPU T4 x2)
- 📊 Monitor: https://wandb.ai
- 🔄 Cell sẽ chạy liên tục, đừng đóng browser

### **Bước 5: Download Model**

Sau khi training xong, add cell mới:

```python
from IPython.display import FileLink

# Download trained model
FileLink('weights/detection/best.pt')

# Download metadata
FileLink('weights/detection/metadata.json')
```

Click vào link để download về máy.

---

## ⚙️ Cấu Hình

### **Kaggle Secrets Required**

Trong Kaggle Settings → Add-ons → Secrets, cần có:

1. **`DVC_SERVICE_ACCOUNT_JSON`**
   - Google Service Account JSON (for DVC)
   - Format: Raw JSON string (không base64)

2. **`WANDB_API_KEY`**
   - WandB API key (40 chars)

### **Notebook Settings**

- **Accelerator:** GPU T4 (hoặc P100)
- **Internet:** Enabled
- **Persistence:** Optional (code cloned từ Git)

---

## 🎯 Expected Results

| Metric               | Target | Typical Result |
| -------------------- | ------ | -------------- |
| Validation mAP@50    | > 0.90 | 0.92 - 0.95    |
| Validation mAP@50-95 | > 0.70 | 0.72 - 0.78    |
| Test mAP@50          | > 0.88 | 0.89 - 0.93    |
| Inference time (T4)  | < 50ms | 30-40ms        |
| Model size           | ~45 MB | YOLOv11-Small  |
| Training time        | ~4h    | GPU T4 x2      |

---

## 🐛 Troubleshooting

### **Issue: GPU Not Available**

**Symptom:**
```
❌ GPU NOT AVAILABLE!
```

**Fix:**
1. Settings → Accelerator → GPU T4
2. Save
3. Restart kernel
4. Re-run cell

### **Issue: DVC Credentials Error**

**Symptom:**
```
❌ DVC_SERVICE_ACCOUNT_JSON not found
```

**Fix:**
1. Settings → Add-ons → Secrets
2. Add secret với key `DVC_SERVICE_ACCOUNT_JSON`
3. Enable secret for this notebook
4. Restart kernel

### **Issue: Dataset Fetch Fails**

**Symptom:**
```
ERROR: failed to pull data from the cloud
```

**Fix:**
- Check DVC credentials (above)
- Verify Google Drive permissions
- Share DVC folder with service account email
- Manual fetch:
  ```python
  !dvc pull data/raw.dvc
  !dvc fetch && dvc checkout
  ```

### **Issue: Out of Memory**

**Symptom:**
```
CUDA out of memory
```

**Fix:**

Sửa `params.yaml`:
```yaml
detection:
  training:
    batch_size: 16  # Giảm từ 32
```

Hoặc giảm epochs để test:
```yaml
detection:
  training:
    epochs: 50  # Thay vì 150
```

---

## 📂 File Structure

```
container-id-research/
├── kaggle_training_notebook.py    ← Copy file này vào notebook
├── scripts/
│   └── setup_kaggle_simple.sh     ← Setup script (không dùng Poetry)
├── src/detection/
│   └── train.py                   ← Training script
├── data/processed/detection/      ← Dataset (YOLO format)
├── weights/detection/             ← Output models
├── params.yaml                    ← Hyperparameters
└── pyproject.toml                 ← Dependencies
```

---

## 🔄 Workflow So Sánh

### **Local Development**
```
Poetry → .venv → Isolated environment
```

### **Kaggle Training**
```
pip → System Python → Direct install
```

**Tại sao khác nhau?**
- Local: Cần isolation cho development
- Kaggle: Ephemeral environment, không cần Poetry overhead

---

## 💡 Tips

### **1. Keep Notebook Running**
- Notebook phải chạy suốt 3-4 giờ
- Minimize browser OK, nhưng đừng đóng tab
- Có thể mở tabs khác

### **2. Monitor Training**
- Check WandB dashboard mỗi 30 phút
- Verify loss đang giảm
- Verify mAP đang tăng

### **3. Save Checkpoints**
Training tự động save:
- `best.pt` - Best model (highest mAP)
- `last.pt` - Latest epoch
- Nếu crash, có thể resume

### **4. Backup Artifacts**
Sau khi training xong:
1. Download `best.pt` ngay
2. Push to DVC (optional):
   ```python
   !dvc add weights/detection/best.pt
   !dvc push weights/detection/best.pt.dvc
   ```
3. Download `.dvc` file
4. Commit to Git từ máy local

---

## ✅ Checklist

### **Trước Khi Training**
- [ ] GPU enabled (T4 or P100)
- [ ] Kaggle Secrets configured (DVC + WandB)
- [ ] Repository cloned
- [ ] Có đủ thời gian (3-4 giờ)
- [ ] Internet stable

### **Sau Khi Training**
- [ ] Download `best.pt`
- [ ] Download `metadata.json`
- [ ] Check WandB metrics
- [ ] Push to DVC (optional)
- [ ] Commit `.dvc` files to Git

---

## 📊 Timeline

| Time | Activity                     |
| ---- | ---------------------------- |
| 0:00 | Copy & paste cell, click Run |
| 0:01 | GPU verification ✅           |
| 0:02 | DVC config ✅                 |
| 0:05 | Dataset fetch & validation ✅ |
| 0:06 | Training starts...           |
| 3:30 | Training completes ✅         |
| 3:35 | Download model ✅             |

---

## 🎓 Bài Học

### **Poetry vs Pip trên Kaggle**

| Aspect       | Poetry (.venv) | Pip (system) |
| ------------ | -------------- | ------------ |
| Setup time   | ~5 min         | ~2 min       |
| Complexity   | High           | Low          |
| Errors       | Driver issues  | Minimal      |
| Suitable for | Local dev      | Kaggle/Colab |

**Kết luận:** Dùng Poetry cho local, pip cho cloud platforms.

---

## 📞 Support

Nếu gặp vấn đề:

1. **Check error messages** trong cell output
2. **Verify Kaggle Secrets** đã enable
3. **Check WandB logs** cho training issues
4. **Reduce batch size** nếu OOM
5. **Check GPU status** bằng `!nvidia-smi`

---

**Chúc bạn training thành công! 🚀**

Mọi thứ đã được chuẩn bị sẵn sàng, chỉ cần copy `kaggle_training_notebook.py` vào notebook và run!

