# TÀI LIỆU ĐẶC TẢ KỸ THUẬT: PIPELINE XỬ LÝ DỮ LIỆU CONTAINER
**Project:** Container Inspection System (Door Detection & ID Localization)
**Version:** 1.0
**Tech Stack:** Python 3.13, DVC (Data Version Control), Scikit-learn, Albumentations.

---

## 1. Tổng quan Kiến trúc (System Architecture)

Hệ thống tuân theo mô hình **Decoupled 2-Stage Pipeline** (Tách biệt 2 giai đoạn) để đảm bảo tính nhất quán của dữ liệu giữa các module.

*   **Stage 1 (Strategy Layer):** Chịu trách nhiệm phân tích thuộc tính, phân tầng (Stratification), xử lý mẫu hiếm (Singleton) và chia tập dữ liệu. Đầu ra là các file JSON chuẩn COCO (Master Splits).
*   **Stage 2 (Adapter Layer):** Chịu trách nhiệm chuyển đổi từ Master JSON sang định dạng YOLO cụ thể cho từng bài toán (Detection/Pose) và lọc dữ liệu theo nghiệp vụ.

### Sơ đồ luồng dữ liệu (Data Flow)

```mermaid
graph LR
    Raw[Raw Data (Images + JSON)] --> Stage1[Stage 1: Split & Stratify]
    Stage1 --> Master[Master JSONs (Train/Val/Test)]
    Master --> Adapter1[Adapter Module 1]
    Master --> Adapter2[Adapter Module 3]
    Adapter1 --> YOLO_Det[YOLO Detection Dataset]
    Adapter2 --> YOLO_Pose[YOLO Pose Dataset]
```

---

## 2. Cấu trúc Thư mục Dự án (Directory Structure)

```text
project_root/
├── dvc.yaml                # DVC Pipeline definition
├── params.yaml             # Cấu hình tham số (seed, split_ratios...)
├── data/
│   ├── raw/                # Dữ liệu gốc (Read-only)
│   │   ├── images/
│   │   └── annotations.json
│   ├── interim/            # Dữ liệu trung gian (Output Stage 1)
│   │   ├── augmented_images/ # Ảnh sinh ra từ Singleton handling
│   │   ├── train_master.json
│   │   ├── val_master.json
│   │   └── test_master.json
│   └── processed/          # Dữ liệu đích (Output Stage 2)
│       ├── module1_door/   # YOLO Format cho Detect
│       └── module3_id/     # YOLO Format cho Pose
└── src/
    ├── stages/
    │   ├── data_split.py   # Code cho Stage 1
    │   └── data_convert.py # Code cho Stage 2
    └── utils/              # Các hàm phụ trợ (stratify logic, iou...)
```

---

## 3. Chi tiết Stage 1: Phân tầng & Chia tập (Data Splitting)

**Mục tiêu:** Tạo ra 3 file JSON bất biến (`train`, `val`, `test`) đảm bảo phân phối công bằng các trường hợp khó (`s_hard`, `s_tricky`).

### 3.1. Input & Configuration
*   **Input:** `data/raw/annotations.json`
*   **Params (`params.yaml`):**
    ```yaml
    split:
      seed: 42
      test_size: 0.15
      val_size: 0.15
      min_instance: 2 # Số lượng tối thiểu để không bị coi là singleton
    ```

### 3.2. Logic Xử lý (Processing Logic)

**Bước 1: Chuẩn bị dữ liệu gốc (Pre-processing)**
* Duyệt qua danh sách ảnh gốc.
* Thêm trường `rel_path` vào đối tượng ảnh trong JSON. Giá trị là đường dẫn tương đối từ root: `data/raw/images/{file_name}`.

**Bước 2: Gán nhãn phân tầng (Stratification Labeling)**
Với mỗi ảnh trong JSON, áp dụng hàm ánh xạ $\Phi$ để tạo trường `s_label`.
*   *Input:* Danh sách `attributes` của ảnh.
*   *Mapping Logic:*
    *   `R_env` (Environment): `['bad_light', 'occluded', 'not_clean']`
    *   `R_geo` (Geometry): `['frontal', 'blurry']`
    *   **Rule:**
        *   Nếu có bất kỳ thuộc tính nào thuộc `R_env` $\rightarrow$ `s_label = 'hard'`
        *   Else nếu có thuộc tính thuộc `R_geo` $\rightarrow$ `s_label = 'tricky'`
        *   Else $\rightarrow$ `s_label = 'common'`

**Bước 3: Phát hiện & Xử lý Singleton (Singleton Handling)**
*   Gom nhóm dữ liệu theo `s_label` kết hợp với các tổ hợp thuộc tính chi tiết.
*   Nếu nhóm nào chỉ có **1 mẫu (Singleton)**:
    1.  Đánh dấu mẫu gốc này thuộc tập **TEST** (để kiểm thử thực tế).
    2.  Thực hiện **Augmentation** (Sử dụng `albumentations`):
        *   Action: `HorizontalFlip` hoặc `RandomBrightnessContrast` (nhẹ).
        *   Naming Convention: Tên file mới = `aug_{original_filename}`.
        *   Save Image: Lưu ảnh mới vào `data/interim/augmented_images/`.
        *   JSON Entry Update: Tạo entry mới cho ảnh này. Thiết lập `rel_path` = `data/interim/augmented_images/{aug_filename}`.
        *   Create Annotation: Tạo entry mới trong JSON cho ảnh này, sao chép annotations từ ảnh gốc (nhớ lật tọa độ bbox/polygon nếu flip ảnh).
        *   Assignment: Gán mẫu nhân bản này vào tập **TRAIN**.

**Bước 4: Chia tập (Splitting)**
*   Sử dụng `sklearn.model_selection.train_test_split`.
*   **Stratify key:** `s_label`.
*   Thực hiện chia 2 lần:
    1.  Full $\rightarrow$ Train + Temp (theo tỷ lệ Train / (Val+Test)).
    2.  Temp $\rightarrow$ Val + Test (theo tỷ lệ 50/50).

### 3.3. Output
*   3 file JSON: `data/interim/{train,val,test}_master.json`.
*   * Cấu trúc JSON mở rộng: Tuân theo chuẩn COCO nhưng bổ sung trường `rel_path` trong danh sách `images`.
        * Ví dụ ảnh gốc: `"rel_path": "data/raw/images/img1.jpg"`
        * Ví dụ ảnh tăng cường: `"rel_path": "data/interim/augmented_images/aug_img1.jpg"`
        * *Mục đích:* Giúp Stage 2 định vị file ảnh mà không cần hard-code đường dẫn thư mục.

---

## 4. Chi tiết Stage 2: Chuyển đổi định dạng (Adapters)

**Mục tiêu:** Chuyển đổi từ Master JSON sang YOLO TXT, áp dụng bộ lọc nghiệp vụ riêng cho từng Module.

**Cơ chế Copy Ảnh (Unified Path Logic):**
Cả hai Adapter (Module 1 & 3) đều sử dụng chung một logic để copy ảnh từ nguồn sang thư mục đích (`processed`):

*   Đọc giá trị `rel_path` từ đối tượng ảnh trong file Master JSON.
*   Thực hiện lệnh copy từ `project_root / rel_path` $\rightarrow$ `processed_dir / images / file_name`.
*   **Lợi ích:** Adapter không cần quan tâm ảnh nằm ở `raw` hay `interim`, loại bỏ hoàn toàn các câu lệnh `if/else` kiểm tra đường dẫn phức tạp.

### 4.1. Adapter Module 1: Container Door Detection
*   **Input:** `data/interim/{train,val,test}_master.json`
*   **Filter Logic:**
    *   Chỉ lấy annotation có `category_id == <ID_CUA_CONTAINER>`.
*   **Conversion Logic (BBox):**
    *   COCO `[x_top_left, y_top_left, w, h]` (pixel) $\rightarrow$ YOLO `[x_center, y_center, w, h]` (normalized 0-1).
*   **Output:**
    *   Thư mục: `data/processed/module1_door/`
    *   Files: `images/`, `labels/`, `data.yaml`.

### 4.2. Adapter Module 3: Container ID Pose
*   **Input:** `data/interim/{train,val,test}_master.json`
*   **Filter Logic (Quan trọng):**
    *   Check `image.attributes['ocr_feasibility']`.
    *   **IF** `split == 'train'` **AND** `ocr_feasibility == 'unreadable'` $\rightarrow$ **DROP** (Không xuất file label, model không học ảnh này).
    *   **IF** `split == 'test'` $\rightarrow$ **KEEP** (Giữ lại để đánh giá khả năng model xử lý ca khó hoặc từ chối nhận diện).
    *   Chỉ lấy annotation có `category_id == <ID_MA_SO_CONTAINER>`.
*   **Conversion Logic (Keypoints):**
    *   Input: COCO Segmentation (Polygon 4 điểm).
    *   Output: YOLO Pose `[class, x_c, y_c, w, h, x1, y1, 2, x2, y2, 2, x3, y3, 2, x4, y4, 2]`.
    *   *Lưu ý:* Tính BBox bao quanh 4 điểm để điền vào 4 tham số đầu.
*   **Output:**
    *   Thư mục: `data/processed/module3_id/`
    *   Files: `images/`, `labels/`, `data.yaml`.

---

## 5. Tích hợp DVC (DVC Pipeline)

File `dvc.yaml` sẽ định nghĩa quy trình chạy tự động.

```yaml
stages:
  # Giai đoạn 1: Phân chia dữ liệu
  split_data:
    cmd: python src/stages/data_split.py --config params.yaml
    deps:
      - src/stages/data_split.py
      - data/raw/annotations.json
    params:
      - split.seed
      - split.test_size
    outs:
      - data/interim/train_master.json
      - data/interim/val_master.json
      - data/interim/test_master.json
      - data/interim/augmented_images/

  # Giai đoạn 2A: Convert cho Module 1
  convert_mod1:
    cmd: python src/stages/data_convert.py --task detection --input data/interim --output data/processed/module1_door
    deps:
      - src/stages/data_convert.py
      - data/interim/train_master.json
    outs:
      - data/processed/module1_door

  # Giai đoạn 2B: Convert cho Module 3
  convert_mod3:
    cmd: python src/stages/data_convert.py --task pose --input data/interim --output data/processed/module3_id
    deps:
      - src/stages/data_convert.py
      - data/interim/train_master.json
    outs:
      - data/processed/module3_id
```

---

## 6. Hướng dẫn Triển khai (Implementation Steps)

1.  **Setup Environment:**
    *   Tạo virtualenv.
    *   Cài đặt: `dvc`, `pandas`, `scikit-learn`, `albumentations`, `pyyaml`, `tqdm`.
2.  **Initialize DVC:** `dvc init`.
3.  **Coding Stage 1 (`src/stages/data_split.py`):**
    *   Implement class `StratifiedSplitter`.
    *   Viết hàm `handle_singletons()` dùng Albumentations.
4.  **Coding Stage 2 (`src/stages/data_convert.py`):**
    *   Implement class `COCO2YOLOConverter`.
    *   Viết method `convert_detection()` và `convert_pose()`.
5.  **Run Pipeline:**
    *   Chạy lệnh `dvc repro`.
    *   DVC sẽ tự động chạy Stage 1 trước, sau đó chạy song song Stage 2A và 2B.
6.  **Verify:**
    *   Kiểm tra thư mục `processed/`. Mở file `.txt` bất kỳ để xem tọa độ đã chuẩn hóa chưa.
    *   Kiểm tra xem ảnh Singleton trong tập Train của Module 3 đã bị loại bỏ chưa (nếu unreadable).
