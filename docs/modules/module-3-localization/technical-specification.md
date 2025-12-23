# MODULE 3: CONTAINER ID REGION ESTIMATION - TECHNICAL SPECIFICATION

## 1. Tổng quan (Overview)
Module này chịu trách nhiệm xác định vùng chứa mã số container (`container_id`) từ ảnh chụp cửa sau container. Phương pháp được sử dụng là **Keypoint Detection** (Pose Estimation) để xác định 4 góc của vùng văn bản, nhằm phục vụ cho việc đánh giá độ nghiêng và nắn chỉnh phối cảnh (perspective rectification) sau này.

## 2. Phạm vi (Scope) - Research Repo
- Huấn luyện và đánh giá model trên tập dữ liệu đã gán nhãn.
- Xây dựng pipeline xử lý dữ liệu từ định dạng CVAT sang định dạng YOLO Pose.
- Thực hiện training trên nền tảng Kaggle.
- Tracking thí nghiệm và quản lý version dữ liệu/model (DVC/Git).

## 3. Kiến trúc Model (Model Architecture)
- **Base Model:** YOLOv11s-Pose (Small version).
- **Lý do chọn:** Cân bằng tốt giữa tốc độ (Real-time inference) và độ chính xác (mAP) trên các thiết bị biên hoặc server tầm trung.
- **Input:** Ảnh màu (RGB). Trong giai đoạn Research, input là ảnh crop cửa sau container.
- **Output:** Toạ độ 4 điểm keypoints $(x, y)$ và Confidence score.

## 4. Định nghĩa dữ liệu (Data Definition)

### 4.1. Keypoints Definition
Vùng `container_id` được định nghĩa bởi 4 điểm keypoints theo thứ tự cố định (Topology):
- **Số lượng:** 4 điểm.
- **Thứ tự (Order):** Theo chiều kim đồng hồ (Clockwise), bắt đầu từ góc trên-trái.
    - Index 0: Top-Left (TL)
    - Index 1: Top-Right (TR)
    - Index 2: Bottom-Right (BR)
    - Index 3: Bottom-Left (BL)
- **Trạng thái:** Tất cả các điểm đều được coi là Visible (v=2 trong định dạng YOLO).

### 4.2. Dataset Structure (YOLO Format)
Cấu trúc thư mục chuẩn bị cho training:
```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
- **Label Format (.txt):**
  `<class-index> <x_center> <y_center> <width> <height> <px1> <py1> <v1> <px2> <py2> <v2> ...`
  - `class-index`: 0 (container_id)
  - `x_center`, `y_center`, `width`, `height`: Bounding box bao quanh 4 điểm (được tính toán từ polygon).
  - `px, py`: Toạ độ keypoint đã chuẩn hoá (normalized 0-1).
  - `v`: Visibility (mặc định = 2).

## 5. Yêu cầu chức năng (Functional Requirements)

### 5.1. Data Preprocessing
- Script chuyển đổi dữ liệu từ CVAT (Polygon XML/JSON) sang YOLO Pose TXT.
- Logic chuyển đổi:
    - Input: Polygon 4 điểm.
    - Process:
        1. Xác định Min/Max X, Y để tạo Bounding Box (cx, cy, w, h).
        2. Map 4 điểm polygon vào 4 keypoints theo đúng thứ tự index 0->3.
        3. Normalize toạ độ theo kích thước ảnh.

### 5.2. Training Pipeline (Kaggle Compatible)
- Code phải tách biệt logic định nghĩa model và logic chạy trên Kaggle.
- Sử dụng `ultralytics` library.
- Hyperparameters cơ bản:
    - `imgsz`: 640
    - `epochs`: 100 (dự kiến, có thể early stop)
    - `batch`: 16 hoặc 32 (tuỳ GPU Kaggle P100/T4)
    - `device`: 0 (GPU)
    - `kpt_shape`: [4, 2] (4 điểm, x-y coordinates)

### 5.3. Inference Interface (System Integration)

Module 3 không hoạt động độc lập mà là một mắt xích trong pipeline.

- **Input Interface:**
  Hàm inference của Module 3 sẽ nhận vào một Data Object (từ output của Module 1), bao gồm:
  - `cropped_image` (numpy.ndarray): Ảnh RGB đã cắt vùng cửa xe. Đây là dữ liệu chính để model Pose xử lý.
  - *(Optional)* `original_shape` & `bbox`: Các thông tin này có thể được truyền kèm nhưng Module 3 không trực tiếp sử dụng cho việc predict keypoints, chỉ pass-through (truyền tiếp) nếu cần.

- **Processing Logic:**
  1. Nhận `cropped_image`.
  2. Feed vào model YOLOv11-Pose (model tự động resize/letterbox).
  3. Nhận về toạ độ 4 keypoints $(x, y)$ tương ứng trên hệ toạ độ của `cropped_image`.

- **Output Interface:**
  Module 3 trả về kết quả bao gồm toạ độ cục bộ (trên ảnh crop) để phục vụ Module 4 và 5:
  
  ```python
  {
      "keypoints": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], # Toạ độ trên ảnh crop
      "confidence": [c1, c2, c3, c4] # Độ tin cậy của từng điểm
  }
  ```

- **Exception Handling:**
  - Nếu `cropped_image` là None hoặc rỗng -> Raise Error.
  - Nếu Model không detect được đủ 4 điểm -> Return None (để hệ thống log lỗi và bỏ qua ảnh này).

## 6. Tiêu chí đánh giá (Evaluation Metrics)

Để đảm bảo chất lượng đầu vào cho Module 5 (OCR), hệ thống sử dụng đa tầng chỉ số đánh giá:

### 6.1. Standard Metrics (Training Monitor)
- **mAP@50-95 (Pose):** Chỉ số chính để theo dõi quá trình hội tụ của model trên Kaggle.

### 6.2. Functional Metrics (Quality Assurance)
Các chỉ số này được tính toán trong script `evaluate_module_3.py` trên tập Test set:

1.  **Mean Euclidean Distance Error (MDE):**
    - Đơn vị: Pixel (trên kích thước ảnh gốc hoặc ảnh crop 640x640).
    - Mục tiêu: MDE < 5 pixels.
    - Công thức: Trung bình cộng khoảng cách Euclid giữa điểm dự đoán và điểm Ground Truth tương ứng.

2.  **Polygon IoU (Intersection over Union):**
    - Tạo Polygon từ 4 điểm dự đoán và 4 điểm nhãn.
    - Tính diện tích giao / diện tích hợp.
    - Mục tiêu: IoU > 0.85.

3.  **Topology Accuracy (Correct Ordering Rate):**
    - Kiểm tra xem điểm được dự đoán là `index 0` có thực sự nằm ở góc trên-trái (dựa trên toạ độ tương đối với các điểm còn lại) hay không.
    - Mục tiêu: 100%. Nếu < 100%, cần xem lại data augmentation (tránh flip ảnh bừa bãi).