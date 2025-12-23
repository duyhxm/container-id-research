# MODULE 3: IMPLEMENTATION PLAN

## Giai đoạn 1: Chuẩn bị dữ liệu (Data Preparation)

**Mục tiêu:** Chuyển đổi dữ liệu thô từ CVAT sang định dạng YOLOv11-Pose sẵn sàng cho training.

- [ ] **Task 1.1: Setup Project Structure**
    - Tạo thư mục `modules/module_3_container_id_estimation`.
    - Tạo thư mục `data/raw` (chứa dữ liệu CVAT) và `data/processed` (chứa dữ liệu YOLO).

- [ ] **Task 1.2: Implement Conversion Script**
    - File: `src/data/convert_cvat_to_yolo_pose.py`.
    - Chức năng:
        - Đọc file annotation từ CVAT (XML hoặc JSON).
        - Parse toạ độ Polygon.
        - Tính toán Bounding Box bao quanh Polygon.
        - Chuẩn hoá toạ độ (Normalize 0-1).
        - Ghi file `.txt` tương ứng vào thư mục `labels`.
        - Copy ảnh tương ứng vào thư mục `images`.
        - Chia tập Train/Val/Test theo file split có sẵn hoặc random (nếu chưa chia).

- [ ] **Task 1.3: Create Dataset Config**
    - File: `configs/module_3_data.yaml`.
    - Nội dung: Định nghĩa đường dẫn `path`, `train`, `val`, `test`, `names` (classes), và `kpt_shape`.

## Giai đoạn 2: Huấn luyện mô hình (Model Training)
**Mục tiêu:** Fine-tune model YOLOv11s-Pose trên Kaggle.

- [ ] **Task 2.1: Local Training Script**
    - File: `src/train_module_3.py`.
    - Sử dụng `ultralytics.YOLO`.
    - Load pre-trained weights `yolo11s-pose.pt`.
    - Cấu hình tham số train (`data`, `epochs`, `imgsz`, `project`, `name`).

- [ ] **Task 2.2: Kaggle Runner Script**
    - File: `notebooks/kaggle_train_runner.py` (hoặc `.ipynb`).
    - Script này sẽ:
        1. Clone repo từ Github.
        2. Cài đặt thư viện (`pip install ultralytics dvc...`).
        3. Pull dữ liệu (nếu dùng DVC remote) hoặc load từ Kaggle Dataset.
        4. Gọi `src/train_module_3.py`.
        5. Zip folder `runs/` (chứa weights `best.pt` và logs) để download về.

## Giai đoạn 3: Đánh giá & Kiểm thử (Evaluation & Testing)
**Mục tiêu:** Đảm bảo model hoạt động đúng logic nghiệp vụ.

- [ ] **Task 3.1: Evaluation Script**
    - File: `src/evaluate_module_3.py`.
    - Chạy validate trên tập Test set.
    - Xuất ra các chỉ số mAP.

- [ ] **Task 3.2: Visualization & Sanity Check**
    - File: `src/visualize_inference.py`.
    - Input: Một vài ảnh trong tập Test.
    - Process: Load `best.pt`, predict.
    - Output: Lưu ảnh đã vẽ 4 điểm (mỗi điểm 1 màu khác nhau để check thứ tự: VD: Đỏ-Xanh-Vàng-Tím) và khung nối.
    - **Quan trọng:** Kiểm tra xem điểm màu Đỏ có luôn ở góc Trái-Trên hay không.

## Giai đoạn 4: Đóng gói (Packaging for Research Repo)
- [ ] **Task 4.1: Update DVC**
    - Add file `best.pt` vào DVC tracking.
    - Push data và model lên Remote Storage (Google Drive).

- [ ] **Task 4.2: Documentation**
    - Cập nhật `README.md` của Module 3 hướng dẫn cách chạy training và inference.