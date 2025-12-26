# KẾ HOẠCH TRIỂN KHAI NGHIÊN CỨU: MODULE 2
**Chiến lược:** Step-by-Step Validation (Kiểm chứng từng bước).
**Công cụ:** Python, OpenCV, Jupyter Notebook.

---

#### TASK 0: INFERENCE & DATA PREPARATION

*(Chạy Model & Chuẩn bị dữ liệu)*

**Mục tiêu:** Tạo ra file `predictions.json` chứa bounding box dự đoán cho toàn bộ tập ảnh, mô phỏng output thực tế của Module 1.

*   **Yêu cầu đầu vào cho Agent:**
    *   File model weights: `weights/detection/best.pt` (hoặc đường dẫn tới model YOLOv11s đã train).
    *   Folder `data/raw/`.
*   **Các bước thực hiện:**
    1.  Load model YOLOv11s.
    2.  Chạy inference trên toàn bộ ảnh trong `data/raw/`.
    3.  Lưu kết quả ra file `predictions_full.json` với cấu trúc:

        - Cấu trúc mỗi phần tử:
        
        ```json
        {"image_id": <int>, "file_name": <string>, "category_id": 1, "bbox": [x, y, w, h], "score": <float>}
        ```
        - **Logic xử lý quan trọng:**

          1. Không được đọc file `annotations.json`
          2. `file_name`: Lấy trực tiếp từ tên file ảnh đang xử lý.
          3. `image_id`: Hãy trích xuất phần số từ tên file ảnh để làm ID (ví dụ: `0016135.jpg` -> `16135`). Nếu tên file không phải số, hãy tự tạo ID tăng dần (0, 1, 2...).
          4. `bbox`: Model YOLO trả về format `[x_center, y_center, w, h]` hoặc `[x1, y1, x2, y2]`. Hãy chuyển đổi nó về đúng định dạng COCO là `[x_top_left, y_top_left, width, height]`.
          5. Chỉ lưu các box có `confidence score > 0.5`.

## TASK 1: GEOMETRIC CHECK & ROI EXTRACTION
*(Kiểm tra Hình học & Cắt vùng quan tâm)*

**Mục tiêu:** Xây dựng Notebook `01_geometric_check.ipynb` để lọc các ảnh có bố cục sai (quá xa/quá gần) và tạo dữ liệu đầu vào chuẩn (Cropped ROIs) cho các bước sau.

*   **Yêu cầu đầu vào cho Agent:**
    *   Folder `data/raw/`: Chứa ảnh gốc.
    *   File `predictions_full.json`: Chứa tọa độ Bounding Box (output giả lập từ Module 1).
*   **Các bước thực hiện (Sub-tasks):**
    1.  Load ảnh và Bounding Box.
    2.  Tính tỷ lệ diện tích: $R = \text{Area}_{BBox} / \text{Area}_{Image}$.
    3.  Thực hiện lọc theo ngưỡng (ví dụ: $10\% < R < 90\%$).
    4.  Visualize: Vẽ BBox lên ảnh gốc, ghi chú Pass/Reject lên hình.
    5.  **Quan trọng:** Crop các vùng container (ROI) của các ảnh Pass và lưu vào folder mới `processed_rois/` để dùng cho Task 2, 3, 4.
*   **Deliverable (Kết quả bàn giao):**
    *   Folder `processed_rois/` chứa ảnh container đã crop.
    *   Báo cáo thống kê: Bao nhiêu ảnh Pass, bao nhiêu ảnh Reject do lỗi hình học.

---

## TASK 2: PHOTOMETRIC ANALYSIS (BRIGHTNESS & CONTRAST)
*(Phân tích Quang trắc: Độ sáng & Tương phản)*

**Mục tiêu:** Xây dựng Notebook `02_photometric_analysis.ipynb` để đo lường và chấm điểm độ sáng, tương phản dựa trên Histogram và Robust Statistics.

*   **Yêu cầu đầu vào cho Agent:**
    *   Folder `processed_rois/` (Kết quả từ Task 1).
*   **Các bước thực hiện (Sub-tasks):**
    1.  Đọc ảnh, chuyển sang Grayscale (Luminance).
    2.  Tính Histogram và các giá trị phân vị: $P_5, P_{50} (Median), P_{95}$.
    3.  Tính toán Metric:
        *   Brightness Metric $M_B = P_{50}$.
        *   Contrast Metric $M_C = P_{95} - P_5$.
    4.  Cài đặt hàm đánh giá chất lượng (Mapping Functions):
        *   $Q_B$ dùng hàm Gaussian (Target=128, Sigma=50).
        *   $Q_C$ dùng hàm Sigmoid.
    5.  Visualize: Vẽ biểu đồ Histogram cạnh bức ảnh. Hiển thị giá trị $M_B, M_C$ và điểm $Q_B, Q_C$.
*   **Deliverable:**
    *   Xác định được ngưỡng (Threshold) tối ưu cho $Q_B$ và $Q_C$ dựa trên việc quan sát các ảnh bị loại.

---

## TASK 3: STRUCTURAL ANALYSIS (SHARPNESS)
*(Phân tích Cấu trúc: Độ nét)*

**Mục tiêu:** Xây dựng Notebook `03_sharpness_analysis.ipynb` để phát hiện ảnh mờ (Blur detection).

*   **Yêu cầu đầu vào cho Agent:**
    *   Folder `processed_rois/`.
*   **Các bước thực hiện (Sub-tasks):**
    1.  Đọc ảnh Grayscale.
    2.  Áp dụng toán tử Laplacian (`cv2.Laplacian`).
    3.  Tính phương sai (Variance) của kết quả Laplacian $\rightarrow M_S$.
    4.  Cài đặt hàm chuẩn hóa điểm số $Q_S$.
    5.  Thử nghiệm trên tập dữ liệu: Sắp xếp các ảnh theo thứ tự độ nét tăng dần để tìm ra "điểm cắt" (cut-off point) giữa ảnh mờ và ảnh nét.
*   **Deliverable:**
    *   Tìm ra ngưỡng `Threshold_Sharp` (ví dụ: variance < 100 là mờ) phù hợp với camera thực tế.

---

## TASK 4: STATISTICAL ANALYSIS (NATURALNESS)
*(Phân tích Thống kê: Độ tự nhiên/Nhiễu)*

**Mục tiêu:** Xây dựng Notebook `04_naturalness_brisque.ipynb` để phát hiện nhiễu và lỗi nén ảnh.

*   **Yêu cầu đầu vào cho Agent:**
    *   Folder `processed_rois/`.
    *   File model BRISQUE: `brisque_model_live.yml` và `brisque_range_live.yml` (Agent cần được cung cấp hoặc tự tải).
*   **Các bước thực hiện (Sub-tasks):**
    1.  Khởi tạo mô hình BRISQUE từ OpenCV (`cv2.quality.QualityBRISQUE`).
    2.  Tính điểm BRISQUE Score ($M_N$) cho từng ảnh.
    3.  Chuyển đổi sang thang điểm chất lượng $Q_N$ (Đảo ngược: điểm thấp là tốt, cao là xấu).
    4.  Phân tích các trường hợp ngoại lệ (Outliers): Xem những ảnh có điểm BRISQUE rất cao (tệ) trông như thế nào (nhiễu hạt, vỡ hình...).
*   **Deliverable:**
    *   Xác định ngưỡng loại bỏ (ví dụ: BRISQUE Score > 80 là Reject).

---

## TASK 5: PIPELINE INTEGRATION & FINAL VALIDATION
*(Tích hợp Pipeline & Đánh giá tổng thể)*

**Mục tiêu:** Xây dựng Notebook `05_pipeline_integration.ipynb`. Đây là bản nháp của code Production. Kết hợp tất cả logic thành một hàm duy nhất.

*   **Yêu cầu đầu vào cho Agent:**
    *   Dữ liệu gốc (`raw_images` + `labels`).
    *   Các tham số/ngưỡng đã chốt được từ Task 2, 3, 4.
*   **Các bước thực hiện (Sub-tasks):**
    1.  Viết Class `ImageUsabilityEvaluator`:
        *   Method `check_geometry()`
        *   Method `check_photometric()`
        *   Method `check_sharpness()`
        *   Method `check_naturalness()`
        *   Method `evaluate()` (Chạy tuần tự Cascade).
    2.  Chạy thử nghiệm trên toàn bộ tập dữ liệu.
    3.  Xuất báo cáo CSV: `Filename | Status (Pass/Fail) | Reason | Scores (B, C, S, N)`.
    4.  Review kết quả: Kiểm tra xem có ảnh nào "Oan sai" (False Positive) hoặc "Lọt lưới" (False Negative) không.
*   **Deliverable:**
    *   Code hoàn chỉnh của Module 2.
    *   File CSV báo cáo kết quả kiểm thử.