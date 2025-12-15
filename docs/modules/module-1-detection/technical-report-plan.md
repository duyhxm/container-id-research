# Kế hoạch Viết Báo cáo Kỹ thuật Module 1: Container Door Detection

## 1. Thông tin chung
*   **Tiêu đề báo cáo:** Báo cáo Kỹ thuật Module 1: Nhận diện Cửa Container (Container Door Detection)
*   **Vị trí lưu trữ:** `docs/modules/module-1-detection/technical-report.md`
*   **Ngôn ngữ:** Tiếng Việt (Thuật ngữ chuyên ngành giữ nguyên Tiếng Anh)
*   **Mục tiêu:** Chứng minh tính khoa học, logic và độ tin cậy (robustness) của quy trình triển khai, phục vụ báo cáo cấp quản lý/chuyên môn.

## 2. Cấu trúc chi tiết

### Phần 1: Tóm tắt tổng quan (Executive Summary)
*   **Nội dung:**
    *   Tổng quan về mục tiêu module.
    *   Nêu bật thành tích chính: mAP@50 đạt **99.5%** (vượt mục tiêu 90%).
    *   Thời gian thực hiện và tài nguyên sử dụng (Kaggle T4 x2).
    *   Tóm tắt các quyết định kỹ thuật quan trọng (YOLOv11-Small, Stratified Splitting).

### Phần 2: Giới thiệu & Bối cảnh nghiên cứu
*   **Nội dung:**
    *   Vai trò của Module 1 trong hệ thống trích xuất Container ID (Bước đầu tiên trong Pipeline 5 bước).
    *   Định nghĩa bài toán: Input (ảnh RGB đa dạng) -> Output (Bounding box cửa container).
    *   Các chỉ số KPI đề ra: mAP > 0.90, Latency < 50ms.

### Phần 3: Phân tích Dữ liệu & Thiết kế Dataset
*   **3.1. Phân tích thực trạng dữ liệu (EDA):**
    *   Dựa trên kết quả từ `notebooks/01-annotated-image-eda.ipynb`.
    *   Thống kê dataset: 831 ảnh gốc.
    *   Phân tích sự mất cân bằng (Imbalance):
        *   Góc chụp: Angled (87%) vs Frontal (13%).
        *   Ánh sáng: Bad light (28%).
        *   Che khuất: Occluded (42%).
    *   *Visual:* Biểu đồ phân phối các thuộc tính (Univariate distribution) - `<!-- TODO: Thêm biểu đồ từ EDA notebook -->`.

*   **3.2. Chiến lược phân tầng dữ liệu (Stratification):**
    *   Phương pháp: **Label Powerset Stratification** kết hợp Priority Ranking.
    *   Định nghĩa 3 nhóm ưu tiên (Priority Groups):
        *   **Hard:** `r_env` (bad_light, occluded, not_clean).
        *   **Tricky:** `r_geo` (frontal, blurry).
        *   **Common:** Các trường hợp còn lại.
    *   Mô tả conceptual về hàm ưu tiên $\Phi$ và ví dụ minh họa (Conceptual description + Example).
    *   Chiến lược xử lý Singleton: Original -> Test set, Augmented Copy -> Train set.

*   **3.3. Kết quả phân chia:**
    *   Quy trình xử lý: 831 raw -> 502 processed (Train: 350, Val: 75, Test: 77).
    *   *Visual:* Flowchart quy trình chuẩn bị dữ liệu (Mermaid: COCO -> Stratification -> YOLO format).
    *   *Visual:* Bảng thống kê phân phối các nhóm Hard/Tricky/Common trên các tập Train/Val/Test.

### Phần 4: Lựa chọn Kiến trúc Model
*   **Nội dung:**
    *   So sánh các biến thể YOLOv11 (Nano vs Small vs Medium).
    *   Tiêu chí so sánh: Model size, Inference time, Expected mAP, Memory footprint.
    *   **Lý do chọn YOLOv11-Small:**
        *   Cân bằng tối ưu Speed/Accuracy (Size ~19MB, Inference ~35ms).
        *   Pretrained COCO hỗ trợ Transfer Learning tốt.
        *   Khả năng xử lý đa hướng (Multiple orientations) và che khuất một phần (Partial occlusions) tốt hơn Nano.
        *   Đơn giản nhưng hiệu quả (Simple but effective).

### Phần 5: Thiết kế Augmentation Strategy
*   **Nội dung:**
    *   **Chi tiết cấu hình (từ `params.yaml`):**
        *   HSV (h=0.015, s=0.7, v=0.4).
        *   Geometric (degrees=10, translate=0.1, scale=0.5, shear=10).
        *   Spatial (fliplr=0.5, mosaic=1.0).
    *   **Biện luận (Rationale) dựa trên đặc thù domain:**
        *   **Shear 10°:** Xử lý thực trạng 87% ảnh chụp góc nghiêng.
        *   **High HSV:** Thích nghi với điều kiện ánh sáng đa dạng (cảng, kho bãi, đêm/ngày).
        *   **Fliplr:** Container có tính đối xứng hình học.
        *   **No Flipud:** Container thực tế không bao giờ lật ngược.
    *   *Visual:* Flowchart pipeline augmentation (Mermaid).
    *   *Visual:* Grid ảnh minh họa Augmentation (Gốc vs Biến đổi) - `<!-- TODO: Thêm ảnh minh họa augmentation -->`.

### Phần 6: Triển khai Training Workflow
*   **Nội dung:**
    *   **Môi trường thực nghiệm:**
        *   Lý do chọn Kaggle: Free T4 x2 GPU, Native Secrets API, Ephemeral (đảm bảo tính tái lập - reproducibility).
    *   **Quy trình tự động hóa (Automation):**
        *   Thiết kế **Single-cell workflow** trong `kaggle_training_notebook.py`.
        *   *Visual:* Flowchart 9 bước (GPU verify -> Clone -> Install -> DVC -> WandB -> Pull -> Validate -> Train -> Sync) (Mermaid).
    *   **Cấu hình Hyperparameters:**
        *   Bảng tham số: Epochs (150), Batch size (32), Optimizer (AdamW).
        *   Lý giải sự điều chỉnh: Tăng epochs để đảm bảo convergence, Batch size tối ưu cho T4.

### Phần 7: Kết quả Thực nghiệm & Phân tích
*   **Nội dung:**
    *   **Tiến trình huấn luyện:**
        *   Phân tích các mốc Epochs quan trọng (1, 10, 50, 150).
        *   *Visual:* Bảng số liệu chi tiết (mAP@50, Precision, Recall) tại các mốc.
    *   **Phân tích hội tụ (Convergence):**
        *   Learning curve: Tăng trưởng nhanh (1-25), Plateau (50), Ổn định (50-150).
        *   Phân tích Loss components (Box, Cls, Dfl) - giải thích ý nghĩa sự giảm của từng loại loss.
        *   *Visual:* Biểu đồ Training curves (Loss & mAP) - `<!-- TODO: Thêm biểu đồ results.png -->`.
    *   **So sánh với mục tiêu đề ra:**
        *   mAP@50: **99.5%** (Target > 90%) -> Đạt xuất sắc.
        *   Inference time: **~35ms** (Target < 50ms) -> Đạt.
        *   Model size: **19MB** (Target ~45MB) -> Tốt hơn kỳ vọng.
    *   *Visual:* Confusion Matrix - `<!-- TODO: Thêm confusion matrix -->`.
    *   *Visual:* Ví dụ kết quả nhận diện trên tập Test (Thành công & Edge cases) - `<!-- TODO: Thêm ảnh kết quả test -->`.

### Phần 8: Thách thức Kỹ thuật & Giải pháp
*   **Nội dung (Format: Vấn đề -> Nguyên nhân -> Giải pháp -> Tác động):**
    1.  **DVC Authentication:** Vấn đề Service Account 403 -> Giải pháp Session Token -> Tác động: Tự động hóa hoàn toàn việc push model.
    2.  **Multi-GPU Bug:** Vấn đề Ultralytics #19519 -> Giải pháp Single GPU (`device=0`) -> Tác động: Đảm bảo tính ổn định của metrics, chấp nhận thời gian train lâu hơn.
    3.  **Stratification với dữ liệu nhỏ:** Vấn đề thiếu mẫu hiếm -> Giải pháp Priority Ranking -> Tác động: Test set công bằng, đại diện đủ các trường hợp khó.
    4.  **Reproducibility:** Vấn đề môi trường tạm thời -> Giải pháp Dynamic Install & Pin version -> Tác động: Môi trường đồng nhất mỗi lần chạy.

### Phần 9: Đánh giá & Kết luận
*   **Nội dung:**
    *   Tổng hợp mức độ hoàn thành mục tiêu.
    *   Khẳng định tính khoa học của phương pháp luận (Data-driven, Systematic, Automated).
    *   Đánh giá chung: Module 1 đã sẵn sàng để tích hợp vào pipeline tổng thể.

### Phần 10: Hướng phát triển (Future Work)
*   **Nội dung:**
    *   Export model sang định dạng ONNX/TensorRT để tối ưu hóa deployment trên edge devices.
    *   Mở rộng dataset lên 1000+ ảnh để tăng cường độ bao phủ các trường hợp cực hiếm.
    *   Thử nghiệm FP16 Inference để giảm thêm latency (nếu cần thiết).

## 3. Yêu cầu về trình bày
*   Sử dụng **Mermaid** cho các biểu đồ quy trình (Flowcharts).
*   Sử dụng placeholders `<!-- TODO: ... -->` cho các hình ảnh/biểu đồ cần chèn sau.
*   Giọng văn: Chuyên nghiệp, khách quan, tập trung vào lý luận và minh chứng.
