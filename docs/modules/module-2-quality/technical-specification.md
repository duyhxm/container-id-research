# TÀI LIỆU ĐẶC TẢ KỸ THUẬT

## MODULE 2: INPUT IMAGE USABILITY ASSESSMENT

**(Đánh giá Khả năng sử dụng của Ảnh đầu vào)**

### 1. TỔNG QUAN (OVERVIEW)

*   **Mục tiêu:** Đóng vai trò là "Cổng kiểm soát chất lượng" (Quality Gatekeeper). Module có nhiệm vụ phân tích ảnh đầu vào và vùng chứa container (ROI) để quyết định xem ảnh có đủ tiêu chuẩn kỹ thuật phục vụ cho quá trình OCR ở Module 5 hay không.
*   **Đầu vào:**
    1.  Ảnh gốc (Raw RGB Image).
    2.  Tọa độ Bounding Box của thực thể container (Output từ Module 1).
*   **Đầu ra:**
    1.  Quyết định: `PASS` (Chấp nhận) hoặc `REJECT` (Từ chối).
    2.  Vector điểm chất lượng (Quality Score Vector) để log lại nguyên nhân từ chối.

---

### 2. CƠ SỞ LÝ LUẬN & PHƯƠNG PHÁP TIẾP CẬN (THEORETICAL BASIS)

Hệ thống đánh giá dựa trên **Mô hình Chất lượng Dựa trên Nhiệm vụ (Task-Based Quality Assessment)**, cụ thể là nhiệm vụ OCR. Các đặc điểm được lựa chọn và đo lường dựa trên hai nền tảng:
1.  **Vật lý (Physical-based):** Đo lường tín hiệu ánh sáng và cấu trúc cạnh (Brightness, Contrast, Sharpness).
2.  **Thống kê (Statistical-based):** Đo lường sự phân bố tự nhiên của điểm ảnh (Naturalness).

Để đảm bảo tính khoa học và loại bỏ sự cảm tính, các giá trị đo lường thô (Raw Metrics) sẽ được ánh xạ qua các **Hàm Chất lượng (Quality Mapping Functions)** để chuẩn hóa về thang điểm [0.0 - 1.0].

---

### 3. ĐẶC TẢ CÁC ĐẶC ĐIỂM (FEATURE SPECIFICATION)

Chúng ta đánh giá 4 đặc điểm cốt lõi sau đây trên vùng ảnh container (đã crop):

#### 3.1. Brightness (Độ sáng)
*   **Định nghĩa:** Mức năng lượng trung tâm của phân bố ánh sáng.
*   **Phương pháp đo lường:** **Robust Statistics** trên Histogram kênh Luminance.
*   **Metric ($M_B$):** Trung vị (Median - Percentile 50th) của Histogram.
    $$ M_B = P_{50}(Hist_{Luminance}) $$
*   **Hàm đánh giá ($Q_B$):** Sử dụng **Hàm Gaussian**. Chúng ta kỳ vọng độ sáng nằm quanh mức trung bình (128 hoặc có thể khác).
    $$ Q_B = \exp \left( - \frac{(M_B - 128)^2}{2 \cdot \sigma_B^2} \right) $$
    *(Gợi ý: $\sigma_B \approx 40-50$)*

#### 3.2. Contrast (Độ tương phản)
*   **Định nghĩa:** Độ rộng hiệu dụng của dải động (Dynamic Range), loại bỏ nhiễu ở hai cực.
*   **Phương pháp đo lường:** **Robust Range** trên Histogram kênh Luminance.
*   **Metric ($M_C$):** Khoảng cách giữa phân vị 95 và phân vị 5.
    $$ M_C = P_{95} - P_{5} $$
*   **Hàm đánh giá ($Q_C$):** Sử dụng **Hàm Sigmoid**. Tương phản càng cao càng tốt, nhưng sẽ bão hòa.
    $$ Q_C = \frac{1}{1 + \exp \left( -k \cdot (M_C - T_C) \right)} $$
    *(Gợi ý: $T_C \approx 50$ là ngưỡng chấp nhận, $k$ là độ dốc)*

#### 3.3. Sharpness (Độ nét)
*   **Định nghĩa:** Mức độ biến thiên đột ngột của cường độ sáng tại các cạnh (edges).
*   **Phương pháp đo lường:** Đạo hàm bậc 2 trong miền không gian.
*   **Metric ($M_S$):** Phương sai của toán tử Laplacian (Variance of Laplacian).
    $$ M_S = \text{Var}(\nabla^2 I) $$
*   **Hàm đánh giá ($Q_S$):** Sử dụng hàm chuẩn hóa tuyến tính có ngưỡng bão hòa (Clipped Linear).
    $$ Q_S = \min \left( \frac{M_S}{Threshold_{Sharp}}, 1.0 \right) $$
    *(Gợi ý: $Threshold_{Sharp} \approx 100-150$)*

#### 3.4. Naturalness (Độ tự nhiên - Noise/Artifacts)
*   **Định nghĩa:** Mức độ tuân thủ quy luật thống kê cảnh tự nhiên (NSS). Dùng để phát hiện nhiễu hạt (Noise) và vỡ nén (Artifacts).
*   **Phương pháp đo lường:** Thuật toán **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator).
*   **Metric ($M_N$):** Điểm BRISQUE thô (Thang 0-100, 0 là tốt nhất).
*   **Hàm đánh giá ($Q_N$):** Ánh xạ nghịch đảo.
    $$ Q_N = 1.0 - \frac{M_N}{100} $$
    *(Lưu ý: Nếu $M_N > 100$, gán $Q_N = 0$)*

---

### 4. QUY TRÌNH XỬ LÝ (PROCESSING PIPELINE)

Module sẽ thực hiện theo quy trình tuần tự (Cascade) để tối ưu hiệu năng. Nếu ảnh trượt các bài test sơ cấp, sẽ bị loại ngay lập tức mà không cần tính toán các bước phức tạp sau.

**Bước 1: Pre-check Hình học (Geometric Check)**
*   *Input:* Bounding Box (BBox) từ Module 1.
*   *Tính toán:* Tỷ lệ diện tích $R = \frac{\text{Area}_{BBox}}{\text{Area}_{Image}}$.
*   *Logic:*
    *   Nếu $R < 10\%$: **REJECT** (Container quá xa/nhỏ).
    *   Nếu $R > 90\%$ hoặc BBox chạm 3/4 cạnh ảnh: **REJECT** (Quá gần/Mất góc).
*   *Action:* Nếu Pass, tiến hành Crop ảnh theo BBox $\rightarrow$ `Img_ROI`.

**Bước 2: Phân tích Quang trắc (Photometric Analysis)**
*   *Input:* `Img_ROI`.
*   *Xử lý:* Chuyển sang Grayscale $\rightarrow$ Tính Histogram.
*   *Tính toán:* $M_B$ (Median) và $M_C$ (P95 - P5).
*   *Đánh giá:* Tính $Q_B$ và $Q_C$.
*   *Logic:* Nếu $Q_B < 0.3$ (Quá tối/cháy) HOẶC $Q_C < 0.3$ (Mờ đục) $\rightarrow$ **REJECT**.

**Bước 3: Phân tích Cấu trúc (Structural Analysis)**
*   *Input:* `Img_ROI` (Grayscale).
*   *Tính toán:* Laplacian Variance $\rightarrow$ $M_S$.
*   *Đánh giá:* Tính $Q_S$.
*   *Logic:* Nếu $Q_S < 0.4$ (Mờ nhòe) $\rightarrow$ **REJECT**.

**Bước 4: Phân tích Thống kê (Statistical Analysis)**
*   *Input:* `Img_ROI` (RGB).
*   *Tính toán:* BRISQUE Score $\rightarrow$ $M_N$.
*   *Đánh giá:* Tính $Q_N$.
*   *Logic:* Nếu $Q_N < 0.2$ (Nhiễu/Vỡ nát nghiêm trọng) $\rightarrow$ **REJECT**.

---

### 5. QUYẾT ĐỊNH CUỐI CÙNG (FINAL DECISION LOGIC)

Nếu ảnh vượt qua tất cả các ngưỡng tối thiểu (Hard Thresholds) ở trên, ta tính **Chỉ số Chất lượng Tổng hợp (WQI - Weighted Quality Index)** để lưu log hoặc ranking (tùy chọn):

$$ WQI = w_1 \cdot (Q_B \cdot Q_C) + w_2 \cdot Q_S + w_3 \cdot Q_N $$

*   *Trọng số gợi ý:* $w_1 = 0.3$ (Sáng/Tương phản), $w_2 = 0.5$ (Độ nét - Quan trọng nhất), $w_3 = 0.2$ (Tự nhiên).
*   **Kết luận:** Trả về `PASS` kèm theo `WQI`.

---

### 6. YÊU CẦU CÔNG NGHỆ (IMPLEMENTATION STACK)

*   **Ngôn ngữ:** Python 3.11.14.
*   **Thư viện chính:**
    *   `OpenCV (cv2)`: Xử lý ảnh, tính Histogram, Laplacian.
    *   `OpenCV Contrib`: Chứa thuật toán BRISQUE (`cv2.quality`).
    *   `NumPy`: Tính toán Percentile ($P_{50}, P_{5}, P_{95}$) và các hàm mũ (exp).
*   **Phần cứng:** CPU (Không bắt buộc GPU cho module này vì các thuật toán đã chọn đều rất nhẹ).