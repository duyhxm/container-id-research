# TÀI LIỆU ĐẶC TẢ KỸ THUẬT

## MODULE 4: ROI RECTIFICATION & FINE-GRAINED QUALITY ASSESSMENT

**(Nắn chỉnh Vùng quan tâm & Đánh giá Chất lượng Tinh chỉnh)**

### 1. TỔNG QUAN (OVERVIEW)

*   **Mục tiêu:** Thực hiện chuẩn hóa hình học (Geometric Normalization) cho vùng văn bản và đánh giá chất lượng hiển thị chi tiết (Fine-grained Quality Assessment) để đảm bảo đầu vào tối ưu cho mô hình Nhận dạng ký tự (OCR).
*   **Đầu vào:**
    *   Ảnh gốc: $I_{src} \in \mathbb{R}^{H \times W \times C}$.
    *   Tập hợp điểm đặc trưng: $K = \{p_1, p_2, p_3, p_4\}$ tương ứng với 4 góc của vùng mã số container.
*   **Đầu ra:**
    *   Quyết định: $D \in \{ \text{PASS}, \text{REJECT} \}$.
    *   Ảnh đã nắn: $I_{rect} \in \mathbb{R}^{h \times w}$ (Grayscale).

---

### 2. CƠ SỞ LÝ LUẬN (THEORETICAL BASIS)

Module vận hành dựa trên hai nguyên lý:
1.  **Biến đổi Phối cảnh (Perspective Transformation):** Khôi phục hình dạng trực diện (Frontal View) của văn bản từ hình chiếu 2D bị biến dạng, giúp chuẩn hóa tỷ lệ ký tự.
2.  **Đánh giá Chất lượng Văn bản (Text Quality Assessment):** Tập trung vào hai thuộc tính quan trọng nhất đối với khả năng tách biên (Binarization) của OCR: Độ tương phản cục bộ (Local Contrast) và Độ sắc nét biên (Edge Sharpness).

---

### 3. ĐẶC TẢ TOÁN HỌC CÁC ĐẶC TRƯNG (MATHEMATICAL FEATURE SPECIFICATION)

#### Giai đoạn 1: Đặc trưng Hình học (Geometric Features)
*Được tính toán dựa trên tọa độ $K$ trước khi xử lý ảnh.*

**3.1. Tính hợp lệ của Tỷ lệ (Aspect Ratio Validity)**
*   **Đo lường ($M_{AR}$):** Tỷ lệ giữa chiều rộng và chiều cao dự kiến của vùng văn bản sau khi nắn.
    $$ M_{AR} = \frac{\max(\|p_{TR} - p_{TL}\|, \|p_{BR} - p_{BL}\|)}{\max(\|p_{BL} - p_{TL}\|, \|p_{BR} - p_{TR}\|)} $$
    *(Sử dụng chuẩn Euclidean L2)*.
*   **Hàm Đánh giá ($Q_{AR}$):** Sử dụng hàm chỉ thị vùng (Zone Indicator Function) dựa trên phân phối thống kê thực tế (Bimodal Distribution).
    $$
    Q_{AR}(M_{AR}) = 
    \begin{cases} 
    1 & \text{if } M_{AR} \in [2.5, 4.5] \quad (\text{Mode 1: Multi-line}) \\
    1 & \text{if } M_{AR} \in [5.0, 9.0] \quad (\text{Mode 2: Single-line}) \\
    0 & \text{otherwise}
    \end{cases}
    $$

#### Giai đoạn 2: Đặc trưng Chất lượng Hiển thị (Visual Quality Features)
*Được tính toán trên ảnh đã nắn $I_{rect}$.*

**3.2. Độ phân giải Ký tự (Character Resolution)**
*   **Đo lường ($M_{Res}$):** Chiều cao vật lý của ảnh đã nắn.
    $$ M_{Res} = \text{height}(I_{rect}) $$
*   **Hàm Đánh giá ($Q_{Res}$):** Hàm bước nhảy (Step Function) dựa trên yêu cầu tối thiểu của OCR engine.
    $$ Q_{Res}(M_{Res}) = \mathbb{I}(M_{Res} \ge 25) $$
    *(Trả về 1 nếu chiều cao $\ge 25$ pixels, ngược lại là 0)*.

**3.3. Độ tương phản Cục bộ (Local Contrast)**
*   **Đo lường ($M_C$):** Sử dụng Khoảng phân vị bền vững (Robust Range) trên Histogram của $I_{rect}$.
    $$ M_C = P_{95}(I_{rect}) - P_{5}(I_{rect}) $$
*   **Hàm Đánh giá ($Q_C$):** Sử dụng **Hàm Sigmoid** để mô hình hóa xác suất tách biệt ký tự thành công.
    $$ Q_C(M_C) = \frac{1}{1 + \exp \left( -\alpha_C \cdot (M_C - \tau_C) \right)} $$
    *   $\tau_C \approx 50$: Ngưỡng tương phản cục bộ tối thiểu (Cao hơn mức 30 của ảnh toàn cảnh).
    *   $\alpha_C \approx 0.1$: Độ dốc chuyển đổi.

**3.4. Độ nét Nét chữ (Stroke Sharpness)**
*   **Tiền xử lý:** Resize ảnh $I_{rect}$ về chiều cao chuẩn $H_{std} = 64$ pixels (giữ nguyên tỷ lệ khung hình) để chuẩn hóa mật độ biên. Gọi ảnh này là $I'_{rect}$.
*   **Đo lường ($M_S$):** Phương sai của toán tử Laplace trên ảnh chuẩn hóa.
    $$ M_S = \text{Var}(\nabla^2 I'_{rect}) $$
*   **Hàm Đánh giá ($Q_S$):** Sử dụng hàm Sigmoid dịch chuyển.
    $$ Q_S(M_S) = \frac{1}{1 + \exp \left( -\alpha_S \cdot (M_S - \tau_S) \right)} $$
    *   $\tau_S \approx 100$: Ngưỡng độ nét biên (Edge sharpness threshold).
    *   $\alpha_S \approx 0.05$: Độ dốc.

---

### 4. QUY TRÌNH XỬ LÝ & RA QUYẾT ĐỊNH (DECISION PIPELINE)

Module áp dụng cơ chế **Sàng lọc Đa tầng (Multi-stage Filtering)**:

**Tầng 1: Kiểm tra Hình học (Geometric Check)**
*   Tính $M_{AR}$.
*   Nếu $Q_{AR}(M_{AR}) == 0 \Rightarrow$ **REJECT** (Lý do: Invalid Geometry).

**Tầng 2: Nắn ảnh (Rectification)**
*   Thực hiện `warpPerspective` chỉ khi Tầng 1 PASS.
*   Thu được $I_{rect}$.

**Tầng 3: Kiểm tra Độ phân giải (Resolution Check)**
*   Tính $M_{Res}$.
*   Nếu $Q_{Res}(M_{Res}) == 0 \Rightarrow$ **REJECT** (Lý do: Low Resolution).

**Tầng 4: Kiểm tra Chất lượng (Quality Check)**
*   Tính $M_C$ và $M_S$.
*   Tính điểm chất lượng $Q_C$ và $Q_S$.
*   **Logic Quyết định:**
    *   Nếu $Q_C < 0.5$ (Tương phản quá kém) $\Rightarrow$ **REJECT**.
    *   Nếu $Q_S < 0.5$ (Quá mờ) $\Rightarrow$ **REJECT**.

**Output:**
*   Nếu vượt qua cả 4 tầng $\Rightarrow$ **PASS**. Trả về $I_{rect}$ cho Module 5.

---

### 5. YÊU CẦU CÔNG NGHỆ (TECHNOLOGY STACK)

*   **Ngôn ngữ:** Python 3.11.14.
*   **Thư viện:**
    *   `OpenCV (cv2)`: `getPerspectiveTransform`, `warpPerspective`, `Laplacian`, `resize`.
    *   `NumPy`: `percentile`, `exp`, `linalg.norm`.
*   **Lưu ý Triển khai:**
    *   Hàm tính $M_{AR}$ cần xử lý ngoại lệ chia cho 0.
    *   Hàm `resize` trong tính độ nét cần dùng nội suy `INTER_AREA` hoặc `INTER_LINEAR` tuỳ theo chiều cao ban đầu.