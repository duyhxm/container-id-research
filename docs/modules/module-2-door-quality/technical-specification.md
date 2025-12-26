# TÀI LIỆU ĐẶC TẢ KỸ THUẬT

## MODULE 2: INPUT IMAGE USABILITY ASSESSMENT FRAMEWORK

**(Khung Đánh giá Khả năng sử dụng của Ảnh đầu vào)**

### 1. TỔNG QUAN (OVERVIEW)

*   **Mục tiêu:** Xây dựng một bộ lọc chất lượng tầng thấp (Low-level Quality Filter) nhằm loại bỏ các mẫu dữ liệu không đạt chuẩn trước khi đưa vào mô hình nhận dạng quang học (OCR).
*   **Đầu vào:**
    *   Không gian ảnh đầu vào: $I \in \mathbb{R}^{H \times W \times C}$.
    *   Vùng quan tâm (ROI) được định nghĩa bởi tập hợp tọa độ $\Omega_{ROI} \subset \Omega_{Image}$.
*   **Đầu ra:**
    *   Quyết định nhị phân: $D \in \{ \text{PASS}, \text{REJECT} \}$.
    *   Vector đặc trưng chất lượng: $\mathbf{q} = [Q_B, Q_C, Q_S, Q_N]^T$.

---

### 2. CƠ SỞ LÝ LUẬN (THEORETICAL BASIS)

Hệ thống được xây dựng dựa trên **Lý thuyết Thống kê Bền vững (Robust Statistics)** và **Mô hình Tâm lý Vật lý (Psychophysical Modeling)**.

1.  **Thống kê Bền vững:** Sử dụng các đại lượng thống kê thứ tự (Order Statistics) như trung vị và phân vị để ước lượng tham số phân phối, nhằm giảm thiểu sai số do nhiễu ngoại lai (outliers) trong điều kiện chụp thực tế.
2.  **Ánh xạ Chất lượng (Quality Mapping):** Các giá trị đo lường vật lý (Physical Metrics) được ánh xạ sang không gian cảm nhận (Perceptual Space) thông qua các hàm kích hoạt phi tuyến (Non-linear Activation Functions) như Gaussian và Sigmoid, mô phỏng phản ứng bão hòa của Hệ thống thị giác người (HVS).

---

### 3. ĐẶC TẢ TOÁN HỌC CÁC ĐẶC TRƯNG (MATHEMATICAL FEATURE SPECIFICATION)

#### 3.0. Kiểm tra Hình học (Geometric Pre-check) - *Spatial Constraints*

Trước khi phân tích các đặc trưng quang học, hệ thống thực hiện kiểm tra ràng buộc không gian để đảm bảo vùng quan tâm (ROI) có kích thước và vị trí hợp lệ.

*   **Đo lường Tỷ lệ Diện tích ($R_{geo}$):**
    $$ R_{geo} = \frac{A_{BBox}}{A_{Image}} = \frac{(x_{max} - x_{min}) \cdot (y_{max} - y_{min})}{W \cdot H} $$
    Trong đó:
    *   $(x_{min}, y_{min}, x_{max}, y_{max})$: Tọa độ bounding box từ Module 1.
    *   $(W, H)$: Kích thước ảnh gốc.

*   **Điều kiện Chấp nhận:**
    $$ 0.10 \leq R_{geo} \leq 0.90 $$

*   **Kiểm tra Chạm Biên (Edge Touching):**
    Bounding box bị từ chối nếu chạm vào **3 trong 4 cạnh** của ảnh, cho thấy container bị cắt xén (cropped) hoặc chụp quá gần.

*   **Biện luận:**
    *   $R_{geo} < 0.10$: Container quá xa, độ phân giải vùng ID không đủ cho OCR.
    *   $R_{geo} > 0.90$: Container chiếm toàn bộ frame, có nguy cơ mất góc hoặc vùng ID bị cắt.
    *   Edge touching: Vi phạm giả định "container nằm trọn trong frame".

Giả sử $I_{ROI}$ là ảnh xám (Luminance channel) của vùng quan tâm đã vượt qua kiểm tra hình học, được coi là một biến ngẫu nhiên rời rạc với hàm mật độ xác suất (PDF) là $p(x)$, trong đó $x \in [0, 255]$.

#### 3.1. Độ sáng (Brightness) - *Central Tendency Estimation*
*   **Đo lường ($M_B$):** Sử dụng Trung vị (Median) để ước lượng xu hướng trung tâm của phân phối độ sáng, đảm bảo tính bền vững trước nhiễu lệch (skewed noise).
    $$ M_B = \text{median}(I_{ROI}) = F^{-1}(0.5) $$
    *(Với $F^{-1}$ là hàm phân phối tích lũy nghịch đảo - Inverse CDF)*.
*   **Hàm Chất lượng ($Q_B$):** Sử dụng **Hàm hạt nhân Gaussian (Gaussian Kernel)** để mô hình hóa vùng chấp nhận quanh giá trị tối ưu $\mu_{opt}$.
    $$ Q_B(M_B) = \exp \left( - \frac{(M_B - \mu_{opt})^2}{2 \sigma^2} \right) $$
    *   $\mu_{opt} \approx 100$: Giá trị kỳ vọng tối ưu (thực nghiệm).
    *   $\sigma \approx 65$: Độ lệch chuẩn cho phép (Bandwidth).

#### 3.2. Độ tương phản (Contrast) - *Dispersion Estimation*
*   **Đo lường ($M_C$):** Sử dụng Khoảng phân vị bền vững (Robust Inter-percentile Range) để ước lượng độ phân tán năng lượng.
    $$ M_C = P_{95}(I_{ROI}) - P_{5}(I_{ROI}) $$
*   **Hàm Chất lượng ($Q_C$):** Sử dụng **Hàm Logistic Tổng quát (Generalized Logistic Function)** để mô hình hóa ngưỡng chuyển đổi mềm (Soft Thresholding).
    $$ Q_C(M_C) = \frac{1}{1 + \exp \left( -\alpha \cdot (M_C - \tau_C) \right)} $$
    *   $\tau_C \approx 50$: Điểm uốn (Inflection point), ngưỡng tương phản tối thiểu.
    *   $\alpha \approx 0.1$: Hệ số độ dốc (Steepness coefficient).

#### 3.3. Độ nét (Sharpness) - *High-Frequency Energy*
*   **Đo lường ($M_S$):** Ước lượng năng lượng của các thành phần tần số cao thông qua phương sai của toán tử Laplace (Laplacian Operator $\nabla^2$).
    $$ M_S = \text{Var} \left( \nabla^2 I_{ROI} \right) = \mathbb{E}[(\nabla^2 I_{ROI})^2] - (\mathbb{E}[\nabla^2 I_{ROI}])^2 $$
*   **Hàm Chất lượng ($Q_S$):** Sử dụng hàm bão hòa tuyến tính từng khúc (Piecewise Linear Saturation).
    $$ Q_S(M_S) = \min \left( \frac{M_S}{\lambda_S}, 1.0 \right) $$
    *   $\lambda_S \approx 100$: Ngưỡng bão hòa độ nét.

#### 3.4. Độ tự nhiên (Naturalness) - *NSS Deviation*
*   **Đo lường ($M_N$):** Sử dụng khoảng cách thống kê giữa ảnh đầu vào và mô hình Thống kê Cảnh tự nhiên (NSS) thông qua thuật toán BRISQUE.
    $$ M_N = \text{BRISQUE}(I_{ROI}) \in [0, 100] $$
*   **Hàm Chất lượng ($Q_N$):** Sử dụng hàm suy giảm tuyến tính (Linear Decay) hoặc hàm mũ nghịch đảo.
    $$ Q_N(M_N) = \max \left( 0, 1 - \frac{M_N}{100} \right) $$

---

### 4. CHIẾN LƯỢC RA QUYẾT ĐỊNH (DECISION STRATEGY)

Hệ thống sử dụng cơ chế **Sàng lọc Tuần tự (Cascade Filtering)** kết hợp với **Đánh giá Tổng hợp (Weighted Fusion)**.

#### 4.1. Điều kiện Tiên quyết (Prerequisite Constraints) - *Hard Thresholding*
Một ảnh $I$ bị từ chối ($D = \text{REJECT}$) nếu vi phạm bất kỳ điều kiện biên nào sau đây:

1.  **Ràng buộc Hình học:** $R_{geo} \notin [0.1, 0.9]$.
2.  **Ràng buộc Quang trắc:** $(Q_B < \epsilon_B) \lor (Q_C < \epsilon_C)$.
3.  **Ràng buộc Cấu trúc:** $Q_S < \epsilon_S$.
4.  **Ràng buộc Thống kê:** $Q_N < \epsilon_N$.

*(Với $\epsilon$ là các ngưỡng tối thiểu - minimum acceptable scores).*

#### 4.2. Chiến lược Tổng hợp Điểm số (Score Aggregation Strategy)

Thay vì sử dụng mô hình cộng tính (Additive Model) vốn cho phép sự bù trừ giữa các đặc trưng (compensatory nature), chúng tôi đề xuất sử dụng **Mô hình Trung bình nhân có trọng số (Weighted Geometric Mean)**. Mô hình này đảm bảo tính chất **phủ quyết (veto property)** – một đặc tính sống còn đối với hệ thống OCR.

Công thức chỉ số chất lượng tổng hợp (WQI) được định nghĩa như sau:

$$ WQI = \prod_{i \in \{B, C, S, N\}} (Q_i + \epsilon)^{w_i} $$

Trong đó:
*   $Q_i$: Các điểm chất lượng thành phần ($Q_B, Q_C, Q_S, Q_N$).
*   $w_i$: Trọng số tầm quan trọng, với $\sum w_i = 1$.
*   $\epsilon$: Một hằng số nhỏ (ví dụ $10^{-6}$) để tránh lỗi logarit của 0 và đảm bảo tính ổn định số học.

**Biện luận lựa chọn:**
Mô hình này hoạt động như một bộ lọc **Soft-AND**. Giá trị WQI sẽ bị kéo xuống mức thấp nếu **bất kỳ** thành phần $Q_i$ nào có giá trị thấp, phản ánh đúng bản chất của chuỗi xử lý OCR: sự thất bại ở bất kỳ khâu nào (thiếu sáng, mất nét, hay nhiễu) đều dẫn đến thất bại của toàn bộ hệ thống.

**Gợi ý bộ trọng số ($w$):**
Dựa trên mức độ ảnh hưởng tới OCR:
*   $w_S$ (Sharpness) = 0.4 (Quan trọng nhất - Mờ là không đọc được).
*   $w_C$ (Contrast) = 0.3 (Quan trọng nhì - Tách biên).
*   $w_B$ (Brightness) = 0.2 (Có thể dùng thuật toán cân bằng sáng để cứu).
*   $w_N$ (Naturalness) = 0.1 (Ít quan trọng nhất).

---

### 5. YÊU CẦU CÔNG NGHỆ (TECHNOLOGY STACK)

*   **Ngôn ngữ:** Python 3.11.14.
*   **Thư viện tính toán:**
    *   `NumPy`: Thực hiện các phép toán đại số tuyến tính và thống kê mô tả.
    *   `OpenCV (Contrib)`: Cung cấp các toán tử xử lý ảnh (Laplacian) và mô hình NSS (BRISQUE).
    *   `SciPy` (Tùy chọn): Nếu cần tính Entropy hoặc khớp các phân phối phức tạp hơn.