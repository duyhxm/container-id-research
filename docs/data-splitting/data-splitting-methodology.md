# PHƯƠNG PHÁP PHÂN TẦNG VÀ CHIA TẬP DỮ LIỆU (DATA STRATIFICATION METHODOLOGY)

## 1. Đặt vấn đề (Problem Formulation)

Trong bài toán nhận diện và trích xuất thông tin từ cửa sau container, bộ dữ liệu thu thập được ($D$) có đặc điểm là kích thước mẫu nhỏ ($N \approx 500$) và sự mất cân bằng dữ liệu nghiêm trọng (Severe Class Imbalance).

Cụ thể, các biến thể "lý tưởng" (ánh sáng tốt, góc chụp thẳng, sạch sẽ) chiếm đa số (>90%), trong khi các biến thể "thử thách" (thiếu sáng, góc nghiêng, bị che khuất) nằm ở phần đuôi dài (long-tail) của phân phối.

Nếu áp dụng phương pháp **Lấy mẫu ngẫu nhiên (Random Sampling)** truyền thống, xác suất để các mẫu hiếm (Rare samples) xuất hiện trong tập Kiểm thử (Test set) là rất thấp, dẫn đến việc đánh giá hiệu năng mô hình bị sai lệch (Biased Evaluation). Ngược lại, nếu áp dụng phương pháp **Phân tầng chi tiết (Detailed Stratification)** dựa trên tích Đề-các của tất cả thuộc tính, ta gặp phải vấn đề **Dữ liệu thưa (Data Sparsity)** và các nhóm đơn lẻ (Singletons), khiến thuật toán chia dữ liệu không thể thực hiện.

## 2. Phương pháp Đề xuất (Proposed Methodology)

Để giải quyết vấn đề trên, chúng tôi áp dụng phương pháp **Phân tầng dựa trên Biến đổi Tập lũy thừa Nhãn có Gộp nhóm (Label Powerset Stratification with Rare-Class Aggregation)**.

Phương pháp này ánh xạ không gian thuộc tính đa chiều của mỗi mẫu dữ liệu về một nhãn phân tầng duy nhất (Stratification Label) thông qua một hàm ưu tiên (Priority Function).

### 2.1. Mô hình hóa Toán học

Gọi $x_i$ là mẫu dữ liệu thứ $i$ trong tập dữ liệu $D$. Mỗi mẫu được gán một vectơ thuộc tính $\mathbf{a}_i$:
$$ \mathbf{a}_i = [a_{i, \text{light}}, a_{i, \text{angle}}, a_{i, \text{occ}}, a_{i, \text{surf}}, a_{i, \text{sharp}}] $$

Chúng tôi định nghĩa hai tập hợp các giá trị thuộc tính cần ưu tiên bảo tồn (Priority Sets):

1.  **Tập hợp Môi trường Khắc nghiệt ($R_{env}$):** Bao gồm các yếu tố ngoại cảnh tác động tiêu cực lên ảnh.
    $$ R_{env} = \{\text{bad\_light}, \text{occluded}, \text{not\_clean}\} $$
2.  **Tập hợp Hình học & Cảm biến Phức tạp ($R_{geo}$):** Bao gồm các yếu tố do thao tác chụp hoặc thiết bị.
    $$ R_{geo} = \{\text{frontal}, \text{blurry}\} $$
    *(Lưu ý: `frontal` được đưa vào nhóm này do sự khan hiếm thực tế trong tập dữ liệu, dù về mặt thị giác máy tính nó là trường hợp lý tưởng).*

### 2.2. Hàm Ánh xạ Phân tầng ($\Phi$)

Chúng tôi xây dựng hàm ánh xạ $\Phi: \mathbf{a}_i \rightarrow S$ để phân loại mỗi mẫu vào một trong ba nhóm phân tầng $S = \{s_{hard}, s_{tricky}, s_{common}\}$. Hàm này hoạt động theo cơ chế **Xếp hạng Ưu tiên (Priority Ranking)** để giải quyết xung đột đa nhãn:

$$
\Phi(\mathbf{a}_i) = 
\begin{cases} 
s_{hard} & \text{nếu } \exists j : a_{i,j} \in R_{env} \\
s_{tricky} & \text{nếu } (\nexists j : a_{i,j} \in R_{env}) \land (\exists k : a_{i,k} \in R_{geo}) \\
s_{common} & \text{trường hợp còn lại}
\end{cases}
$$

**Giải thích cơ chế:**
*   **$s_{hard}$ (Hard & Rare):** Nhóm ưu tiên cao nhất. Bất kỳ ảnh nào chứa ít nhất một lỗi môi trường (tối, bẩn, che) đều được gán vào đây để đảm bảo chúng được chia đều cho tập Train và Test.
*   **$s_{tricky}$ (Tricky):** Nhóm ưu tiên thứ hai. Chứa các ảnh sạch về môi trường nhưng có vấn đề về độ nét hoặc là các mẫu `frontal` hiếm gặp.
*   **$s_{common}$ (Common):** Nhóm còn lại, chứa các mẫu phổ biến (thường là `angled` + `good_light`).

### 2.3. Lược bỏ Thuộc tính (Feature Selection Logic)

Trong quá trình xây dựng hàm $\Phi$, chúng tôi chủ động lược bỏ các thuộc tính mang tính chất **Hệ quả (Consequential Attributes)** như `quality_gate` và `ocr_feasibility`.
*   **Lý do:** Các thuộc tính này có tính phụ thuộc cao (High Correlation) vào các thuộc tính nguyên nhân (ánh sáng, độ nét...). Việc đưa chúng vào phân tầng sẽ gây ra hiện tượng đa cộng tuyến và làm loãng không gian mẫu không cần thiết.
*   Thuộc tính `door_state` cũng được lược bỏ để giảm chiều dữ liệu (Dimensionality Reduction), do tác động của nó lên bài toán phát hiện đối tượng là không đáng kể so với các yếu tố môi trường.

## 3. Quy trình Thực hiện (Execution Workflow)

Dữ liệu được chia thành 3 tập: **Train (70%)**, **Validation (15%)**, và **Test (15%)** sử dụng kỹ thuật **Stratified Sampling** dựa trên nhãn $s$ sinh ra từ hàm $\Phi$.

### 3.1. Xử lý Dữ liệu Đơn lẻ (Handling Singletons)

Đối với các trường hợp cực hiếm (Singleton) - nơi chỉ tồn tại duy nhất một mẫu dữ liệu thỏa mãn một tổ hợp thuộc tính đặc biệt (ví dụ: `frontal` + `bad_light`), chúng tôi áp dụng chiến lược **Nhân bản có kiểm soát (Controlled Duplication)** để đảm bảo sự hiện diện của các trường hợp này trong cả tập huấn luyện và kiểm thử mà không gây rò rỉ dữ liệu (Data Leakage).

Quy trình cụ thể như sau:

1.  **Phân bổ Mẫu gốc (Original Sample Allocation):** Mẫu ảnh gốc được ưu tiên đưa vào tập **Kiểm thử (Test Set)**. Điều này đảm bảo việc đánh giá hiệu năng mô hình được thực hiện trên dữ liệu thực tế, không bị biến đổi nhân tạo.

2.  **Sinh mẫu Huấn luyện (Training Sample Generation):** Một bản sao đã qua tăng cường (Augmented copy) của mẫu gốc được đưa vào tập **Huấn luyện (Train Set)**.

3.  **Chiến lược Tăng cường Bảo toàn Ngữ nghĩa (Semantic-Preserving Augmentation):**
    Do đặc thù của bài toán bao gồm việc nhận diện và định vị vùng văn bản (Container ID), các phép biến đổi hình học làm đảo lộn trật tự ký tự (như Lật ngang - Horizontal Flip) bị **nghiêm cấm**. Thay vào đó, chúng tôi sử dụng các kỹ thuật tăng cường an toàn:
    *   **Biến đổi Hình học Nhẹ (Minor Geometric Transformations):** Xoay ảnh (Rotation) với biên độ nhỏ ($\pm 5^\circ$) hoặc Kéo dãn (Shear/Scale) để giả lập sự thay đổi góc chụp mà không làm biến dạng ký tự quá mức.
    *   **Biến đổi Môi trường (Environmental Transformations):** Thay đổi độ sáng/tương phản (Random Brightness/Contrast) hoặc thêm nhiễu mờ chuyển động (Motion Blur) để giả lập điều kiện ánh sáng và thao tác chụp thực tế.

Chiến lược này giúp mô hình học được các đặc trưng hiếm (Rare Features) thông qua bản sao biến thể, trong khi vẫn giữ lại bản gốc nguyên vẹn để kiểm chứng năng lực thực tế của mô hình.

### 3.2. Xử lý Dữ liệu Module-Specific

Quy trình chia dữ liệu đảm bảo tính đồng bộ (Synchronization) giữa Module 1 (Detection) và Module 3 (Localization). Tuy nhiên, có sự lọc dữ liệu đặc thù:
*   **Module 1:** Sử dụng toàn bộ dữ liệu.
*   **Module 3:** Các mẫu trong tập Train có thuộc tính `ocr_feasibility` là `unknown` hoặc `unreadable` sẽ bị loại bỏ để tránh nhiễu (Noise). Các mẫu tương tự trong tập Test được giữ lại để đánh giá khả năng xử lý ngoại lệ (False Positive Rejection).

## 4. Kết luận

Phương pháp phân tầng này đảm bảo rằng tập kiểm thử (Test Set) là một đại diện thu nhỏ và công bằng của toàn bộ không gian dữ liệu, bao gồm cả các trường hợp biên (Edge cases). Điều này cho phép chúng tôi đánh giá tính bền vững (Robustness) của mô hình một cách định lượng và chính xác, thay vì chỉ dựa trên các chỉ số trung bình bị sai lệch bởi dữ liệu phổ biến.
