## Mô tả Kỹ thuật Hệ thống Định danh Container (ISO 6346)

Hệ thống định danh container được thiết lập dựa trên lý thuyết **Mã hóa kiểm tra lỗi (Error-detecting code)** nhằm đảm bảo tính toàn vẹn dữ liệu trong chuỗi cung ứng toàn cầu. Một mã số hợp lệ là một chuỗi $S$ có độ dài $|S| = 11$.

### 1. Cấu trúc Hình thức (Formal Structure)
Chuỗi $S$ được cấu thành từ hai tập hợp ký tự: tập chữ cái $L = \{A, B, \dots, Z\}$ và tập chữ số $N = \{0, 1, \dots, 9\}$.

$$S = \underbrace{c_1 c_2 c_3}_{\text{Owner Code}} \underbrace{c_4}_{\text{Category}} \underbrace{c_5 c_6 c_7 c_8 c_9 c_{10}}_{\text{Serial Number}} \underbrace{c_{11}}_{\text{Check Digit}}$$

* **Mã chủ sở hữu (Owner Code):** $\{c_1, c_2, c_3\} \in L^3$, được đăng ký duy nhất với BIC.
* **Ký hiệu loại (Equipment Category):** $c_4 \in \{U, J, Z\}$. Trong đó **'U'** đại diện cho container tiêu chuẩn.
* **Số sê-ri (Serial Number):** $\{c_5, \dots, c_{10}\} \in N^6$, định danh tài sản của chủ sở hữu.
* **Chữ số kiểm tra (Check Digit):** $c_{11} \in N$, dùng để xác thực 10 ký tự đứng trước.



---

### 2. Cơ sở Lý thuyết Toán học & Thuật toán Xác thực

Thuật toán xác thực dựa trên **Số học Modulo (Modular Arithmetic)** và **Trọng số lũy thừa (Exponential Weighting)** để phát hiện tối đa các lỗi nhập liệu (như gõ sai ký tự hoặc hoán vị vị trí).

#### Bước 1: Ánh xạ giá trị phi tuyến tính (Mapping)
Mỗi ký tự $c_i$ được gán một giá trị nguyên $V(c_i)$ theo bảng mã ISO. Để tăng khoảng cách Hamming và giảm tỷ lệ trùng lặp (collision), tiêu chuẩn này **loại bỏ các bội số của 11** ($11, 22, 33$).

* **Nếu $c_i \in N$:** $V(c_i) = \text{giá trị số của chính nó}$.
* **Nếu $c_i \in L$:** Ánh xạ $A \to 10, B \to 12, \dots, K \to 21$ (nhảy bậc qua số 11); $L \to 23, \dots, V \to 34$ (nhảy bậc qua số 22); $W \to 35, \dots, Z \to 38$ (nhảy bậc qua số 33).



#### Bước 2: Tính tổng trọng số (Weighted Sum)
Hệ thống sử dụng trọng số là lũy thừa cơ số 2 để đảm bảo mỗi vị trí có tầm ảnh hưởng khác nhau đến kết quả cuối cùng:
$$\Sigma = \sum_{i=1}^{10} V(c_i) \cdot 2^{i-1}$$

#### Bước 3: Phép toán Modulo 11
Chữ số kiểm tra lý thuyết $D_{calc}$ là phần dư của phép chia tổng cho 11, với một ràng buộc bổ sung để đưa kết quả về tập $N$:
$$D_{calc} = (\Sigma \pmod{11}) \pmod{10}$$
*(Lưu ý: Nếu $\Sigma \pmod{11} = 10$, quy ước $c_{11} = 0$)*.

**Điều kiện xác thực:** Mã số container được coi là hợp lệ khi và chỉ khi $V(c_{11}) = D_{calc}$.

---

### 3. Ràng buộc Logic cho Mô hình OCR (OCR Constraints)

Khi thực hiện nhận diện từ ảnh chụp cửa sau container, thuật toán hậu xử lý cần áp dụng các bộ lọc sau để tăng độ chính xác:

* **Ràng buộc Định dạng (Regex):** `^[A-Z]{3}[UJZ][0-9]{7}$`. Mọi chuỗi không khớp với pattern này cần được loại bỏ ngay lập tức để tiết kiệm tài nguyên tính toán.
* **Ma trận Nhầm lẫn (Confusion Matrix):** OCR thường nhầm lẫn giữa các ký tự có cấu trúc hình học tương đồng. Trong trường hợp Check Digit không khớp, thuật toán nên thực hiện thử sai (heuristic) trên các cặp:
    * $\{0, O, D\}$
    * $\{1, I, L\}$
    * $\{5, S\}$
    * $\{8, B\}$
* **Trọng số tin cậy (Confidence Score):** Nếu $c_{11}$ có độ tin cậy thấp, hãy dùng giá trị $D_{calc}$ vừa tính toán được từ 10 ký tự đầu để "gợi ý" lại kết quả cho mô hình.