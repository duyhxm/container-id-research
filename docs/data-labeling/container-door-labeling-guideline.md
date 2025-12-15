# Guideline Vẽ Bounding Box cho Cửa sau Container

## Nguyên tắc Vàng (The Golden Rule)

> **Mục tiêu:** Bounding box phải là hình chữ nhật **nhỏ nhất có thể** bao trọn toàn bộ các bộ phận cấu thành nên **thực thể cửa sau**, bao gồm **hai cánh cửa chính** và các **thanh khóa (locking bars)** gắn liền trên đó.

**Lý do:**
*   **Nhỏ nhất có thể (Tightness):** Giảm thiểu vùng nền (background) bên trong box, giúp mô hình tập trung học các đặc trưng của cửa, không bị nhiễu bởi các chi tiết không liên quan.
*   **Bao trọn thực thể cửa (Completeness):** Đảm bảo rằng mọi vị trí tiềm năng của ID container đều nằm bên trong box, cung cấp một vùng tìm kiếm đáng tin cậy cho các module sau.

---

## Quy tắc 1: Trường hợp Cơ bản (Chụp chính diện, cửa đóng)

*   **Hành động:** Vẽ một hình chữ nhật duy nhất.
*   **Điểm bắt đầu:** Đặt con trỏ ở góc trên-trái của cánh cửa bên trái.
*   **Điểm kết thúc:** Kéo con trỏ đến góc dưới-phải của cánh cửa bên phải.
*   **Kiểm tra:** Bốn cạnh của bounding box phải **chạm sát** vào bốn cạnh ngoài cùng của hai cánh cửa. Bao gồm cả phần gioăng cao su (rubber seal) nếu nó là cạnh ngoài cùng của cửa.

---

## Quy tắc 2: Xử lý các Tình huống Đặc biệt

Đây là phần quan trọng nhất để đảm bảo tính nhất quán.

**2.1. Khi Cửa Mở (Door State: `open`)**

*   **Vấn đề:** Hai cánh cửa không còn là một khối chữ nhật duy nhất.
*   **Quy tắc:** **VẪN VẼ MỘT BOUNDING BOX DUY NHẤT** bao trọn cả hai cánh cửa.
*   **Lý do:** Mô hình cần học cách nhận diện "thực thể cửa sau" bất kể trạng thái. Việc luôn chỉ có một box cho một thực thể sẽ giúp mô hình học nhất quán hơn. Ngoài ra, việc vẽ một box lớn bao trọn khoảng trống ở giữa giúp đảm bảo ngữ cảnh không gian cho vùng ID, vì ID thường nằm trên cánh cửa bên phải (hoặc trái).
*   **Lưu ý:** Bounding box này sẽ chứa một khoảng nền ở giữa hai cánh cửa. Đây là một sự đánh đổi chấp nhận được.



**2.2. Khi Chụp Nghiêng (View Angle: `angled`)**

*   **Vấn đề:** Cửa bị biến dạng phối cảnh, không còn là hình chữ nhật trong không gian 2D.
*   **Quy tắc:** Bounding box luôn là một hình chữ nhật thẳng (axis-aligned). Hãy vẽ sao cho nó chạm vào **4 điểm xa nhất** của cửa sau trong không gian ảnh (điểm trên cùng, điểm dưới cùng, điểm trái cùng, điểm phải cùng).
*   **Lưu ý:** Các góc của bounding box sẽ không trùng với các góc thực tế của cửa. Sẽ có những vùng tam giác nhỏ chứa nền bên trong box. Đây là điều bình thường.



**2.3. Khi Bị Che khuất (Occlusion: `occluded`)**

*   **Vấn đề:** Một phần của cửa không thể nhìn thấy.
*   **Quy tắc:** Hãy **tưởng tượng** và vẽ bounding box cho **toàn bộ hình dạng của cửa như thể nó không bị che.**
*   **Lý do:** Chúng ta muốn mô hình học cách dự đoán toàn bộ đối tượng ngay cả khi chỉ nhìn thấy một phần của nó. Điều này giúp mô hình bền vững hơn.
*   **Hành động:** Dựa vào các phần có thể nhìn thấy, hãy ước lượng vị trí các góc bị che khuất và vẽ box cho phù hợp.

---

## Quy tắc 3: Những Gì KHÔNG được bao gồm

Để đảm bảo box đủ "chặt", hãy loại trừ các bộ phận sau:
*   **Thân vỏ container:** Không bao gồm phần khung kim loại của container bao quanh cửa.
*   **Bản lề (Hinges):** Chỉ bao gồm phần bản lề nằm trên cánh cửa, không bao gồm phần gắn vào thân container.
*   **Gầm xe và bánh xe.**
*   **Mặt đất hoặc các vật thể khác xung quanh.**

---

## Quy tắc 4: Xử lý Ảnh Chụp Màn hình (Screenshots)

*   **Vấn đề:** Bounding box nên vẽ đến đâu?
*   **Quy tắc:** **Chỉ vẽ bounding box cho cửa container.** KHÔNG được bao gồm bất kỳ phần giao diện người dùng (UI elements) nào của ứng dụng bên ngoài.
*   **Lý do:**
    1.  **Tính nhất quán:** Mô hình của em đang học cách nhận diện "cửa container", không phải "cửa container bên trong một cái điện thoại". Việc chỉ vẽ box cho cửa giúp giữ vững định nghĩa về đối tượng.
    2.  **Tính thực tế:** Trong thực tế, em có thể xây dựng một bước tiền xử lý đơn giản để tự động cắt bỏ các khung giao diện này trước khi đưa vào mô hình. Coi như việc gán nhãn này đang mô phỏng kết quả của bước tiền xử lý đó.
*   **Hành động khi gán nhãn thuộc tính:**
    *   `quality_gate`: Rất có thể sẽ là `fail`, vì ảnh chụp màn hình thường làm giảm độ phân giải và thêm nhiễu không cần thiết.
    *   `ocr_feasibility`: Đánh giá dựa trên chất lượng của vùng ID **bên trong ảnh chụp màn hình**. Nếu nó quá mờ do bị chụp lại, hãy gán là `unreadable`.

## Quy tắc 5: Xử lý Ảnh có Lớp phủ (Overlays)

*   **Vấn đề:** Lớp phủ che mất một phần cửa.
*   **Quy tắc:** Coi lớp phủ (text, bản đồ, logo) như một **vật thể che khuất (occluder)**. Áp dụng **Quy tắc 2.3** một cách nghiêm ngặt.
*   **Hành động:**
    1.  **Gán nhãn thuộc tính:**
        *   `occlusion`: `occluded`.
        *   `quality_gate`: `fail` (vì lớp phủ làm mất thông tin và có thể gây lỗi cho các module sau).
        *   `ocr_feasibility`: Nếu lớp phủ che đúng vào vùng ID, gán `unreadable`.
    2.  **Vẽ bounding box:** **Tưởng tượng** và vẽ bounding box cho toàn bộ cửa container **như thể lớp phủ không tồn tại.**

*   **Lý do:**
    1.  **Nhất quán:** Quy tắc này thống nhất cách xử lý mọi loại vật thể che khuất, dù đó là một người, một cái cây, hay một lớp text. Điều này làm cho guideline của em đơn giản và mạnh mẽ.
    2.  **Bền vững:** Dạy cho mô hình cách "nhìn xuyên qua" hoặc "suy luận" ra đối tượng hoàn chỉnh ngay cả khi có nhiễu.
    3.  **Quan trọng cho OCR:** Việc có một bounding box hoàn chỉnh của cửa sẽ giúp các module sau định vị vùng ID chính xác hơn, ngay cả khi chúng phải tự xử lý lớp phủ trên đó.