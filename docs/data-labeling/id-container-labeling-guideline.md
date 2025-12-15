# Guideline Gán nhãn Vùng ID Container (Keypoints)

**Mục tiêu:** Xác định chính xác 4 điểm góc của vùng chứa mã định danh (ID) container để phục vụ cho bài toán nắn chỉnh ảnh (Perspective Transform) và nhận dạng ký tự (OCR).

**Công cụ sử dụng:** Công cụ `Points` (Keypoints) trên CVAT.
**Số lượng điểm:** Bắt buộc 4 điểm cho mỗi vùng ID.

---

## Nguyên tắc Vàng (The Golden Rule)

> **Mục tiêu:** Xác định 4 điểm tạo thành một tứ giác bao trọn **toàn bộ chuỗi ký tự định danh**, bao gồm đủ 3 thành phần: **Mã chủ sở hữu (4 chữ cái) + Số sê-ri (6 số) + Số kiểm tra (1 số)**.
>
> **Thứ tự chấm điểm:** BẮT BUỘC tuân thủ thứ tự chiều kim đồng hồ, bắt đầu từ góc trên-trái:
> 1.  **Top-Left (TL):** Góc trên cùng bên trái.
> 2.  **Top-Right (TR):** Góc trên cùng bên phải.
> 3.  **Bottom-Right (BR):** Góc dưới cùng bên phải.
> 4.  **Bottom-Left (BL):** Góc dưới cùng bên trái.

**Lý do:** Thứ tự điểm nhất quán là yếu tố sống còn để thuật toán nắn ảnh (Perspective Warp) hoạt động đúng. Nếu chấm lộn xộn, ảnh sau khi nắn sẽ bị lật ngược hoặc xoắn lại.

---

## Quy tắc 1: Xác định Phạm vi (Scope)

*   **Bao gồm:**
    *   Toàn bộ 11 ký tự của ID (Ví dụ: `MSKU 123456 7`).
    *   Nếu ID được in thành một dòng (phổ biến nhất), 4 điểm sẽ bao quanh dòng đó.
    *   Nếu số kiểm tra (check digit) nằm trong một ô vuông hoặc cách xa một chút, vẫn phải mở rộng vùng chọn để bao lấy nó.

*   **Loại trừ (KHÔNG bao gồm):**
    *   Mã loại kích thước (ISO Code) thường nằm ngay dưới ID (Ví dụ: `45G1`, `22T6`).
    *   Thông tin trọng lượng (MAX GROSS, TARE).
    *   Logo hãng tàu hoặc các nhãn dán quảng cáo khác.

---

## Quy tắc 2: Độ chặt (Tightness)

*   **Hành động:** Các điểm chấm phải nằm sát mép ngoài của các ký tự ở 4 góc.
*   **Lưu ý:** Không cần chừa lề (margin) quá rộng. Thuật toán cắt ảnh sau này có thể tự động thêm padding nếu cần. Việc chấm sát giúp mô hình học được vị trí chính xác của văn bản.

---

## Quy tắc 3: Xử lý các Tình huống Đặc biệt

**3.1. ID nằm dọc (Vertical ID)**
*   **Mô tả:** Đôi khi ID được in dọc từ trên xuống dưới.
*   **Hành động:** Vẫn chấm 4 điểm bao quanh cột văn bản đó.
*   **Thứ tự điểm:** Vẫn tuân thủ logic của văn bản:
    *   Điểm 1 (TL): Góc trên bên trái của chữ cái đầu tiên.
    *   Điểm 2 (TR): Góc trên bên phải của chữ cái đầu tiên.
    *   Điểm 3 (BR): Góc dưới bên phải của số cuối cùng.
    *   Điểm 4 (BL): Góc dưới bên trái của số cuối cùng.

**3.2. ID bị chia dòng (Split ID)**
*   **Mô tả:** Mã hãng tàu (`MSKU`) nằm dòng trên, số sê-ri (`123456 7`) nằm dòng dưới.
*   **Hành động:**
    *   **Ưu tiên:** Nếu có một mã ID khác nằm ngang trọn vẹn ở vị trí khác trên cửa, hãy ưu tiên gán nhãn cho mã nằm ngang đó.
    *   **Bắt buộc:** Nếu chỉ có mã bị chia dòng, hãy chấm 4 điểm bao trọn **cả hai dòng**. Vùng chọn sẽ bao gồm cả khoảng trống giữa hai dòng.

**3.3. ID bị cong hoặc nghiêng (Distortion)**
*   **Mô tả:** Do bề mặt container lồi lõm (dạng sóng) hoặc góc chụp nghiêng.
*   **Hành động:** Chấm 4 điểm sao cho chúng tạo thành mặt phẳng giả lập chứa văn bản. Đừng lo lắng nếu cạnh nối các điểm cắt qua các đường sóng của container.

**3.4. ID bị che khuất hoặc mờ (Occlusion/Blur)**
*   **Hành động:**
    *   Nếu vẫn có thể đoán được vị trí các góc (ví dụ chỉ mất 1 góc nhỏ), hãy ước lượng và chấm điểm theo quy tắc "Tưởng tượng".
    *   Nếu ID bị che khuất quá nhiều hoặc mờ đến mức không thể đọc (`ocr_feasibility` = `unreadable`), **KHÔNG GÁN NHÃN** vùng ID cho ảnh này (hoặc gán nhưng đánh dấu là khó - tùy thuộc vào chiến lược training, nhưng với MVP thì nên bỏ qua để tránh làm nhiễu model).
