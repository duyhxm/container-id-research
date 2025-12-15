# Guideline Gán nhãn Thuộc tính Ảnh Cửa sau Container

**Mục tiêu:** Phân loại mỗi ảnh theo một bộ thuộc tính nhất quán để phục vụ cho việc phân tích dữ liệu và đánh giá tính bền vững (robustness) của mô hình.

**Nguyên tắc vàng:** Khi phân vân, hãy chọn giá trị đại diện cho trường hợp "thử thách" hơn (ví dụ: `angled`, `bad_light`, `blurry`).

---

## 1 Yếu tố Hình học (Geometric Factors)

*   **Câu hỏi:** Camera ở đâu so với container và cửa ở trạng thái nào?

*   **Thuộc tính:** `view_angle` (Góc nhìn)
    *   `frontal`: Gán khi các cạnh dọc của cửa container gần như song song với các cạnh của khung ảnh.
    *   `angled`: Gán khi các cạnh dọc của cửa hội tụ rõ rệt, và/hoặc có thể nhìn thấy mặt bên của container.

*   **Thuộc tính:** `door_state` (Trạng thái cửa)
    *   `closed`: Gán khi cả hai cánh cửa đều ở vị trí đóng kín, áp sát vào nhau.
    *   `open`: Gán khi một hoặc cả hai cánh cửa không ở vị trí đóng kín.

## 2 Yếu tố Chiếu sáng (Illumination Factors)

*   **Câu hỏi:** Điều kiện ánh sáng tại hiện trường có lý tưởng không?

*   **Thuộc tính:** `lighting` (Ánh sáng)
    *   `good_light`: Gán khi toàn bộ khu vực cửa container được chiếu sáng tương đối đều, không có vùng nào quá tối hoặc quá sáng đến mức mất chi tiết cấu trúc.
    *   `bad_light`: Gán khi **BẤT KỲ** điều kiện nào sau đây xảy ra:
        *   **Tối/Thiếu sáng:** Một phần lớn của cửa chìm trong bóng tối, khó phân biệt rõ cấu trúc.
        *   **Cháy sáng/Lóa:** Có các vùng trắng xóa trên bề mặt cửa do phản xạ ánh sáng mạnh (mặt trời, đèn pha), làm mất chi tiết.
        *   **Ngược sáng:** Nguồn sáng chính nằm phía sau container, khiến phần cửa nhìn về phía camera bị tối đen.

## 3 Yếu tố Toàn vẹn của Đối tượng (Object Integrity)

*   **Câu hỏi:** Bản thân cửa container có nguyên vẹn và sạch sẽ không?

*   **Thuộc tính:** `occlusion` (Sự che khuất)
    *   `not_occluded`: Gán khi cửa không bị che bởi bất kỳ vật thể bên ngoài nào.
    *   `occluded`: Gán **CHỈ KHI** có một vật thể khác (người, xe, hàng hóa, cột điện,...) nằm giữa camera và cửa, che mất một phần của cửa.

*   **Thuộc tính:** `surface` (Bề mặt)
    *   `clean`: Gán khi bề mặt chủ yếu là màu sơn gốc, có thể có bụi bẩn nhẹ nhưng không đáng kể.
    *   `not_clean`: Gán khi bề mặt có nhiều bùn đất, vết bẩn lớn, gỉ sét lan rộng, hoặc hình vẽ graffiti làm thay đổi đáng kể diện mạo.

## 4 Yếu tố Chất lượng Ghi hình (Capture Quality)

*   **Câu hỏi:** Bức ảnh được chụp có rõ nét không?

*   **Thuộc tính:** `sharpness` (Độ nét)
    *   `sharp`: Gán khi các cạnh viền của cửa, tay nắm, hoặc các con số (nếu có) hiện ra sắc nét. Có thể đọc được chữ nếu phóng to.
    *   `blurry`: Gán khi các cạnh viền bị nhòe, không sắc nét. Áp dụng cho cả hai trường hợp mờ do mất nét (defocus blur) và mờ do chuyển động (motion blur).
	
## 5 Yếu tố Tổng hợp (Meta-Factor)

*   **Định nghĩa:**
    > Thuộc tính này không mô tả một đặc tính vật lý cụ thể của ảnh, mà là một **kết luận tổng hợp** dựa trên tất cả các yếu tố đã phân tích. Nó đóng vai trò như một cổng kiểm soát chất lượng giả lập (simulated quality gate).

*   **Mô tả chi tiết:**
    Sau khi đã gán nhãn cho 4 yếu tố trên, hãy trả lời câu hỏi cuối cùng: "Nếu tôi là người xây dựng một hệ thống hoàn hảo, tôi có muốn người dùng gửi lên một bức ảnh tệ như thế này không?". Thuộc tính này giúp phân loại các trường hợp cực đoan để phục vụ cho việc phân tích hiệu năng sâu hơn sau này.

*   **Thuộc tính:** `quality_gate` (Cổng chất lượng)
    *   `pass`: Gán cho tất cả các ảnh không thuộc trường hợp `fail`. Đây là giá trị mặc định. Một ảnh có thể có một vài thuộc tính xấu (ví dụ: hơi nghiêng, hơi mờ) nhưng vẫn được `pass` nếu tổng thể vẫn có thể sử dụng được, miễn là đối tượng vẫn có thể được nhận diện khi nhìn bởi mắt người.

    *   `fail`: Gán khi ảnh gặp phải **BẤT KỲ** vấn đề nghiêm trọng nào sau đây, khiến cho việc nhận diện gần như bất khả thi hoặc không đáng tin cậy:
        *   **Quá mờ (`blurry`):** Mờ đến mức không thể phân biệt được cấu trúc cơ bản của cửa (ví dụ: không thấy đường viền giữa 2 cánh cửa).
        *   **Ánh sáng quá tệ (`bad_light`):** Tối đen đến mức gần như toàn bộ cửa chìm trong bóng tối, hoặc cháy sáng đến mức mất hết chi tiết.
        *   **Bị che khuất quá nhiều (`occluded`):** Hơn 50% diện tích của cửa bị che bởi vật thể khác.
        *   **Góc chụp quá hẹp (`angled`):** Chụp quá nghiêng khiến mặt phẳng chính của cửa gần như không thể nhìn thấy.
        *   **Sai đối tượng:** Bức ảnh không chứa cửa sau container.
		
## 6 Yếu tố Đặc thù Tác vụ (Task-Specific Factor)

*   **Định nghĩa:**
    > Thuộc tính này không đánh giá chất lượng toàn bộ bức ảnh, mà tập trung đánh giá **khả năng thành công của tác vụ cuối cùng (trích xuất ID)** dựa trên chất lượng của vùng thông tin quan trọng.

*   **Mô tả chi tiết:**
    Sau khi đã xác định được cửa container, hãy tập trung vào vùng chứa ID (số container, mã ISO, v.v.). Dựa vào khả năng đọc của mắt người, hãy đưa ra phán đoán về tính khả thi của việc OCR. Thuộc tính này sẽ là "sự thật" (ground truth) để đo lường hiệu quả của toàn bộ pipeline.

*   **Thuộc tính:** `ocr_feasibility` (Khả năng OCR)
    *   **Quy tắc quyết định:** Hãy tự hỏi: "Nếu tôi đưa vùng ảnh chứa ID này cho một người khác, họ có thể đọc được nó một cách chắc chắn không?".

    *   `readable`: Gán khi các ký tự của ID container **rõ ràng, sắc nét và có thể đọc được bằng mắt thường** mà không cần phải đoán.
        *   *Ví dụ:* Ảnh chụp gần, lấy nét đúng vào vùng ID, không bị lóa.

    *   `unreadable`: Gán khi **BẤT KỲ** điều kiện nào sau đây xảy ra với vùng chứa ID:
        *   **Quá mờ:** Vùng ID bị mờ do mất nét hoặc chuyển động, khiến các ký tự bị nhòe vào nhau.
        *   **Quá xa:** Ảnh chụp từ xa khiến vùng ID có độ phân giải quá thấp, các ký tự chỉ là những đốm pixel.
        *   **Bị lóa/tối:** Vùng ID bị cháy sáng hoặc chìm trong bóng tối.
        *   **Bị che khuất/bẩn:** Vùng ID bị che bởi vật khác hoặc bị bùn đất che lấp.
