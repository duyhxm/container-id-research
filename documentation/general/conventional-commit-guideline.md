# QUY CHUẨN CONVENTIONAL COMMITS HỢP NHẤT

**(Unified Conventional Commits Standard - UCCS)**

**Phiên bản:** 1.0.0
**Trạng thái:** BAN HÀNH
**Phạm vi áp dụng:** Phát triển phần mềm, Quản trị Dữ liệu (Data Engineering), Trí tuệ nhân tạo (AI/ML), và Quản lý Tài liệu kỹ thuật (Technical Writing).

-----

## 1. TRIẾT LÝ CỐT LÕI (CORE PHILOSOPHY)

Lịch sử Git không phải là một kho lưu trữ các bản sao lưu (backups). Nó là một **Cơ sở dữ liệu ngữ nghĩa theo thời gian** (Semantic Temporal Database). Mỗi commit là một **đơn vị thay đổi nguyên tử** (atomic unit of change) có cấu trúc, cho phép con người và máy móc hiểu được sự tiến hóa của dự án mà không cần đọc mã nguồn.

Phương trình của một Commit hợp lệ:

$$Commit = Type(Scope) + Description + [Body]$$

-----

## 2. HỆ THỐNG PHÂN LOẠI (TAXONOMY)

Chúng ta sử dụng mô hình **"Ba Lăng Kính" (The Tri-Lens Model)** để phân loại tính chất của sự thay đổi. Mọi thay đổi đều phải rơi vào một trong ba nhóm tác động sau:

### Nhóm I: Tác động Ngoại vi (External Impact)

*Tác động trực tiếp đến người dùng cuối hoặc kết quả đầu ra của hệ thống.*

| Type | Ký hiệu | Định nghĩa Chính xác |
| :--- | :--- | :--- |
| **Feature** | `feat` | Bổ sung một chức năng mới, một nội dung mới hoặc một khả năng mới mà hệ thống chưa từng có. |
| **Fix** | `fix` | Khắc phục một hành vi sai lệch, một thông tin không chính xác, hoặc một lỗi trong logic vận hành. |

### Nhóm II: Tác động Nội tại (Internal Structural)

*Tác động đến chất lượng mã nguồn/dữ liệu nhưng bảo toàn hành vi đầu ra.*

| Type | Ký hiệu | Định nghĩa Chính xác |
| :--- | :--- | :--- |
| **Refactor**| `refactor`| Thay đổi cấu trúc nội bộ nhằm cải thiện độ phức tạp, hiệu năng hoặc khả năng đọc hiểu, nhưng **không** thay đổi hành vi bên ngoài. |
| **Style** | `style` | Các thay đổi về hình thức trình bày (formatting, spacing, indentation) hoàn toàn không ảnh hưởng đến logic thực thi hay ngữ nghĩa nội dung. |

### Nhóm III: Tác động Hỗ trợ (Meta & Supporting)

*Tác động đến hệ sinh thái, quy trình và tài liệu mô tả.*

| Type | Ký hiệu | Định nghĩa Chính xác |
| :--- | :--- | :--- |
| **Documentation**| `docs` | Thay đổi các tài liệu mô tả về hệ thống (README, Wiki, Comments) nhằm làm rõ nghĩa, không thay đổi logic hệ thống. |
| **Chore** | `chore` | Các tác vụ bảo trì định kỳ, cập nhật công cụ, cấu hình môi trường, dependencies không ảnh hưởng đến mã nguồn sản phẩm. |
| **Revert** | `revert` | Hoàn tác lại một commit trước đó. |

-----

## 3. MA TRẬN ÁNH XẠ NGỮ CẢNH (CONTEXT MAPPING MATRIX)

Bảng dưới đây là quy chuẩn để áp dụng các `type` vào từng lĩnh vực cụ thể, đảm bảo tính nhất quán trên toàn bộ hệ thống kỹ thuật.

| Type | 💻 Software Engineering | 📄 Documentation (Wiki) | 🗄️ Data Engineering | 🧠 AI / Machine Learning |
| :--- | :--- | :--- | :--- | :--- |
| `feat` | Thêm tính năng, API, màn hình mới. | Thêm trang mới, chương mới. | Thêm bảng (table), nguồn dữ liệu (source), pipeline mới. | Thêm model mới, kỹ thuật training mới, feature mới. |
| `fix` | Sửa bug, lỗi logic, lỗ hổng bảo mật. | Sửa thông tin sai, link chết (404), lỗi chính tả nghiêm trọng. | Sửa dữ liệu bẩn (data cleaning), sửa logic transform sai. | Sửa lỗi data leakage, sửa công thức tính loss function. |
| `refactor`| Tối ưu code, tách hàm, giảm nợ kỹ thuật. | Tái cấu trúc mục lục, chia nhỏ/gộp trang, sắp xếp lại ý. | Tối ưu câu query SQL, chuẩn hóa lại schema. | Tái cấu trúc code training, modularize notebook. |
| `style` | Linting, format code (Prettier). | Format bảng, thêm icon, căn chỉnh layout markdown. | Format script SQL, đổi tên biến cho dễ đọc. | Format code Python, sắp xếp lại cell trong Notebook. |
| `docs` | Viết JSDoc, cập nhật README. | Viết lại câu từ cho dễ hiểu (copywriting), thêm ví dụ. | Cập nhật Data Dictionary, mô tả column. | Ghi chú experiment, giải thích model architecture. |
| `chore` | Update library, config CI/CD. | Cập nhật sidebar, footer, config của site tài liệu. | Cập nhật quyền truy cập DB, migration script. | Cập nhật môi trường (conda env), download dataset mới. |

-----

## 4. QUY ĐỊNH VỀ PHẠM VI (SCOPE SPECIFICATION)

Tham số `scope` là một biến định danh vị trí (Location Identifier).

### 4.1. Quy tắc cú pháp

1.  **Dạng thức:** Danh từ (Noun).
2.  **Định dạng:** Chữ thường (lowercase), sử dụng gạch nối cho cụm từ (`kebab-case`).
3.  **Bao bọc:** Nằm trong cặp ngoặc đơn `()`.

### 4.2. Quy tắc chọn lựa Scope

  * **Software:** Tên Module / Package / Service / Component (`auth`, `api`, `button`).
  * **Documentation:** Tên Thư mục chứa tài liệu / Tên Chương (`getting-started`, `guides`).
  * **Data:** Tên Schema / Pipeline / Table (`sales-mart`, `etl-users`).
  * **AI/ML:** Tên Model / Experiment / Step (`preprocessing`, `resnet-50`).

### 4.3. Quy tắc "Phạm vi Rỗng" (Null Scope Rule)

Nếu một thay đổi có tính chất:

1.  Toàn cục (Global impact).
2.  Đa điểm (Cross-cutting concern) ảnh hưởng trên 3 module trở lên.
3.  Không thể định danh bằng một danh từ cụ thể.
    $\rightarrow$ **Bắt buộc bỏ trống `scope`.** Thông tin chi tiết phải được đưa vào phần Body.

-----

## 5. THUẬT TOÁN QUYẾT ĐỊNH (DECISION ALGORITHM)

Khi một commit chứa nhiều loại thay đổi giao thoa, áp dụng **Định lý Ưu tiên (Priority Theorem)** để xác định `type` duy nhất:

> **External ($T_{ext}$) \> Internal ($T_{int}$) \> Supporting ($T_{sup}$)**

**Pseudo-code logic:**

```python
def determine_commit_type(changes):
    if changes.affects_user_output():
        if changes.is_new_capability():
            return 'feat'
        return 'fix'

    elif changes.affects_internal_structure():
        if changes.changes_logic_or_organization():
            return 'refactor'
        return 'style'

    else: # changes.affects_meta_info()
        if changes.is_documentation_text():
            return 'docs'
        return 'chore'
```

-----

## 6. QUY TẮC VĂN PHẠM (GRAMMAR RULES)

Để đảm bảo sự "sạch sẽ" và chuyên nghiệp của bản ghi:

1.  **Thì mệnh lệnh (Imperative Mood):** Luôn dùng "Add", "Fix", "Change". **Tuyệt đối không** dùng "Added", "Fixed", "Changed".
      * *Đúng:* `feat: add user login` (Hãy thêm đăng nhập người dùng - ra lệnh cho codebase).
      * *Sai:* `feat: added user login` (Đã thêm đăng nhập người dùng - kể lể).
2.  **Không chấm câu:** Không sử dụng dấu chấm `.` ở cuối phần Description.
3.  **Viết thường:** Phần `description` bắt đầu bằng chữ thường (trừ khi là danh từ riêng).
4.  **Giới hạn độ dài:** Dòng đầu tiên (Header) không vượt quá **72 ký tự**.

-----

**Kết luận:**

Việc tuân thủ quy chuẩn UCCS không chỉ là tuân thủ quy tắc, mà là sự tôn trọng đối với đồng nghiệp, đối với bản thân trong tương lai, và đối với tính vẹn toàn của dự án. Hãy áp dụng nó với sự kỷ luật và chính xác của một kỹ sư khoa học máy tính.
