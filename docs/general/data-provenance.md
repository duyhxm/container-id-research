# Một Khuôn khổ Thống nhất để Truy vết Nguồn gốc Tác phẩm Dữ liệu trong các Hệ thống Có thể Tái lập

(A Unified Framework for Data Artifact Provenance in Reproducible Systems)

## Tóm tắt

Tài liệu này đề xuất một khuôn khổ phân loại chính thức và chặt chẽ cho tất cả các tác phẩm dữ liệu (data artifacts) trong các dự án khoa học dữ liệu và học máy. Bằng cách neo hệ thống vào các nguyên tắc cơ bản về ranh giới hệ thống và tính tất định, khuôn khổ này cung cấp một phương pháp luận rõ ràng để đảm bảo tính tái lập (reproducibility), khả năng kiểm toán (auditability) và cộng tác hiệu quả, bất kể quy mô hay độ phức tạp của dự án.

---

## 1. Các Nguyên tắc Nền tảng (Core Principles)

Mọi quyết định phân loại đều phải tuân thủ hai nguyên tắc cơ bản sau:

### 1.1. Ranh giới Hệ thống (The System Boundary)

Chúng ta định nghĩa một "ranh giới" ảo bao quanh repository.

*   **Các quy trình Nội sinh (Endogenous Processes):** Là các hoạt động tính toán hoàn toàn nằm bên trong ranh giới. Chúng chỉ phụ thuộc vào các tác phẩm đã được kiểm soát phiên bản (bằng Git và DVC) và do đó, có thể tái lập được một cách hoàn hảo.
*   **Các quy trình Ngoại sinh (Exogenous Processes):** Là các hoạt động tương tác với thế giới bên ngoài ranh giới (ví dụ: gọi API, truy vấn cơ sở dữ liệu trực tiếp, nhập liệu thủ công). Các quy trình này vốn dĩ không tất định từ góc nhìn của repository.

### 1.2. Tính Tất định và Tính Ngẫu nhiên có Kiểm soát (Determinism and Controlled Stochasticity)

Một quy trình được coi là **tất định (deterministic)** nếu, với cùng một tập hợp đầu vào đã được kiểm soát phiên bản, nó luôn tạo ra một đầu ra giống hệt nhau trên phương diện bit (bit-for-bit identical).
*   **Lưu ý quan trọng:** Đối với các quy trình chứa yếu tố ngẫu nhiên (ví dụ: khởi tạo trọng số, chia tập dữ liệu), chúng chỉ được coi là tất định khi **nguồn ngẫu nhiên của chúng (ví dụ: `random seed`) được kiểm soát phiên bản như một phụ thuộc đầu vào.**

---

## 2. Phân loại Tác phẩm Dữ liệu (Artifact Classification Taxonomy)

Dựa trên các nguyên tắc trên, tất cả tác phẩm dữ liệu được phân vào một trong ba loại chính, loại trừ lẫn nhau.

### Loại I: Tác phẩm Hạt giống (Seed Artifacts)

Đây là những đầu vào nền tảng, được đưa vào hệ thống một cách thủ công hoặc được coi là "chân lý" ban đầu mà không thể được tạo ra bằng code trong repository.

*   **Tiêu chí:** Tác phẩm không phải là đầu ra của bất kỳ quy trình thực thi, được kiểm soát phiên bản nào bên trong repository. Sự tồn tại của nó là tiên nghiệm (a priori).
*   **Phân loại con & Giao thức:**
    *   **1.1. Dữ liệu Tĩnh (Static Data):** Các file dữ liệu không đổi (ví dụ: `human_annotations.jsonl`, `business_rules.csv`).
        *   **Hành động:** Sử dụng **`dvc add`**.
    *   **1.2. Cấu hình & Siêu tham số (Configs & Hyperparameters):** Các biến số điều khiển hành vi hệ thống (ví dụ: `learning_rate`, `batch_size`).
        *   **Hành động:** Không dùng `dvc add`. Hãy lưu trữ trong **`params.yaml`** và kiểm soát phiên bản bằng **Git**. Điều này cho phép DVC theo dõi sự thay đổi hiệu suất dựa trên sự thay đổi tham số.

### Loại II: Tác phẩm Nhập vào (Ingested Artifacts)

Đây là các snapshot của thế giới bên ngoài, được đưa vào bên trong ranh giới hệ thống thông qua một quy trình có kịch bản.

*   **Tiêu chí:** Tác phẩm là đầu ra của một script được kiểm soát phiên bản, nhưng script đó tương tác với một nguồn ngoại sinh, không tất định. Việc chạy lại script không đảm bảo tạo ra kết quả giống hệt.
*   **Ví dụ:**
    *   Dữ liệu được lấy từ một API theo thời gian thực (`realtime_metrics_snapshot_20231027T1000.json`).
    *   Kết quả trích xuất từ một cơ sở dữ liệu sản xuất đang hoạt động.
    *   Dữ liệu thu thập từ một quy trình cào web (web scraping).
*   **Giao thức Quản lý Phiên bản:** Quy trình hai bước để "đóng băng" trạng thái của thế giới bên ngoài.
    1.  **Thực thi script** để tạo ra tệp dữ liệu thô.
        ```bash
        python scripts/ingest/fetch_from_production_db.py
        ```
    2.  **Đóng băng đầu ra** bằng **`dvc add`**, biến đầu ra không tất định này thành một đầu vào cố định và bất biến cho tất cả các giai đoạn sau.
        ```bash
        dvc add data/ingested/production_snapshot_20231027.parquet
        ```

### Loại III: Tác phẩm Phái sinh (Derived Artifacts)

Đây là tất cả các tác phẩm được tạo ra một cách tất định bởi các quy trình nội sinh bên trong ranh giới hệ thống. Trạng thái của chúng hoàn toàn được quyết định bởi trạng thái của các phụ thuộc.

*   **Tiêu chí:** Tác phẩm là đầu ra của một script, và **tất cả** các phụ thuộc của script đó (code, dữ liệu đầu vào, tham số) đều là các tác phẩm đã được kiểm soát phiên bản trong hệ thống (Loại I, II, hoặc III khác).
*   **Giao thức Quản lý Phiên bản:** Sử dụng **`dvc stage add`** (hoặc `dvc run`) để định nghĩa "công thức" tạo ra tác phẩm, ghi lại toàn bộ dòng chảy dữ liệu (data lineage).
    ```bash
    dvc stage add -n featurize -d scripts/build_features.py -d data/processed/cleaned_data.pkl -o data/features/final_features.pkl ...
    ```
*   **Phân loại con (Sub-classification):** Để tăng cường sự rõ ràng, các Tác phẩm Phái sinh nên được phân loại nhỏ hơn dựa trên vai trò của chúng trong pipeline:
    *   **3.1. Tác phẩm Trung gian (Intermediate Artifacts):** Dữ liệu đang trong quá trình chuyển đổi. Chúng vừa là đầu ra của một bước, vừa là đầu vào của bước tiếp theo (ví dụ: `cleaned_data.parquet`, `tokenized_text.pkl`, `user_features.feather`).
    *   **3.2. Tác phẩm Cuối cùng (Terminal Artifacts):** Là các sản phẩm cuối cùng của một pipeline, thường được tiêu thụ bởi người dùng hoặc các hệ thống khác.
        *   **3.2.1. Mô hình (Models):** Các đối tượng mô hình đã được huấn luyện, chứa đựng "trí thông minh" có thể thực thi (ví dụ: `xgboost_model.bin`, `transformer_epoch_10.pt`).
        *   **3.2.2. Sản phẩm Bàn giao (Deliverables):** Các kết quả dành cho việc phân tích, được DVC nhận diện đặc biệt để tạo báo cáo tự động.
			*   **Số liệu (Metrics):** Các file chứa cặp khóa-giá trị đơn giản (ví dụ: `scores.json` chứa `{"accuracy": 0.95}`). Khai báo bằng cờ `-M` hoặc `--metrics` trong `dvc run`.
			*   **Biểu đồ (Plots):** Các file dữ liệu dạng bảng dùng để vẽ đồ thị (ví dụ: `pr_curve.csv`). Khai báo bằng cờ `--plots` trong `dvc run`.

---

## 3. Ma trận Quyết định và Quy trình Làm việc

Áp dụng tuần tự các câu hỏi sau để xác định loại của bất kỳ tác phẩm dữ liệu nào.

| Bước | Câu hỏi                                                                                             | Nếu CÓ                                                                    | Nếu KHÔNG                                   |
| :--- | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- | :------------------------------------------- |
| 1    | Tác phẩm có được tạo ra bởi một script thực thi nằm trong repository không?                         | Chuyển đến Bước 2.                                                         | **Loại I: Hạt giống**. Dùng `dvc add`.      |
| 2    | Script tạo ra nó có tương tác với một hệ thống bên ngoài ranh giới (API, DB, web,...) không?         | **Loại II: Nhập vào**. Chạy script, sau đó dùng `dvc add` trên đầu ra.      | Chuyển đến Bước 3.                           |
| 3    | Toàn bộ các phụ thuộc của script (code, dữ liệu, tham số) đều đã được kiểm soát phiên bản chưa? | **Loại III: Phái sinh**. Dùng `dvc stage add` để ghi lại quy trình. | **Bất thường:** Phụ thuộc chưa được kiểm soát. Hãy khắc phục điều này trước. |

---

## 4. Những Cân nhắc Thực tế và Trường hợp Biên

### 4.1. Cạm bẫy của "Versioning câu lệnh truy vấn" (The SQL Trap)

Một sai lầm phổ biến là chỉ kiểm soát phiên bản file `.sql` thay vì dữ liệu trả về. **Điều này vi phạm tính tái lập**, vì cơ sở dữ liệu là trạng thái thay đổi (mutating state). Cùng một câu SQL chạy vào hai thời điểm khác nhau sẽ cho ra kết quả khác nhau.
*   **Giải pháp:**
    *   **Snapshot:** Luôn lưu kết quả truy vấn ra file (parquet/csv) và version file đó (như Loại II).
    *   **Sampling (Với dữ liệu quá lớn):** Nếu không thể lưu toàn bộ, hãy lưu một bản mẫu (sample) đại diện bằng DVC để phát triển và kiểm thử code.
    *   **Metadata Logging:** Nếu bắt buộc dùng dữ liệu trực tiếp từ Data Lake, hãy tính toán và lưu trữ hash hoặc thống kê phân phối (distribution statistics) của dữ liệu tại thời điểm chạy để phát hiện sự trôi dạt dữ liệu (data drift).

### 4.2. Cơ chế Cộng tác và Bộ nhớ đệm (Collaboration & Shared Cache)

Để làm việc nhóm hiệu quả, hệ thống cần được cấu hình **Remote Storage** (S3, GCS, Azure Blob).
*   **Cơ chế Shared Cache:** Khi một thành viên đã chạy quy trình tốn kém (ví dụ: Tiền xử lý dữ liệu mất 5 giờ) và đẩy lên Remote (`dvc push`), các thành viên khác chỉ cần `dvc pull` để tải kết quả về ngay lập tức mà không cần chạy lại.
*   **Quy tắc Vàng:** "Không bao giờ `git push` code mới nếu chưa `dvc push` dữ liệu/tác phẩm tương ứng."

### 4.3. Phụ thuộc Môi trường

Môi trường thực thi là một dạng phụ thuộc ẩn.
*   Sử dụng Docker để đóng gói môi trường.
*   File `Dockerfile` hoặc `requirements.txt` phải được coi là Tác phẩm Hạt giống (quản lý bằng Git) và mọi thay đổi trong đó đều yêu cầu kiểm tra lại tính đúng đắn của toàn bộ pipeline.

## 5. Cam kết Chất lượng

Chúng tôi hiểu rằng: **"Code chỉ là phần nổi của tảng băng chìm; Dữ liệu và Cấu hình mới là phần chìm quyết định sự ổn định."** Việc tuân thủ khuôn khổ này không phải là sự quan liêu, mà là sự tôn trọng đối với tính trung thực khoa học và công sức của chính đội ngũ trong tương lai.
