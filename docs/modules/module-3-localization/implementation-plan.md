# MODULE 3: IMPLEMENTATION PLAN

## Giai đoạn 1: Huấn luyện (Training Pipeline)

**Mục tiêu:** Thiết lập training script tích hợp WandB để chạy trên Kaggle.

- [ ] **Task 1.1: Develop Training Script**
    - **File:** `src/localization/train_and_evaluate.py`.
    - **Yêu cầu:**
        - Load model `yolo11s-pose.pt`.
        - Tích hợp `wandb` (init project, log metrics).
        - Hỗ trợ tham số dòng lệnh (argparse) để dễ dàng config khi chạy trên Kaggle.
        - Test model trên tập test và tính toán các chỉ số đánh giá.

- [ ] **Task 1.2: Kaggle Notebook Setup**
    - **File:** `notebooks/train_module_3_runner.ipynb`.
    - **Flow:** Clone Repo -> Pull Data (DVC) -> Run `src/train_module_3.py` -> Save Outputs.
    - Tham khảo file `train_notebook.py` để học hỏi từ module 1.