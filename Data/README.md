# Dataset

Dataset (~410 MB) không được lưu trong repo vì vượt giới hạn 100 MB của GitHub.
Bạn cần tải về từ Kaggle và đặt đúng vị trí như hướng dẫn bên dưới.

## Nguồn dữ liệu

Kaggle: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

## Hướng dẫn tải

1. Mở link Kaggle ở trên (cần đăng nhập tài khoản Kaggle).
2. Bấm **Download** để tải file về, giải nén nếu là file `.zip`.
3. Đổi tên file thành `AIML Dataset.csv` (nếu tên file tải về khác).
4. Đặt file vào thư mục `Data/` của dự án, để có đường dẫn đúng như sau:

```
Fraud-Detection-/
└── Data/
    ├── README.md
    └── AIML Dataset.csv   <-- đặt file ở đây
```

## Lưu ý

- File `AIML Dataset.csv` đã được thêm vào `.gitignore`, nên sẽ không bị đẩy lên GitHub.
- Phải có file này trước khi chạy các notebook trong thư mục `Notebooks/`.