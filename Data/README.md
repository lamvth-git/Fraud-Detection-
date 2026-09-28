# Fraud Detection — Ứng dụng học sâu trong phát hiện bất thường và quản lý luồng giao dịch ngân hàng thông minh

Hệ thống phát hiện gian lận giao dịch ngân hàng kết hợp Random Forest, mạng LSTM (Deep Learning)và phân cụm K-Means/PCA, được trực quan hoá trên một dashboard Streamlit.

> Toàn bộ mô hình đã được huấn luyện sẵn (file `.pkl`/`.pth` có trong thư mục `Notebooks/`). Bạn không cần chạy lại notebook để dùng được dashboard — chỉ cần làm theo 6 bước bên dưới.

## 1. Cấu trúc thư mục

```
Fraud-Detection-/
├── Data/
│   ├── AIML Dataset.csv            # Bộ dữ liệu PaySim (6.362.620 giao dịch)
│    └── README.md
├── Notebooks/
│   ├── Analysis_model.ipynb           # EDA + huấn luyện RF, KMeans, LSTM
│   ├── Prepare_Extra_Artifacts.ipynb  # Sinh thêm artifact phục vụ dashboard
│   ├── FraudDetectionPipeline_RF.pkl  # Pipeline Random Forest đã huấn luyện
│   ├── lstm_fraud_model.pth           # Trọng số mô hình LSTM
│   ├── lstm_preprocessing.pkl         # Bộ tiền xử lý cho LSTM
│   ├── kmeans_model.pkl / cluster_scaler.pkl / pca_model.pkl
│   ├── pca_background_sample.csv      # Mẫu nền để vẽ biểu đồ phân cụm
│   ├── threshold.json                 # Ngưỡng cảnh báo (0.6)
│   └── requirements.txt
└── app.py                          # Dashboard Streamlit (chạy file này)

```

## 2. Yêu cầu môi trường

- Python 3.9 trở lên
- Đã cài Git (không bắt buộc, chỉ cần nếu clone repo)

## 3. Cài đặt

Mở terminal tại thư mục gốc `Fraud-Detection-/` (nơi chứa `app.py`) và chạy lần lượt:

```bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường ảo
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# 3. Cài thư viện
pip install -r Notebooks/requirements.txt
```

**Thư viện sử dụng** (đã ghim đúng phiên bản đã dùng để huấn luyện mô hình, tránh cảnh báo lệch phiên bản khi load file `.pkl`):

```
streamlit
pandas
numpy
scikit-learn==1.9.0
joblib
plotly
```

> ⚠️ Lưu ý: các file `.pkl` trong `Notebooks/` được lưu bằng **scikit-learn 1.9.0**. Nếu cài bản scikit-learn khác, chương trình vẫn có thể chạy nhưng sẽ hiện cảnh báo `InconsistentVersionWarning` và có rủi ro dự đoán sai lệch — nên giữ đúng phiên bản như trên.

## 4. Chạy ứng dụng

Vẫn đứng ở thư mục gốc `Fraud-Detection-/` (không `cd` vào `Notebooks/`), chạy:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`. Nếu không tự mở, copy URL hiện trong terminal và dán vào trình duyệt.

Ứng dụng gồm 2 tab:

- **Chấm điểm thủ công**: nhập thông tin một giao dịch, hệ thống tính điểm rủi ro bằng mô hình Random Forest và hiển thị cụm hành vi tương ứng.
- **Mô phỏng Real-time**: mô phỏng luồng giao dịch liên tục lấy ngẫu nhiên từ `Data/AIML Dataset.csv`, cập nhật KPI và cảnh báo gian lận theo thời gian thực.

## 5. (Tùy chọn) Huấn luyện lại mô hình từ đầu

Chỉ cần thực hiện nếu muốn tái tạo lại các file mô hình từ dữ liệu gốc:

```bash
pip install torch matplotlib seaborn jupyter
```

Sau đó mở và chạy lần lượt (bằng VS Code + extension Jupyter, hoặc `jupyter notebook`):

1. `Notebooks/Analysis_model.ipynb` — phân tích dữ liệu, huấn luyện Random Forest, LSTM, K-Means.
2. `Notebooks/Prepare_Extra_Artifacts.ipynb` — sinh các artifact còn lại phục vụ dashboard (`cluster_scaler.pkl`, `pca_model.pkl`, `pca_background_sample.csv`).

## 6. Xử lý lỗi thường gặp
| `ModuleNotFoundError: No module named 'streamlit'`: Chưa kích hoạt venv hoặc cài sai môi trường; Kích hoạt lại venv (bước 3), kiểm tra Python Interpreter đang chọn trong VS Code.
| `FileNotFoundError: Data/AIML Dataset.csv`: Terminal không đứng ở thư mục gốc dự án, `cd` về đúng thư mục chứa `app.py` rồi chạy lại.
| `InconsistentVersionWarning` khi load `.pkl`: Phiên bản scikit-learn khác với lúc huấn luyện (1.9.0). Cài đúng `scikit-learn==1.9.0` như trong `requirements.txt`.
| `Port 8501 is already in use`: Đã có phiên Streamlit khác đang chạy. Đóng phiên cũ, hoặc chạy `streamlit run app.py --server.port 8502`.

## 7. Nguồn dữ liệu

Bộ dữ liệu PaySim — mô phỏng giao dịch tiền di động dựa trên đặc tính thống kê của dữ liệu tài chính thực:

> Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. Proc. 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus, pp. 249–255.