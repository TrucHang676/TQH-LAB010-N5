# 💄 TQH-LAB010-N5

### Phân tích sự ưu tiên của người tiêu dùng Việt Nam đối với mỹ phẩm trong và ngoài nước trên sàn Tiki

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Dashboard-Dash%20%7C%20Plotly-0A66C2?logo=plotly" alt="Dash Plotly">
  <img src="https://img.shields.io/badge/Data-Tiki%20Cosmetics-orange" alt="Data">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

Dự án môn học của Nhóm 05 - HCMUS, tập trung thu thập, làm sạch, phân tích và trực quan hóa dữ liệu mỹ phẩm trên Tiki.

---

## 1. Tổng quan dự án

Dự án nghiên cứu hành vi tiêu dùng mỹ phẩm trên sàn Tiki, trọng tâm là so sánh:

- Mỹ phẩm trong nước
- Mỹ phẩm ngoài nước/nhập khẩu

Nhóm triển khai đầy đủ pipeline:

- Thu thập dữ liệu từ nguồn thực tế
- Tiền xử lý và chuẩn hóa dữ liệu
- Phân tích thị phần, giá cả, uy tín, thương hiệu
- Xây dựng dashboard tương tác bằng Dash/Plotly
- Mở rộng phân tích Machine Learning với regression và clustering

Quy mô dữ liệu: hơn 7.000 sản phẩm mỹ phẩm.

## 2. Mục tiêu nghiên cứu

Dự án hướng đến các câu hỏi:

- Người tiêu dùng sẵn sàng chi trả cho mỹ phẩm nội địa so với nhập khẩu như thế nào?
- Mức độ tin tưởng của khách hàng (rating/review) đang nghiêng về nhóm nào?
- Thương hiệu Việt đang có lợi thế ở những phân khúc nào?
- Hàng nội địa có đang cạnh tranh tốt với hàng ngoại trên cùng nền tảng bán lẻ không?

## 3. Cấu trúc thư mục thực tế

```text
Lab01/                                        # Thư mục gốc của toàn bộ đồ án
├── README.md                                 # Mô tả dự án, cấu trúc, cách cài và cách chạy
├── requirements.txt                          # Danh sách thư viện cần cài cho toàn bộ dự án
├── .gitignore                                # Loại trừ file/thư mục không đưa lên Git
│
├── dashboard/                                # Ứng dụng dashboard chính để phân tích dữ liệu
│   ├── app.py                                # Entry point của dashboard Dash
│   ├── data_loader.py                        # Hàm nạp dữ liệu dùng chung cho các trang
│   ├── theme.py                              # Quy ước màu sắc, theme và style helper
│   ├── assets/                               # CSS cho dashboard chính
│   │   ├── stylePage0.css                    # Giao diện trang Overview
│   │   ├── stylePage1.css                    # Giao diện trang Thị phần
│   │   ├── stylePage2.css                    # Giao diện trang Uy tín
│   │   ├── stylePage3.css                    # Giao diện trang Thương hiệu
│   │   └── stylePage4.css                    # Giao diện trang Giá cả
│   ├── data/                                 # Dữ liệu phục vụ riêng cho dashboard
│   │   └── tiki_cosmetics_processed.csv      # Dữ liệu đã xử lý dùng để hiển thị biểu đồ
│   └── pages/                                # Các trang phân tích trong dashboard
│       ├── page0_overview.py                 # Trang tổng quan toàn bộ dự án
│       ├── page1_thi_phan.py                 # Phân tích thị phần và phân bố doanh thu
│       ├── page2_uy_tin.py                   # Phân tích uy tín và mức độ tin cậy
│       ├── page3_thuong_hieu.py              # Phân tích thương hiệu và trạng thái verified
│       └── page4_gia_ca.py                   # Phân tích giá cả và khuyến mãi
│
├── machine learning/                         # Module ML độc lập cho dự đoán và phân cụm
│   ├── app.py                                # Entry point của dashboard ML
│   ├── data_loader.py                        # Nạp dữ liệu cho các trang ML
│   ├── train_model1.py                       # Script huấn luyện mô hình regression
│   ├── train_model2.py                       # Script huấn luyện mô hình clustering
│   ├── README_ML.md                          # Hướng dẫn riêng cho module ML
│   ├── .gitignore                            # Loại trừ file tạm của module ML
│   ├── assets/                               # CSS cho dashboard ML
│   │   ├── stylePage0.css                    # Style dùng chung
│   │   ├── stylePage3.css                    # Style tái sử dụng từ dashboard chính
│   │   └── stylePageML.css                   # Giao diện riêng cho trang ML
│   ├── data/                                 # Dữ liệu đã xử lý phục vụ module ML
│   │   └── tiki_cosmetics_processed.csv      # Dataset đầu vào cho train/visualize ML
│   ├── pages/                                # Các trang ML trong dashboard
│   │   ├── page_ml_clustering.py             # Giao diện và logic phân cụm
│   │   └── page_ml_regression.py             # Giao diện và logic dự đoán lượt bán
│   └── ml_models/                            # Tài nguyên mô hình và kết quả đã lưu
│       ├── model1_feature_importance.csv     # Độ quan trọng đặc trưng của model regression
│       ├── model1_learning_curve.csv         # Đường học của model regression
│       ├── model1_partial_dependence.csv     # Partial dependence của model regression
│       ├── model1_predictions.csv            # Kết quả dự đoán mẫu của model regression
│       ├── model1_regressor.joblib           # Mô hình regression đã train
│       ├── model2_cluster_labels.csv         # Nhãn cụm sau khi phân cụm
│       ├── model2_cluster_samples.csv        # Mẫu đại diện cho từng cụm
│       ├── model2_corr_filtered.csv          # Ma trận tương quan đã lọc
│       ├── model2_corr_raw.csv               # Ma trận tương quan gốc
│       ├── model2_elbow_silhouette.csv       # Kết quả elbow/silhouette
│       ├── model2_kmeans.joblib              # Mô hình KMeans đã train
│       ├── model2_pca.joblib                 # PCA dùng để giảm chiều
│       ├── model2_scaler.joblib              # Bộ scaler dùng tiền xử lý
│       └── precompute_benchmarks.py          # Script tiền tính các chỉ số/benchmark
│
├── data/                                     # Dữ liệu gốc và dữ liệu đã xử lý ở cấp dự án
│   ├── tiki_cosmetics_processed.csv          # Bản dữ liệu sạch dùng cho phân tích
│   └── tiki_cosmetics_raw.csv                # Bản dữ liệu thô sau khi crawl
│
├── Group-5.pdf                               # Báo cáo chính của nhóm
│
├── notebooks/                                # Notebook phân tích của từng thành viên
│   ├── DataVisualization _ 23120192.ipynb    # Notebook trực quan hóa của thành viên 23120192
│   ├── DataVisualization _ 23120193.ipynb    # Notebook trực quan hóa của thành viên 23120193
│   ├── DataVisualization _ 23120201.ipynb    # Notebook trực quan hóa của thành viên 23120201
│   └── DataVisualization _ 23120206.ipynb    # Notebook trực quan hóa của thành viên 23120206
│
└── scripts/                                  # Notebook hỗ trợ crawl, EDA và preprocessing
    ├── crawling.ipynb                        # Thu thập dữ liệu từ Tiki
    ├── eda_overview.ipynb                    # Khám phá dữ liệu tổng quan
    └── preprocessing.ipynb                   # Làm sạch và chuẩn hóa dữ liệu
```

Ghi chú:

- Các thư mục môi trường như .venv/, .git/, __pycache__/ không liệt kê chi tiết.
- Dữ liệu được đặt cả ở cấp dự án và trong từng module để thuận tiện chạy độc lập.

## 4. Quy trình thực hiện

### 4.1 Thu thập dữ liệu

Dữ liệu được crawl từ Tiki, gồm các trường điển hình:

- Tên sản phẩm
- Thương hiệu
- Giá
- Xuất xứ
- Lượt bán
- Lượt đánh giá
- Điểm rating
- Danh mục sản phẩm

### 4.2 Tiền xử lý dữ liệu

- Xử lý giá trị thiếu
- Chuẩn hóa quốc gia xuất xứ
- Phân nhóm Trong nước/Ngoài nước
- Loại bỏ dữ liệu bất thường
- Chuẩn hóa biến phục vụ phân tích

### 4.3 Feature Engineering

- Estimated Revenue = price × sold_count
- Review Ratio = review_count / sold_count
- Price Segment
- Product Type
- DSI (Domestic Strength Index)

### 4.4 Phân tích và trực quan hóa

Dashboard chính trong thư mục dashboard/pages gồm 5 trang:

- Overview
- Thị phần
- Uy tín
- Thương hiệu
- Giá cả

## 5. Module Machine Learning

Module trong thư mục machine learning gồm:

- Regression: dự đoán lượt bán
- Clustering: phân cụm sản phẩm/nhóm thị trường

Đầu ra ML đã lưu sẵn:

- Mô hình huấn luyện (.joblib)
- Feature importance, learning curve, partial dependence
- Kết quả phân cụm và benchmark

## 6. Cài đặt

Yêu cầu:

- Python 3.10+

Cài thư viện:

```bash
pip install -r requirements.txt
```

Khuyến nghị tạo môi trường ảo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Cách chạy dự án

### 7.1 Dashboard phân tích chính

```bash
cd dashboard
python app.py
```

Truy cập: http://127.0.0.1:8050

### 7.2 Dashboard Machine Learning

```bash
cd "machine learning"
python app.py
```

Truy cập: http://127.0.0.1:8051

### 7.3 Notebook và pipeline

- scripts/: crawling, preprocessing, EDA tổng quan
- notebooks/: notebook phân tích của từng thành viên

## 8. Công nghệ sử dụng

- Python
- pandas, numpy
- requests, beautifulsoup4, webdriver-manager
- matplotlib, seaborn, plotly
- dash, dash-bootstrap-components
- scikit-learn
- jupyter, notebook, ipykernel

## 9. Thành viên nhóm

| MSSV | Họ và tên | Vai trò |
|---|---|---|
| 23120201 | Nguyễn Thị Trúc Hằng | Leader - Crawling, EDA, Dashboard, Report |
| 23120192 | Nguyễn Nhật Vy | EDA, Machine Learning, Report |
| 23120193 | Trần Kim Yến | Preprocessing, EDA, Dashboard, Report |
| 23120206 | Ngô Thuận An | EDA, Machine Learning, Report |

## 10. Ghi chú

- Dữ liệu mang tính thời điểm và có thể thay đổi theo thời gian.
- Một số chỉ số suy luận (Estimated Revenue, DSI) phục vụ mục tiêu học tập và phân tích.
- Có thể mở rộng thêm sentiment analysis, dự đoán bán chạy, hoặc so sánh đa sàn TMĐT.

## 11. Deploy tự động 2 dashboard (Render)

Repo đã có sẵn file `render.yaml` để deploy đồng thời:

- Service 1: `dashboard`
- Service 2: `machine learning`

Thiết lập lần đầu (chỉ làm 1 lần):

```bash
git push origin main
```

1. Vào Render → **New +** → **Blueprint**.
2. Chọn repo GitHub của dự án.
3. Render đọc `render.yaml` và tạo 2 web services tự động.
4. Chờ deploy lần đầu hoàn tất.

Sau lần đầu, chỉ cần 1 lệnh để deploy lại:

```bash
git push origin main
```

Vì `autoDeploy: true`, mỗi lần push lên branch chính thì cả 2 service sẽ tự build và deploy lại.

Link web sau deploy:

- Dashboard chính: https://tiki-dashboard.onrender.com
- Dashboard Machine Learning: https://tiki-ml-dashboard.onrender.com

Nếu bạn đổi tên service trên Render thì URL cũng đổi theo.

---

Nếu bạn sử dụng lại repo cho mục đích học tập/tham khảo, vui lòng ghi nguồn Nhóm 05 - HCMUS.
