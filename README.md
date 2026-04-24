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
Lab01/
├── README.md
├── requirements.txt
├── .gitignore
│
├── dashboard/
│   ├── app.py
│   ├── data_loader.py
│   ├── theme.py
│   ├── assets/
│   │   ├── stylePage0.css
│   │   ├── stylePage1.css
│   │   ├── stylePage2.css
│   │   ├── stylePage3.css
│   │   └── stylePage4.css
│   ├── data/
│   │   └── tiki_cosmetics_processed.csv
│   └── pages/
│       ├── page0_overview.py
│       ├── page1_thi_phan.py
│       ├── page2_uy_tin.py
│       ├── page3_thuong_hieu.py
│       └── page4_gia_ca.py
│
├── machine learning/
│   ├── app.py
│   ├── data_loader.py
│   ├── train_model1.py
│   ├── train_model2.py
│   ├── README_ML.md
│   ├── .gitignore
│   ├── assets/
│   │   ├── stylePage0.css
│   │   ├── stylePage3.css
│   │   └── stylePageML.css
│   ├── data/
│   │   └── tiki_cosmetics_processed.csv
│   ├── pages/
│   │   ├── page_ml_clustering.py
│   │   └── page_ml_regression.py
│   └── ml_models/
│       ├── model1_feature_importance.csv
│       ├── model1_learning_curve.csv
│       ├── model1_partial_dependence.csv
│       ├── model1_predictions.csv
│       ├── model1_regressor.joblib
│       ├── model2_cluster_labels.csv
│       ├── model2_cluster_samples.csv
│       ├── model2_corr_filtered.csv
│       ├── model2_corr_raw.csv
│       ├── model2_elbow_silhouette.csv
│       ├── model2_kmeans.joblib
│       ├── model2_pca.joblib
│       ├── model2_scaler.joblib
│       └── precompute_benchmarks.py
│
├── data/
│   ├── tiki_cosmetics_processed.csv
│   └── tiki_cosmetics_raw.csv
│
├── docs/
│   ├── Lab 01.pdf
│   └── Report_Lab01.docx
│
├── notebooks/
│   ├── DataVisualization _ 23120192.ipynb
│   ├── DataVisualization _ 23120193.ipynb
│   ├── DataVisualization _ 23120201.ipynb
│   └── DataVisualization _ 23120206.ipynb
│
└── scripts/
    ├── crawling.ipynb
    ├── eda_overview.ipynb
    └── preprocessing.ipynb
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

---

Nếu bạn sử dụng lại repo cho mục đích học tập/tham khảo, vui lòng ghi nguồn Nhóm 05 - HCMUS.
