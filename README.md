# 💄 TQH-LAB010-N5  
### Phân tích sự ưu tiên của người tiêu dùng Việt Nam đối với mỹ phẩm trong và ngoài nước trên sàn **Tiki**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Dashboard-Dash%20%7C%20Plotly-6f42c1?logo=plotly" alt="Dash Plotly">
  <img src="https://img.shields.io/badge/Data-Tiki%20Cosmetics-orange" alt="Tiki Cosmetics">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/Course-Data%20Analysis-informational" alt="Course">
</p>

<p align="center">
  Đồ án môn học của <b>Nhóm 05 – HCMUS</b>, tập trung thu thập, làm sạch, phân tích và trực quan hóa dữ liệu mỹ phẩm trên sàn thương mại điện tử Tiki.
</p>

---

## 📌 Tổng quan dự án

Dự án nghiên cứu hành vi tiêu dùng mỹ phẩm trên sàn **Tiki**, với trọng tâm là so sánh giữa hai nhóm sản phẩm:

- **Mỹ phẩm trong nước**
- **Mỹ phẩm ngoài nước / nhập khẩu**

Từ dữ liệu thực tế được crawl từ Tiki, nhóm tiến hành:

- Thu thập và tổng hợp dữ liệu sản phẩm
- Làm sạch, chuẩn hóa và tạo biến phân tích
- Khai phá các insight về **giá cả**, **uy tín**, **chất lượng cảm nhận**, **thị phần** và **mức độ cạnh tranh của hàng nội địa**
- Xây dựng **dashboard tương tác** để trình bày kết quả

> Quy mô dữ liệu: **hơn 7.000 sản phẩm mỹ phẩm** trên Tiki.

---

## 🎯 Mục tiêu nghiên cứu

Dự án được xây dựng để trả lời các câu hỏi như:

- Người tiêu dùng sẵn sàng chi trả bao nhiêu cho mỹ phẩm nội địa so với hàng nhập khẩu?
- Mức độ tin tưởng của khách hàng, thể hiện qua rating và review, đang nghiêng về nhóm hàng nào?
- Thương hiệu Việt đang có lợi thế ở những phân khúc nào?
- Tỷ lệ chuyển đổi từ **lượt bán** sang **lượt đánh giá** của từng nhóm sản phẩm ra sao?
- Hàng nội địa có đang cạnh tranh tốt với hàng ngoại trên cùng nền tảng bán lẻ hay không?

---

## 🧠 Giá trị nổi bật của đồ án

- Khai thác dữ liệu từ môi trường thương mại điện tử thực tế
- Kết hợp giữa **data collection**, **data preprocessing**, **analysis** và **visual analytics**
- Có dashboard trực quan, dễ trình bày trong báo cáo hoặc thuyết trình
- Có thể mở rộng thành bài toán nghiên cứu hành vi người tiêu dùng hoặc phân tích cạnh tranh thị trường

---

## 👥 Thành viên nhóm

| MSSV | Họ và tên | Vai trò |
|---|---|---|
| 23120201 | **Nguyễn Thị Trúc Hằng** | Leader – Crawling & Preprocessing |
| 23120192 | **Nguyễn Nhật Vy** | Quality & Satisfaction Analysis |
| 23120193 | **Trần Kim Yến** | Brand & Trust Analysis |
| 23120206 | **Ngô Thuận An** | Pricing & Consumer Behavior Analysis |

---

## 🗂️ Cấu trúc thư mục

```text
TQH-LAB010-N5/
├── dashboard/                              # Ứng dụng dashboard trực quan hóa
│   ├── assets/                             # CSS dùng cho các trang dashboard
│   │   ├── stylePage0.css                  # CSS cho trang Overview
│   │   ├── stylePage1.css                  # CSS cho trang Thị phần
│   │   └── stylePage2.css                  # CSS cho trang Uy tín
│   │
│   ├── data/                               # Dữ liệu đặt bên trong dashboard để app sử dụng
│   │
│   ├── pages/                              # Các trang phân tích của dashboard
│   │   ├── page0_overview.py               # Trang tổng quan
│   │   ├── page1_thi_phan.py               # Trang phân tích thị phần
│   │   ├── page2_uy_tin.py                 # Trang phân tích uy tín
│   │   ├── page3_chat_luong.py             # Trang phân tích chất lượng
│   │   └── page4_gia_ca.py                 # Trang phân tích giá cả
│   │
│   ├── app.py                              # File chạy chính của dashboard Dash
│   ├── data_loader.py                      # Đọc và nạp dữ liệu dùng chung cho các trang
│   └── theme.py                            # Theme, màu sắc và helper giao diện
│
├── data/                                   # Dữ liệu chính của dự án
│   ├── tiki_cosmetics_processed.csv        # Dữ liệu đã làm sạch và xử lý
│   └── tiki_cosmetics_raw.csv              # Dữ liệu thô thu thập từ Tiki
│
├── docs/                                   # Tài liệu môn học và báo cáo
│   ├── Lab 01.pdf                          # Đề bài / tài liệu Lab 01
│   └── Report_Lab01.docx                   # Báo cáo của nhóm
│
├── notebooks/                              # Notebook phân tích của từng thành viên
│   ├── DataVisualization _ 23120192.ipynb
│   ├── DataVisualization _ 23120193.ipynb
│   ├── DataVisualization _ 23120201.ipynb
│   └── DataVisualization _ 23120206.ipynb
│
├── old_data/                               # Dữ liệu cũ / bản lưu trước đó
│   ├── tiki_cosmetics_processed.csv
│   └── tiki_cosmetics_raw.csv
│
├── scripts/                                # Notebook phục vụ crawl và tiền xử lý
│   ├── crawling.ipynb                      # Thu thập dữ liệu
│   ├── eda_overview.ipynb                  # Khám phá dữ liệu tổng quan
│   └── preprocessing.ipynb                 # Làm sạch và tiền xử lý dữ liệu
│
├── .gitignore                              # Khai báo file/thư mục không đưa lên Git
├── README.md                               # Mô tả dự án và hướng dẫn sử dụng
└── requirements.txt                        # Danh sách thư viện cần cài
```

---

## ⚙️ Quy trình thực hiện

### 1. Thu thập dữ liệu
Dữ liệu được thu thập từ Tiki thông qua quá trình crawl theo danh mục và từ khóa sản phẩm, bao gồm các trường như:

- tên sản phẩm
- thương hiệu
- giá
- xuất xứ
- lượt bán
- lượt đánh giá
- điểm rating
- danh mục sản phẩm

### 2. Tiền xử lý dữ liệu
Nhóm thực hiện:

- xử lý giá trị thiếu
- chuẩn hóa quốc gia xuất xứ
- phân loại sản phẩm thành **trong nước** và **ngoài nước**
- loại bỏ dữ liệu bất thường
- chuẩn hóa biến phục vụ phân tích

### 3. Feature Engineering
Một số biến được xây dựng thêm:

- **Estimated Revenue** = `price × sold_count`
- **Review Ratio** = `review_count / sold_count`
- **Price Segment** = phân khúc giá
- **Product Type** = nhóm ngành hàng như Skincare, Makeup, Hair Care, Body Care, Fragrance
- **DSI (Domestic Strength Index)** = chỉ số phản ánh sức mạnh cạnh tranh của hàng nội địa

### 4. Phân tích dữ liệu
Các phân tích chính tập trung vào:

- **Thị phần**
- **Giá cả**
- **Uy tín thương hiệu**
- **Chất lượng cảm nhận**
- **Hiệu quả cạnh tranh của hàng nội địa**

### 5. Trực quan hóa
Kết quả được trình bày qua:

- notebook phân tích chi tiết
- dashboard tương tác bằng Dash/Plotly

---

## 📊 Dashboard

Dashboard được xây dựng để hỗ trợ xem nhanh các insight quan trọng theo từng chủ đề.

### Các trang chính
- **Overview**
- **Thị phần**
- **Uy tín**
- **Chất lượng**
- **Giá cả**

### Demo
Bạn có thể thêm ảnh chụp dashboard vào đây sau khi hoàn thiện giao diện:

```markdown
<p align="center">
  <img src="docs/images/dashboard-overview.png" width="900" alt="Dashboard Overview">
</p>
```

Hoặc nhiều ảnh:

```markdown
<p align="center">
  <img src="docs/images/page-overview.png" width="48%">
  <img src="docs/images/page-thi-phan.png" width="48%">
</p>
```

---

## 🚀 Cách cài đặt

### 1. Clone repository
```bash
git clone https://github.com/TrucHang676/TQH-LAB010-N5.git
cd TQH-LAB010-N5
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

Nếu cần, có thể cài thủ công:

```bash
pip install pandas numpy requests matplotlib seaborn plotly dash tqdm
```

---

## ▶️ Cách chạy dự án

### Xem pipeline dữ liệu
Mở các notebook trong thư mục `scripts/` để xem:

- quá trình crawl dữ liệu
- quá trình preprocessing

### Xem phân tích
Mở các notebook trong thư mục `notebooks/`.

### Chạy dashboard
```bash
cd dashboard
python app.py
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:8050
```

---

## 📈 Một số đầu ra chính

- Bộ dữ liệu mỹ phẩm Tiki đã được làm sạch
- Các notebook phân tích theo chủ đề
- Dashboard trực quan hóa kết quả
- Insight về hành vi tiêu dùng và mức độ cạnh tranh giữa hàng nội và hàng ngoại

---

## 🛠️ Công nghệ sử dụng

### Ngôn ngữ
- **Python 3.10+**

### Thư viện chính
- **pandas, numpy** – xử lý dữ liệu
- **requests** – crawl dữ liệu
- **matplotlib, seaborn** – trực quan hóa trong notebook
- **plotly, dash** – dashboard tương tác
- **tqdm** – theo dõi tiến độ xử lý

---

## 📚 Gợi ý mở rộng trong tương lai

- Bổ sung phân tích theo thương hiệu cụ thể
- Dự đoán khả năng bán chạy của sản phẩm
- So sánh dữ liệu giữa nhiều sàn thương mại điện tử
- Kết hợp sentiment analysis từ nội dung review
- Triển khai dashboard online

---

## 📝 Ghi chú

- Dữ liệu được thu thập tại thời điểm thực hiện đồ án, nên có thể thay đổi theo thời gian.
- Một số chỉ số như **Estimated Revenue** hay **DSI** mang tính suy luận để phục vụ mục tiêu học tập và phân tích.
- Repo này phù hợp cho mục đích học tập, tham khảo quy trình xử lý dữ liệu và xây dựng dashboard.

---

## 🙌 Ghi nhận

Đây là đồ án của **Nhóm 05 – HCMUS**.  
Nếu bạn sử dụng lại repo cho mục đích học tập hoặc tham khảo, vui lòng ghi nguồn nhóm thực hiện.

---

## ⭐ Mẹo để README đẹp hơn trên GitHub

Bạn có thể thêm các phần sau nếu muốn repo “xịn” hơn nữa:

- ảnh chụp dashboard thật
- GIF demo thao tác dashboard
- mục lục tự động
- sơ đồ pipeline
- liên kết tới báo cáo PDF
- liên kết tới video thuyết trình

Ví dụ phần liên kết:

```markdown
## 🔗 Tài liệu liên quan

- [Báo cáo PDF](docs/report.pdf)
- [Slide thuyết trình](docs/slides.pdf)
- [Video demo](https://...)
```
