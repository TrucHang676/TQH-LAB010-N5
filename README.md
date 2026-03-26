# Đồ án Lab 01: Phân tích sự ưu tiên của người tiêu dùng Việt Nam đối với mỹ phẩm trong và ngoài nước trên sàn thương mại điện tử Tiki

**Nhóm 05 - Lớp CQ2023/24 - Khoa Công nghệ Thông tin (HCMUS)**

---

## I. Giới thiệu dự án
Dự án tập trung vào việc thu thập và phân tích dữ liệu từ hơn **7.000 sản phẩm** thuộc ngành hàng Mỹ phẩm trên sàn thương mại điện tử Tiki. Mục tiêu cốt lõi là so sánh hành vi, tâm lý và sự ưu tiên của người tiêu dùng đối với các dòng sản phẩm **Trong nước (Domestic)** và **Ngoài nước (International)**.

### Các câu hỏi nghiên cứu chính:
* Người tiêu dùng sẵn sàng chi trả bao nhiêu cho mỹ phẩm nội địa so với hàng nhập khẩu?
* Niềm tin của khách hàng (thể hiện qua Rating và Review) nghiêng về nhóm hàng nào?
* Các thương hiệu Việt (Local Brands) đang chiếm ưu thế ở những ngách sản phẩm nào?
* Tỉ lệ chuyển đổi từ lượt bán sang lượt đánh giá của nhóm hàng nào cao hơn?

---

## II. Thành viên nhóm & Vai trò
| MSSV | Họ và Tên | Vai trò chính |
| :--- | :--- | :--- |
| [23120201] | **Nguyễn Thị Trúc Hằng** | Leader - Data Engineer (Crawl & Preprocessing) |
| [23120192] | **Nguyễn Nhật Vy** | Data Analyst - Quality & Satisfaction Analysis |
| [23120193] | **Trần Kim Yến** | Data Analyst - Brand & Trust Analysis |
| [23120206] | **Ngô Thuận An** | Data Analyst - Pricing & Consumer Behavior |

---

## III. Cấu trúc thư mục (Project Structure)
Dự án được tổ chức theo tiêu chuẩn khoa học để dễ dàng quản lý và tái sử dụng:

```text
LAB01/
├── dashboard/                              # Ứng dụng Dashboard trực quan hóa
│   ├── main.py                             # Script chính khởi chạy Dashboard
│   └── utils.py                            # Các hàm bổ trợ xử lý hiển thị
├── data/
│   ├── tiki_cosmetics_raw.csv              # Dữ liệu thô (JSON/CSV) vừa cào từ API Tiki
│   └── tiki_cosmetics_processed.csv        # Dữ liệu sạch đã qua chuẩn hóa và làm sạch
├── scripts/
│   ├── crawling.ipynb                      # Thu thập dữ liệu
│   └── preprocessing.ipynb                 # Quy trình làm sạch và Feature Engineering
├── docs/
│   ├── Lab 01.pdf                          # Đề bài và yêu cầu từ GV
│   └── Report.docx                         # Báo cáo
├── notebooks/                              # Các phân tích chi tiết của từng thành viên
│   ├── DataVisualization _ 23120192.ipynb
│   ├── DataVisualization _ 23120193.ipynb
│   ├── DataVisualization _ 23120201.ipynb
│   └── DataVisualization _ 23120206.ipynb
├── requirements.txt                        # Danh sách các thư viện cần cài đặt
├── README.md                               # Tổng quan dự án và hướng dẫn sử dụng
└── .gitignore                              # Loại bỏ các file nặng và venv khi push Git
```

---

## IV. Quy trình xử lý dữ liệu (Pipeline)

### 1. Thu thập dữ liệu (Data Collection)
* **Kỹ thuật:** Sử dụng `Requests` để tương tác trực tiếp với Tiki API (`v2/products`).
* **Chiến lược:** Kết hợp quét theo danh mục (Category) và từ khóa (Keywords) để vượt qua giới hạn hiển thị của sàn.
* **Quy mô:** Thu thập dữ liệu đa chiều gồm giá cả, xuất xứ, lượt bán, và các chỉ số niềm tin.

### 2. Tiền xử lý & Làm sạch (Data Preprocessing)
* **Xử lý Missing Data:** Suy luận xuất xứ dựa trên cờ `is_imported` và thông tin thương hiệu.
* **Chuẩn hóa (Normalization):** Mapping hàng trăm biến thể tên quốc gia về danh mục chuẩn bằng Dictionary.
* **Feature Engineering:** Tạo các chỉ số mới phục vụ phân tích:
  * Tỉ lệ phản hồi: Review_Ratio = Review_Count / Sold_Count
  * Doanh thu ước tính: Estimated_Revenue = Price * Sold_Count
  * Phân loại nhóm hàng: Skincare, Makeup, Body Care, Fragrance...

---

## V. Công nghệ sử dụng
* **Ngôn ngữ:** Python (v3.10+)
* **Thư viện:** * `Pandas`, `Numpy`: Xử lý và tính toán dữ liệu bảng.
  * `Requests`: Thu thập dữ liệu từ Web API.
  * `Matplotlib`, `Seaborn`: Trực quan hóa dữ liệu.
  * `Tqdm`: Theo dõi tiến độ chạy code thời gian thực.

---

## VI. Hướng dẫn chạy dự án
1. **Cài đặt thư viện:**
   ```bash
   pip install pandas requests tqdm numpy matplotlib seaborn
   ```
2. **Thu thập dữ liệu:** Chạy file `scripts/crawling.ipynb`. File thô sẽ được lưu tại `data/`.
3. **Làm sạch dữ liệu:** Chạy file `scripts/preprocessing.ipynb`. File sạch sẽ xuất hiện tại `data/`.
4. **Xem kết quả:** Mở các file trong thư mục `notebooks/` để xem biểu đồ và nhận xét chuyên sâu.
