# Mỹ phẩm Tiki - Machine Learning Dashboard 🧠

Dự án này là module Machine Learning độc lập thuộc hệ thống phân tích dữ liệu Mỹ phẩm Tiki (tháng 3/2026). Module tập trung vào hai bài toán chính: Dự đoán lượt bán (Regression) và Phân cụm thị trường (Clustering).

## 🚀 Hướng dẫn chạy ứng dụng

**1. Cài đặt thư viện:**
Mở terminal và cài đặt các thư viện cần thiết:
```bash
pip install dash pandas numpy scikit-learn xgboost plotly joblib
```

**2. Chạy Dashboard:**
Di chuyển vào thư mục `dashboardML` và chạy file `app.py`:
```bash
cd dashboardML
python app.py
```

**3. Truy cập ứng dụng:**
Mở trình duyệt và truy cập vào địa chỉ: **http://127.0.0.1:8051**

---

## 📊 Kịch bản Demo (Dành cho Module 1 - What-If Analysis)

Trong thanh navbar, chọn **Module 1 · Regression**, cuộn xuống **Section 03**. Sử dụng thanh tìm kiếm (Auto-fill) để gõ tên và chọn các sản phẩm sau nhằm chứng minh độ chính xác của mô hình ở nhiều góc độ:

### Kịch bản 1: Sản phẩm bán chạy (Top Performer)
Mô hình dự đoán sát thực tế và định vị chính xác vị thế "Top Performer" của sản phẩm trên thị trường.
*   **Son dưỡng môi chuyên biệt dành cho môi khô, nứt nẻ Mentholatum Medi Lip** (Giá: 59,000đ | Sold: 498) -> *Mô hình dự đoán 492 (rất chính xác).*
*   **Nước tẩy trang cho da mụn Acnes Micellar Water 200ml** (Giá: 86,000đ | Sold: 197)

### Kịch bản 2: Sản phẩm trung bình (Average)
Sản phẩm kén khách hoặc ngách, mô hình không bị "ảo tưởng" sức mạnh mà dự đoán rất sát mức bán thấp.
*   **Bộ Sữa Rửa Mặt Ohui Trắng Ohui Extreme White Cleansing Foam** (Giá: 900,000đ | Sold: 6) -> *Mô hình dự đoán 6.*
*   **Bộ 3 sản phẩm ngăn ngừa và thâm Some By Mi AHA-BHA-PHA** (Giá: 910,000đ | Sold: 6)

### Kịch bản 3: Giảm giá khủng (Demo Burn Ratio)
Để demo tính năng phân tích **Burn Ratio (Tỷ lệ đốt Margin)**. Hệ thống sẽ cảnh báo đỏ vì mức giảm giá quá sâu so với trung bình ngành.
*   **Xịt Dưỡng Trắng Da Toàn Thân White Conc Body Lotion C II 245 mL** (Khuyến mãi 70%) -> *Cảnh báo: ĐỐT QUÁ NHIỀU.*
*   **Nước Tẩy Trang Dưỡng Ẩm Chiết Xuất Hạt Ý Dĩ Reihaku Hatomugi** (Khuyến mãi 56%) -> *Cảnh báo: Cao hơn 75% ngành.*

### Kịch bản 4: Bán nguyên giá (Không khuyến mãi)
Sản phẩm tự tin bán nguyên giá không cần giảm, Burn Ratio tốt.
*   **SỮA TẮM SHIKIORIORI CHIẾT XUẤT QUẢ HỒNG DƯỠNG ẨM** (Giá: 169,000đ | Khuyến mãi 0%) -> *Hiển thị: ✅ Không có KM.*

---

## 📁 Cấu trúc thư mục chính cần Push lên GitHub
*   `app.py`: File khởi động chính của ứng dụng.
*   `pages/`: Chứa giao diện của 2 module (`page_ml_regression.py` và `page_ml_clustering.py`).
*   `assets/`: Chứa file CSS (`stylePageML.css`, `stylePage3.css`).
*   `ml_models/`: Thư mục chứa các mô hình đã train (`.joblib`) và file thông tin cấu hình (`.json`).
*   `data_loader.py`: File xử lý dữ liệu đầu vào.
*   `train_model1.py` & `train_model2.py`: Source code train AI (Tuỳ chọn push nếu cần nộp source train).
