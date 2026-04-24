# Machine Learning Dashboard - Mỹ phẩm Tiki

Module này là dashboard Machine Learning độc lập trong đồ án Lab01, gồm 2 phân tích chính:

- Module 1 (Regression): dự đoán lượt bán (`sold_count`)
- Module 2 (Clustering): phân khúc thị trường bằng K-Means

Trang này hướng dẫn cách chạy local, train model, deploy Render, và xử lý lỗi thiếu artifact.

## 1. Cấu trúc module

```text
machine learning/
	app.py
	data_loader.py
	train_model1.py
	train_model2.py
	pages/
		page_ml_regression.py
		page_ml_clustering.py
	ml_models/
		*.joblib
		*.csv
		*.json
```

`ml_models/` là thư mục rất quan trọng. Dashboard đọc artifact trong thư mục này để hiển thị kết quả.

## 2. Chạy local

Từ thư mục gốc project:

```bash
pip install -r requirements.txt
cd "machine learning"
python app.py
```

App chạy ở địa chỉ:

- `http://127.0.0.1:8051`

## 3. Train model (khi cần)

Thông thường repo đã có artifact sẵn. Chỉ train lại khi:

- đổi data đầu vào
- đổi feature engineering
- đổi logic model

Lệnh train từ thư mục `machine learning`:

```bash
python train_model1.py
python train_model2.py
```

Sau khi train xong, cần kiểm tra `ml_models/` đã có đầy đủ:

- `model1_regressor.joblib`
- `model1_metrics.json`
- `model1_feature_meta.json`
- `model1_predictions.csv`
- `model2_kmeans.joblib`
- `model2_pca.joblib`
- `model2_scaler.joblib`
- `model2_metrics.json`
- `model2_profile.json`
- và các file `.csv/.json` phụ trợ khác

## 4. Deploy Render

Repo đã sử dụng `render.yaml` để deploy 2 web service.

Service ML được khai báo:

- `rootDir: "machine learning"`
- start command: `gunicorn app:server --bind 0.0.0.0:$PORT`

Lượt đầu:

1. Push code lên GitHub
2. Trên Render chọn `New +` -> `Blueprint`
3. Chọn repo, Render đọc `render.yaml` và tạo service

Từ lần sau chỉ cần:

```bash
git push origin main
```

Nếu bật `autoDeploy: true` thì Render sẽ tự deploy lại sau mỗi lần push.

## 5. Lỗi thường gặp và cách sửa

### Lỗi: "Chưa có model - chạy train_model*.py trước"

Nguyên nhân thường gặp trên server:

- Artifact trong `ml_models/` chưa được push lên git
- Các file `.json` bị `.gitignore` chặn

Checklist sửa nhanh:

1. Đảm bảo `.gitignore` có allow rule:
	 - `!machine learning/ml_models/*.json`
2. Kiểm tra git đã track artifact:
	 - `git ls-files "machine learning/ml_models"`
3. Add + commit + push lại
4. Chờ Render build xong và vào lại trang ML

### Lỗi: deploy xong nhưng vào trang chậm

Free plan của Render có cold start. Lần đầu truy cập có thể mất 30-60 giây.

## 6. Ghi chú vận hành

- Không cần train lại trên Render nếu artifact đã đủ trong repo.
- Khi train lại local, nhớ commit artifact mới trong `ml_models/`.
- Nếu đổi schema data, phải train lại cả Module 1 và Module 2 để đồng bộ.

## 7. Link sau deploy

- ML Dashboard: `https://tiki-ml-dashboard.onrender.com`

Nếu đổi tên service trên Render thì URL cũng đổi theo.
