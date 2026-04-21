# Kế hoạch áp dụng Machine Learning (Solo) — Nhóm 05

> **Bài toán chung:** Phân tích sự ưu tiên của người tiêu dùng Việt Nam đối với mỹ phẩm trong và ngoài nước trên Tiki
> **Dataset:** `tiki_cosmetics_processed.csv` — 7,179 sản phẩm × 36 cột (đã sạch, 19 engineered features)
> **Phạm vi:** 1 người thực hiện — áp dụng **2 mô hình** + **chiến lược đánh giá đầy đủ**
> **Mục tiêu ML:** Điểm cộng tối đa +10% — tập trung vào ý nghĩa phân tích, trực quan hóa kết quả, **và đặc biệt là cách đánh giá model một cách nghiêm túc**.

---

## 1. Chọn 2 mô hình — Tại sao?

Khi chỉ áp dụng 2 mô hình, nguyên tắc chọn là:
1. **Khác paradigm:** 1 supervised + 1 unsupervised → cho thấy breadth kiến thức ML
2. **Kể chuyện nhất quán:** 2 model bổ sung cho nhau, không trùng insight
3. **Doable solo:** tổng effort ~ 3–5 ngày code + đánh giá

### Mô hình 1: Regression — Dự đoán `sold_count` (supervised)
*Câu hỏi:* **"Yếu tố nào quyết định một sản phẩm mỹ phẩm bán chạy? Xuất xứ có thực sự ảnh hưởng không?"*

### Mô hình 2: Clustering — K-Means phân khúc sản phẩm (unsupervised)
*Câu hỏi:* **"Thị trường mỹ phẩm Tiki tự nhiên chia thành bao nhiêu phân khúc? Nội/ngoại phân bố thế nào trong mỗi phân khúc?"*

**Hai model kết hợp = một câu chuyện trọn vẹn:**
- Model 1 giải thích **mối quan hệ nhân quả** (feature → lượt bán)
- Model 2 cung cấp **góc nhìn tổng thể** (phân khúc có sẵn trong dữ liệu)
- Cùng nhau trả lời được: *"Ở mỗi phân khúc, yếu tố bán chạy có khác nhau không?"*

---

## 2. MODEL 1 — Regression: Dự đoán lượt bán

### 2.1. Bài toán
- **Target:** `log1p(sold_count)` — dùng log vì phân phối cực lệch (median=1, max=553,516)
- **Features đề xuất (≈ 12 cột):**

| Nhóm | Feature | Lý do chọn |
|---|---|---|
| Giá | `price` (log1p), `discount_rate`, `price_segment` | Pricing là biến trực tiếp |
| Chất lượng | `rating`, `review_count` (log1p), `review_ratio` | Tín hiệu tin tưởng |
| Xuất xứ | `origin_class_corrected` (nội/ngoại), `product_type` | Phần cốt lõi của bài toán |
| Uy tín | `is_official_store`, `tiki_verified`, `has_authentic_badge` | Badge ảnh hưởng mua hàng |
| Brand | `brand_name` (top 20 + "Other") | High cardinality → cần gộp |

- **Lưu ý đặc biệt:**
  - Loại bỏ sản phẩm `sold_count = 0` khỏi tập train? → **KHÔNG**, giữ lại để model học được "sản phẩm ế"
  - Loại `is_extreme_outlier = True`? → **CÓ**, để giảm noise (khoảng ~50 sản phẩm)
  - `rating = 0` có nghĩa là "chưa có đánh giá" → tạo thêm feature `has_rating = (rating > 0)`

### 2.2. Mô hình thử nghiệm

| Model | Vai trò | Siêu tham số cần chỉnh |
|---|---|---|
| **DummyRegressor (mean)** | **Baseline bắt buộc** — so sánh cho thấy ML có giá trị | (none) |
| **LinearRegression** | Baseline diễn giải được (coefficients) | (none) |
| **Ridge** | Linear + regularization → ổn định hơn | `alpha` (1, 10, 100) |
| **RandomForestRegressor** | Bắt non-linear + interaction | `n_estimators=200`, `max_depth=10` |
| **GradientBoostingRegressor** | Thường cho accuracy tốt nhất | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4` |

---

## 3. MODEL 2 — Clustering: Phân khúc sản phẩm

### 3.1. Bài toán
- **Không có target** — tìm nhóm tự nhiên trong dữ liệu
- **Features dùng (6 features đã scale):**
  - `log(price)`, `rating`, `review_ratio`, `discount_rate`, `log(sold_count + 1)`, `tiki_verified`
  - Tất cả scale về `StandardScaler()` — *bắt buộc* với K-Means
- **Loại outlier trước khi cluster:** dùng `is_extreme_outlier = False`

### 3.2. Mô hình
- **K-Means** với `k ∈ {3, 4, 5, 6, 7, 8}` — chọn K tối ưu bằng Elbow + Silhouette
- Optional: chạy thêm **Agglomerative Clustering** để so sánh kết quả có ổn định không

---

## 4. Chiến lược đánh giá (Phần quan trọng nhất)

Đánh giá ML **không chỉ là một con số metric** — đó là một quy trình gồm nhiều lớp:

```
       [ Dữ liệu chất lượng ]
              ↓
     [ Split hợp lý ]
              ↓
   [ Baseline so sánh ] ← mình có hơn baseline không?
              ↓
   [ Metric định lượng ] ← số cụ thể trên test set
              ↓
   [ Cross-Validation ] ← metric có ổn định không?
              ↓
   [ Residual / Cluster Validity ] ← model SAI ở đâu?
              ↓
   [ Robustness checks ] ← đổi seed/split có giữ kết quả không?
              ↓
   [ Business Validation ] ← insight có hợp lý ngành không?
```

### 4.1. Đánh giá Model 1 (Regression)

#### (a) Split data
- **Train/Validation/Test = 70/15/15**, stratify theo `popularity_tier` để giữ phân phối
- **Random seed cố định** (`random_state=42`) để reproducible
- Test set **chỉ dùng 1 lần** ở cuối — tuyệt đối không dùng để tuning

#### (b) Baseline bắt buộc
Tính metric cho 2 baseline trước khi đánh giá model chính:
1. **DummyRegressor(strategy="mean")** — dự đoán luôn là mean
2. **Naive Rule:** "sản phẩm có rating cao + giá thấp thì bán chạy" (viết rule thủ công)

→ Nếu model ML **không hơn baseline** thì toàn bộ bài toán ML vô nghĩa. Báo cáo phải nêu rõ % cải thiện.

#### (c) Metric chính

| Metric | Giải thích | Scale |
|---|---|---|
| **R²** | % biến thiên target mà model giải thích được | Log scale |
| **RMSE** | Sai số trung bình bình phương | Cả log + gốc |
| **MAE** | Sai số trung bình tuyệt đối | Cả log + gốc |
| **MAPE** (cẩn thận) | Sai số % — chỉ tính cho sold_count > 10 | Gốc |
| **Spearman correlation** | Có giữ đúng thứ hạng sản phẩm không? | - |

Báo cáo **cả RMSE trên log scale và trên scale gốc** — vì log scale thì nhỏ nhưng chuyển về scale gốc có thể sai hàng trăm đơn vị.

#### (d) Cross-Validation
- **K-Fold = 5** trên tập train+val
- Báo cáo mean ± std của R²/RMSE
- **Mục đích:** xem metric có ổn định không. Nếu std > 20% mean → model không ổn định

#### (e) Learning Curves (phát hiện overfitting)
Vẽ đường "training score" và "validation score" theo `train_size`:
- Nếu 2 đường xa nhau → overfit → giảm `max_depth`
- Nếu 2 đường chạm đáy thấp → underfit → tăng complexity
- Dùng `sklearn.model_selection.learning_curve`

#### (f) Residual Analysis (bắt buộc)
Sau khi predict, vẽ:
1. **Residual plot:** `(y_pred vs residual)` — nếu có pattern → model bỏ sót feature
2. **Q-Q plot của residuals** — residual có phải normal không?
3. **Top 20 sản phẩm model sai nhiều nhất** — xem thử chúng có gì đặc biệt (brand lạ? outlier chưa loại?)

#### (g) Feature Importance Reliability
- Không tin 1 lần tính feature importance
- Dùng **Permutation Importance** (`sklearn.inspection.permutation_importance`) với `n_repeats=10`
- Báo cáo top features với **error bar** (std)

#### (h) Robustness Check
- Chạy lại với 3 random seeds khác nhau ({42, 7, 2024}) → R² có dao động trong ± 0.05 không?
- Nếu dao động lớn → model fragile

#### (i) Business Validation
Tạo 3 "test case nhân tạo":
- Sản phẩm giá 200K, rating 4.8, có Tiki Verified, brand Cocoon → model dự đoán ~ X
- Sản phẩm giá 2M, rating 0 (chưa có), brand lạ → model dự đoán ~ Y
- Sản phẩm giá 500K, rating 3, discount 50% → model dự đoán ~ Z

Check logic: X > Y? X > Z? Nếu không đúng thứ tự → model bị vô lý → điều tra.

#### (j) Interpret "Xuất xứ có thực sự ảnh hưởng?"
- **Chiến lược chính:** Train riêng 2 model — một cho sản phẩm nội, một cho ngoại → so sánh feature importance
- Nếu top features giống nhau → xuất xứ không phải yếu tố quyết định
- Nếu top features khác nhau → insight sâu sắc cho bài toán chung
- Kết hợp với coefficient của dummy variable `origin_class_corrected` trên model full: hệ số có significant không?

---

### 4.2. Đánh giá Model 2 (Clustering)

Clustering khó hơn vì **không có ground truth**. Phải dùng cả metric **nội tại** + **bên ngoài** + **nghiệm toán**.

#### (a) Chọn K tối ưu
Chạy K-Means với k = 2..10, ghi lại 3 chỉ số:

| Chỉ số | Cách diễn giải | Hướng tốt |
|---|---|---|
| **Inertia** (within-cluster SSE) | Tổng khoảng cách bình phương đến tâm | Giảm dần, tìm "khuỷu tay" (elbow) |
| **Silhouette Score** | [-1, 1] — cluster có tách biệt không | Cao (>0.25 là chấp nhận được, >0.5 là tốt) |
| **Davies-Bouldin Index** | Tỷ lệ intra/inter cluster | Thấp càng tốt |

**Ba chỉ số phải đồng thuận** — nếu elbow chọn K=4 nhưng silhouette chọn K=6, ghi rõ trong báo cáo và chọn theo ý nghĩa business (ít cluster hơn thường dễ diễn giải hơn).

#### (b) Cluster Stability
- Chạy K-Means 10 lần với `random_state` khác nhau
- Dùng **Adjusted Rand Index** (ARI) giữa các lần chạy → ARI > 0.8 mới coi là stable
- Nếu cluster assignment thay đổi nhiều → K chưa phù hợp hoặc cần nhiều data hơn

#### (c) So sánh với "nhóm mặc định"
- Các cluster tìm được có khác "phân loại theo `product_type`" không?
- Tính ARI giữa `cluster_label` và `product_type` — nếu ARI ~ 1 → ML chỉ tái phát hiện product_type, **không có insight mới**
- Nếu ARI ~ 0.2 → cluster thể hiện chiều **khác** (ví dụ: giá × chất lượng)

#### (d) Profile từng cluster (Business Validation)
Với mỗi cluster, tính:
- Mean/median của tất cả feature
- % nội / ngoại
- Top 3 product_type phổ biến
- Top 3 country origin
- Top 3 brand

→ Đặt tên business cho mỗi cluster, ví dụ:
- *Cluster 1: "Cao cấp ngoại nhập — trust cao"* (n=1200, giá mean 1.2M, rating 4.8, 95% ngoại, phần lớn brand Pháp/Nhật)
- *Cluster 4: "Bình dân nội địa — ít engagement"* (n=900, giá mean 150K, rating 3.9, 80% nội, brand Cocoon dẫn đầu)

**Nếu không đặt tên được → cluster không có ý nghĩa → chạy lại với K khác.**

#### (e) Visual Validation
- PCA giảm chiều xuống 2D → vẽ scatter, màu theo cluster
- Xem trực quan: các cluster có tách nhau không hay chồng lẫn?
- t-SNE là option bổ sung nhưng không bắt buộc (dễ gây hiểu nhầm về khoảng cách)

#### (f) So sánh với Agglomerative
- Chạy `AgglomerativeClustering(n_clusters=K)` với cùng K
- So sánh 2 phân cụm bằng ARI
- Nếu 2 phương pháp đồng thuận → kết quả tin cậy

---

### 4.3. Đánh giá chéo 2 model (Liên kết)

Sau khi có cả 2 model:

1. **Cluster × Regression:** Trong mỗi cluster, model 1 có R² khác nhau không? Có cluster nào model 1 kém dự đoán hơn? → insight về "cluster khó dự đoán"

2. **Regression trên từng cluster:** Train Model 1 riêng cho từng cluster → feature importance có khác nhau? Ví dụ: trong cluster "Cao cấp ngoại", có thể `price` ít quan trọng hơn `brand_name` — đây là insight rất mạnh.

3. **Bảng tổng hợp cuối báo cáo:**

| Cluster | Tên | Kích thước | % nội | % ngoại | R² Model 1 | Top feature ảnh hưởng sold_count |
|---|---|---|---|---|---|---|
| 1 | Cao cấp ngoại | 1,200 | 5% | 95% | 0.52 | brand, rating |
| 2 | Bình dân nội | 900 | 80% | 20% | 0.41 | price, discount |
| ... | | | | | | |

---

## 5. Cấu trúc code & dashboard

### 5.1. Folder

```
TQH-LAB010-N5/
├── dashboard/
│   ├── pages/
│   │   └── page_ml_regression.py         # tab ML mới (Model 1 — Regression)
│   ├── ml_models/
│   │   ├── model1_regressor.joblib       # pipeline đã train
│   │   ├── model1_metrics.json           # val/test/CV metrics
│   │   ├── model1_feature_meta.json      # brand/type/origin classes
│   │   ├── model1_predictions.csv        # y_true vs y_pred (test set)
│   │   ├── model1_feature_importance.csv # permutation importance
│   │   ├── model1_learning_curve.csv     # train/val R² vs size
│   │   └── model1_partial_dependence.csv # PDP price effect
│   ├── assets/
│   │   └── stylePageML.css               # style riêng cho page ML
│   ├── train_model1.py                   # script huấn luyện + đánh giá
│   └── app.py                            # thêm route /ml
└── docs/
    └── ML_Plan.md                        # file này
```

### 5.2. Layout `page5_ml.py` (3 sub-section)

```
──────────────────────────────────────────
[Tab ML] Ứng dụng Machine Learning
──────────────────────────────────────────

SECTION 1: KPI Overview (4 cards)
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Regression   │ Clustering   │ Baseline gap │ Số phân khúc │
│ R² = 0.52    │ Silh. = 0.38 │ +48% vs mean │ K = 5        │
└──────────────┴──────────────┴──────────────┴──────────────┘

SECTION 2A: Regression — Dự đoán lượt bán
  - Bar chart: Top 10 Permutation Importance (có error bar)
  - Scatter: Actual vs Predicted (test set)
  - Residual plot
  - [INTERACTIVE] Form "What-if": nhập giá/brand/rating → predict lượt bán
  - Line: Partial Dependence của `price`

SECTION 2B: Clustering — Phân khúc thị trường
  - Line chart: Elbow (inertia theo K) + Silhouette theo K
  - Scatter 2D PCA, màu theo cluster
  - Radar chart: profile mỗi cluster (6 features đã scale)
  - Bảng summary: tên cluster, size, % nội/ngoại, top brand, top country
  - [INTERACTIVE] Slider chọn K (3–8) → cluster thay đổi real-time

SECTION 3: Kết luận liên kết
  - Bảng "Cluster × Regression quality" (mục 4.3 ở trên)
  - Text block 5 insight kinh doanh
  - Text block hạn chế + hướng phát triển
```

---

## 6. Timeline (Solo, 7–10 ngày)

| Ngày | Việc |
|---|---|
| **D1** | Viết notebook `ml_training.ipynb`: load data, preprocess, ColumnTransformer, split |
| **D2** | Train Model 1 (5 variants), cross-validate, chọn model tốt nhất |
| **D3** | Evaluation Model 1: learning curves, residual analysis, permutation importance, robustness |
| **D4** | Train + tune Model 2: Elbow, Silhouette, stability test |
| **D5** | Profile clusters, đặt tên business, vẽ radar/PCA |
| **D6** | Đánh giá chéo 2 model (mục 4.3), tạo bảng tổng hợp |
| **D7** | Code `page5_ml.py` + tích hợp vào `app.py` |
| **D8** | Viết section ML trong `Report_Lab01.docx` (2-3 trang) |
| **D9** | Test dashboard, chụp hình cho báo cáo |
| **D10** | Buffer / polish |

---

## 7. Checklist hoàn thành

### Code
- [ ] 2 models train & save thành công (`.joblib`)
- [ ] Evaluation results JSON có đầy đủ metrics
- [ ] Tab ML trong dashboard hiển thị và tương tác được
- [ ] Có ≥ 1 widget interactive (form what-if / slider K)

### Evaluation quality
- [ ] Có baseline (dummy) để so sánh Model 1
- [ ] Cross-validation với mean ± std
- [ ] Learning curves đã vẽ
- [ ] Residual analysis đã làm
- [ ] Permutation importance thay vì feature importance mặc định
- [ ] Robustness check 3 seeds
- [ ] Business validation với 3 test case
- [ ] Silhouette + Elbow + Davies-Bouldin (Model 2)
- [ ] Cluster stability ARI ≥ 0.8
- [ ] So sánh cluster với product_type (ARI)
- [ ] Profile + tên business cho mỗi cluster

### Báo cáo
- [ ] Giải thích chọn 2 model (vì sao không 1 hay 4)
- [ ] Kết quả cả 2 model + metrics
- [ ] Ít nhất 2 biểu đồ evaluation (không chỉ feature importance)
- [ ] 5 insight kinh doanh kết nối 2 model với bài toán chung
- [ ] Liệt kê hạn chế thẳng thắn (data leak? features bỏ sót?)

---

## 8. Tránh các bẫy thường gặp

| Bẫy | Biểu hiện | Cách tránh |
|---|---|---|
| **Data leakage** | R² cao bất thường (> 0.95) | Kiểm tra có feature nào tính từ target không (vd: `review_ratio` nếu tính từ sold_count → leak) |
| **Overfit** | Train R² = 0.9, Test R² = 0.3 | Dùng `max_depth` nhỏ, regularization, learning curves |
| **Chọn K bừa** | K=5 vì "số đẹp" | Luôn có biểu đồ Elbow + Silhouette biện minh |
| **Cluster trùng với product_type** | ARI(cluster, product_type) > 0.8 | Loại product_type khỏi features clustering |
| **Quên baseline** | Báo cáo R²=0.4 mà không biết mean predictor R² = 0.38 | Luôn tính DummyRegressor trước |
| **Tin vào 1 random seed** | "Seed 42 thì R² = 0.55" | Chạy 3 seeds và báo cáo mean ± std |
| **Metric đẹp nhưng vô nghĩa** | MAPE 10% nhưng chỉ vì target lớn | Vẽ residual plot để thấy model sai ở đâu |

---

## 9. Câu hỏi có thể giảng viên hỏi khi vấn đáp

Chuẩn bị câu trả lời cho:
1. "Vì sao chọn K=5 mà không phải K=4?" → dùng biểu đồ Elbow + Silhouette
2. "R²=0.52 có cao hay thấp?" → "Cao hơn baseline 48%, và với dữ liệu e-commerce skewed thế này, 0.5 là hợp lý"
3. "Nếu em thêm dữ liệu thì model có tốt hơn không?" → dùng learning curve để trả lời
4. "Làm sao biết cluster có ý nghĩa kinh doanh?" → profile + tên business + so sánh ARI với product_type
5. "Model có bị overfit không?" → learning curve + train/val gap
6. "Xuất xứ có thực sự quan trọng với lượt bán?" → permutation importance + train riêng 2 nhóm

---

## 10. Ý nghĩa chốt

Với chỉ **2 model + 1 chiến lược đánh giá nghiêm túc**, bài sẽ có **giá trị hơn** nhóm có 5 model mà không ai đánh giá chúng. Điểm cộng tối đa +10% phụ thuộc **40% vào chọn model đúng, 60% vào cách đánh giá và trực quan hóa kết quả**.

Câu chuyện cuối cùng gói gọn:
> *"Mình tìm thấy 5 phân khúc tự nhiên trong thị trường mỹ phẩm Tiki. Trong đó, 3 phân khúc là nội địa, 2 là ngoại. Mô hình dự đoán lượt bán giải thích được 52% biến thiên — **hơn baseline 48%**. Yếu tố ảnh hưởng mạnh nhất đến lượt bán là `rating` và `brand_name`, trong khi xuất xứ chỉ đứng thứ 5 — điều này thách thức giả định ban đầu rằng 'nội/ngoại' là yếu tố quyết định."*

Đây là insight có giá trị nghiệm toán + business, không phải bản báo cáo số.
