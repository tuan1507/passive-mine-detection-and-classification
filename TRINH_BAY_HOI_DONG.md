# Phát hiện và phân loại mìn thụ động bằng học máy — Đánh giá lại trung thực và ứng dụng hỗ trợ rà phá

> Tài liệu trình bày trước hội đồng · Đồ án tốt nghiệp VKU 2026
> Mọi con số trong tài liệu lấy từ output đã lưu của các notebook `Notebook/01…07` và từ `Notebook/product_v3`.

---

## 1. Bài toán là gì?

Một cảm biến từ trường **thụ động** (FLC, *fluxgate*) được rê trên mặt đất. Khi gặp vật kim loại chôn ngầm, từ trường Trái Đất bị nhiễu và cảm biến cho ra một **điện áp dị thường V**. Gọi là "thụ động" vì cảm biến **không phát sóng**: nó chỉ đo sự biến dạng của từ trường có sẵn, nên không thể kích nổ ngòi nổ cảm ứng.

Mỗi lần đo cho ta 3 thông tin:

| Ký hiệu | Ý nghĩa | Kiểu |
|---|---|---|
| **V** | Điện áp dị thường từ (0–10,6 V, chuẩn hoá về [0,1]) | Số thực |
| **H** | Độ cao đầu dò so với mặt đất (0–20 cm, chuẩn hoá) | Số thực |
| **S** | Loại đất: khô/ẩm × cát/mùn/vôi (6 loại) | Hạng mục |

Từ ba thông tin đó, hệ thống phải trả lời **hai câu hỏi**:

1. **Phát hiện (nhị phân):** dưới tín hiệu này **có mìn hay không**?
2. **Phân loại (5 lớp):** nếu có thì là loại gì: *Không mìn · Mìn chống tăng · Mìn chống bộ binh · Mìn chống bộ binh dạng bẫy · Mìn M14*.

Đầu ra cuối cùng của đồ án là **một danh sách xếp hạng ưu tiên đào** cho toàn bộ tín hiệu của một khu vực khảo sát.

---

## 2. Vì sao có bài toán này?

### 2.1. Lý do thực tế

- Bom mìn, vật nổ sót lại sau chiến tranh vẫn đang ô nhiễm một phần lớn diện tích Việt Nam và Lào, và vẫn gây thương vong hằng năm. *(Khi đưa lên slide, nên trích số liệu chính thức của VNMAC hoặc NRA Lào.)*
- Máy dò kim loại báo **rất nhiều tín hiệu**, nhưng **phần lớn là kim loại vụn** (mảnh bom, đinh, vỏ hộp). Tín hiệu nào cũng phải được đào kiểm tra thủ công, nên việc này **tốn thời gian, tốn tiền và nguy hiểm** cho người đào.
- Nếu biết trước **tín hiệu nào nhiều khả năng là mìn thật**, đội rà phá có thể **xử lý các điểm nguy hiểm trước**, bố trí người và thiết bị phù hợp với loại mìn (chống tăng khác chống bộ binh), và giảm thời gian các điểm nguy hiểm còn nằm ngoài hiện trường.

### 2.2. Lý do khoa học: một kết quả "quá đẹp" cần được kiểm chứng

Bài báo gốc *Yilmaz, Kahraman & Söyler (2018), "Passive Mine Detection and Classification Method Based on Hybrid Model", IEEE Access 6:47870–47888* công bố:

- **98,2%** độ chính xác phát hiện mìn;
- **85,8%** độ chính xác phân loại 5 loại mìn.

Đây là những con số rất cao cho một bộ dữ liệu chỉ có **338 mẫu và 3 đặc trưng**. Nếu đem con số đó ra thực địa mà nó không có thật, **người đào mìn sẽ phải trả giá**. Vì vậy câu hỏi nghiên cứu của đồ án là:

> **Nếu đánh giá đúng phương pháp (không rò rỉ dữ liệu), mô hình thực sự đạt bao nhiêu, và nên dùng nó vào việc gì cho an toàn?**

---

## 3. Dữ liệu

- **UCI Land Mines Dataset**, gồm 338 mẫu sau khi làm sạch, không có giá trị khuyết và không có dòng trùng.
- Phân loại 5 lớp **gần cân bằng**: 71 / 70 / 66 / 66 / 65 mẫu. Đoán mò đạt khoảng 21%.
- Phát hiện **mất cân bằng**: 267 có mìn và 71 không mìn. Một mô hình **luôn đoán "có mìn"** đã đạt **79,0%** mà không học được gì. Vì thế đồ án đo **recall của lớp "có mìn"** chứ không chỉ accuracy.
- Dữ liệu đã được chuẩn hoá [0,1] trên toàn bộ tập từ trước khi công bố, và không có dữ liệu thô. Đây là một giới hạn được ghi nhận.

---

## 4. Phương pháp: 7 giai đoạn

| GĐ | Notebook | Câu hỏi | Kỹ thuật |
|---|---|---|---|
| 1 | `01_kham_pha` | Dữ liệu trông như thế nào? | EDA, tương quan Spearman |
| 2 | `02_tai_hien_paper` | Có tái hiện được bài báo không? Con số đến từ đâu? | k-NN có trọng số, tối ưu bằng **Differential Evolution** (thay GA); 4 giá trị k × 3 độ đo |
| 3 | `03_khac_phuc_ro_ri` | **Bỏ rò rỉ thì còn bao nhiêu?** *(đóng góp chính)* | Tách train/validation/test; **nested cross-validation**; thí nghiệm A/B + Wilcoxon |
| 4 | `04_so_sanh_mo_hinh` | Mô hình hiện đại có tốt hơn không? | 9 mô hình trên cùng một khung đánh giá; Recall, F1, macro-F1, đường PR |
| 5 | `05_kiem_dinh_thong_ke` | Chênh lệch giữa các mô hình có thật không? | Friedman + Nemenyi, biểu đồ Critical Difference |
| 6 | `06_chi_phi_loi` | Bỏ sót mìn đắt hơn đào nhầm: chọn ngưỡng thế nào? | Ma trận chi phí, ngưỡng chọn trên validation |
| 7 | `07_giai_thich_model` | Mô hình dựa vào biến nào? | Permutation importance, trọng số DE, SHAP |
| 8 | `08_san_pham_core` | Lõi sản phẩm làm gì, từng bước? | Bản notebook của `core.py` + `train_final.py`, có ô đối chiếu với `core.py` |
| 9 | `09_cai_thien_do_chinh_xac` | Có nâng được phân loại từ 59,5% không? | 13 cấu hình, ablation đặc trưng, kiểm tra trên seed mới, so sánh khả năng xếp hạng |
| 10 | `10_kiem_thu_app` | App có chạy đúng không? | Logic từng trang dạng notebook + Streamlit AppTest |

**Nguyên tắc xuyên suốt:**

> *Dữ liệu nào đã tham gia chọn hoặc chỉnh mô hình (trọng số, siêu tham số, ngưỡng, scaler) thì không được dùng để chấm điểm.*

---

## 5. Kết quả chính

### 5.1. Tìm ra nguồn của 85,8% và 98,2% (GĐ2)

Bài báo tối ưu trọng số `[w_V, w_H, w_S]` bằng GA, và **chấm điểm từng bộ trọng số trên chính tập test** (Công thức 3 và Thuật toán 1). Mô hình đã "nhìn trộm đáp án". Dùng chính các bảng số của bài báo, đồ án phân rã con số công bố như sau:

| Tầng | Phân loại | Phát hiện |
|---|:---:|:---:|
| (A) k-NN chuẩn, không trọng số | 46,9% | 86,7% |
| (B) + độ đo mờ *fuzzy* | 73,5% (**+26,6**) | 93,8% (+7,1) |
| (C) + trọng số GA tối ưu **trên tập test** | **85,8%** (**+12,3**) | **98,2%** (+4,4) |

Như vậy con số công bố đến từ hai phần:
- **Fuzzy metric:** công thức nằm sau paywall, không tái lập được. Mức tăng +26,6 điểm chỉ nhờ đổi độ đo là bất thường.
- **Rò rỉ dữ liệu khi tối ưu trọng số.**

Với các độ đo chuẩn EU/MA/MI, pipeline tái hiện của đồ án sai lệch so với bài báo trung bình chỉ **2,4–5,8 điểm**, nên có thể tin pipeline này đúng.

### 5.2. Đo trực tiếp mức thổi phồng do rò rỉ (GĐ3)

Đồ án chạy 15 lần với cùng tập test, so hai cách:
- **A (rò rỉ):** tối ưu trọng số trên chính tập test.
- **B (trung thực):** tối ưu trên một tập validation riêng.

| Nhiệm vụ | A (rò rỉ) | B (trung thực) | Thổi phồng | Wilcoxon |
|---|:---:|:---:|:---:|:---:|
| Phân loại | 63,4% | 56,6% | **+6,8 điểm** | p = 0,0006 |
| Phát hiện | 93,5% | 90,4% | **+3,1 điểm** | p = 0,0014 |

Cả hai p-value đều < 0,05. Mức thổi phồng là **lỗi có hệ thống**, không phải do ngẫu nhiên.

### 5.3. Con số trung thực (nested CV, 5 fold × 3 lần lặp)

| | Bài báo công bố | **Đánh giá trung thực** | Baseline không học |
|---|:---:|:---:|:---:|
| Phát hiện | 98,2% | **90,3% ± 4,0** | 79,0% |
| Phân loại 5 lớp | 85,8% | **62,0% ± 5,1** | 21,0% |

### 5.4. So sánh 9 mô hình (GĐ4–5)

**Phát hiện** (xếp theo F1 của lớp "có mìn"):

| Mô hình | Recall (mìn) | Precision | F1 |
|---|:---:|:---:|:---:|
| CatBoost | 94,8 | 94,6 | **94,6** |
| kNN+DE (bài báo, đã bỏ rò rỉ) | 92,8 | 94,9 | 93,8 |
| LightGBM | 93,8 | 93,6 | 93,7 |
| RandomForest | 93,5 | 91,9 | 92,7 |
| … | | | |
| LogReg | 91,8 | 82,1 | 86,6 |

**Phân loại:** MLP đứng đầu với 63,3%, tiếp theo là kNN+DE 62,0%, CatBoost 58,7%, LightGBM 56,9%, …, ExtraTrees thấp nhất với 41,9%.

**Kiểm định thống kê:**
- Friedman (p ≈ 10⁻¹³ đến 10⁻¹⁷) cho thấy các mô hình **có khác nhau**.
- Nemenyi (CD = 3,10) cho thấy **nhóm dẫn đầu không tách biệt được**:
  - Phát hiện: CatBoost, LightGBM, kNN+DE, MLP, RandomForest, XGBoost.
  - Phân loại: MLP, kNN+DE, CatBoost, LightGBM.

> **Thông điệp:** với 338 mẫu, **không thuật toán nào thắng áp đảo**. Nút thắt nằm ở **dữ liệu**, không phải ở thuật toán.

### 5.5. An toàn: bỏ sót một quả mìn có thể trả giá bằng một mạng người (GĐ6)

| Kiểu lỗi | Hậu quả |
|---|---|
| Báo "không mìn" nhưng thực ra **có** (FN) | **Người đào có thể thiệt mạng** |
| Báo "có mìn" nhưng thực ra **không** (FP) | Tốn công đào một hố trống |

Đồ án chọn ngưỡng sao cho recall đạt 100% **trên validation**, rồi đo trên một **tập test tách biệt** qua 20 fold:

- Recall thực tế trên test: **97,4% ± 3,6**.
- Số tín hiệu không mìn có thể xếp xuống cuối hàng đợi: **37,3% ± 16,8**.
- Chỉ **50% số fold giữ được recall 100%**. Nửa còn lại vẫn bỏ sót ít nhất một quả mìn.
- Khi gán chi phí FN = 1000 và FP = 1, ngưỡng tối ưu giảm xuống khoảng 0,11. Mô hình trở nên "đa nghi": thà báo nhầm còn hơn bỏ sót.
- Loại đất dễ bỏ sót mìn nhất là **Ẩm & mùn** (recall 91,3%) và **Khô & cát** (91,5%).
- PR-AUC = 0,989 và ROC-AUC = 0,956.

> **Kết luận an toàn:** không một ngưỡng cố định nào **đảm bảo 0% bỏ sót** trên dữ liệu mới. Vì vậy hệ thống **chỉ được dùng để sắp thứ tự đào**, **không bao giờ dùng để bỏ qua** một tín hiệu.

### 5.6. Mô hình dựa vào đâu? (GĐ7)

Hai phương pháp độc lập (permutation importance trên mô hình cây và trọng số DE trên k-NN), cùng với SHAP, đều cho cùng một kết luận: **V (dị thường từ) chiếm 57–69% tầm quan trọng**. Điều này khớp với vật lý: vật thể kim loại càng lớn hoặc càng gần thì dị thường càng mạnh.

Thứ tự giữa H và S **không ổn định** giữa các phương pháp và giữa các fold. Vì vậy đồ án chỉ khẳng định "V quan trọng nhất", không khẳng định thứ tự H/S. Điều này cũng giải thích vì sao trọng số công bố trong bài báo cho hai thứ tự khác nhau ở hai bài toán.

---

## 6. Sản phẩm: hệ thống hỗ trợ xếp hạng ưu tiên đào

### 6.1. Hai mô hình cho hai bài toán con (GĐ9: `09_cai_thien_do_chinh_xac.ipynb`)

| Việc | Mô hình | Vì sao |
|---|---|---|
| **Phân loại loại mìn** (5 lớp) | **MLP (16-8)** trên đặc trưng theo miền: V, H, **log V**, **độ ẩm + vật liệu đất (one-hot)** | Tốt nhất trong 13 cấu hình thử; ổn định trên hai bộ chia fold khác nhau |
| **Phát hiện / xếp hạng** (P(có mìn)) | **HistGradientBoosting nhị phân** trên V, H, S | Xếp hạng tốt nhất (ROC-AUC 0,948). MLP phân loại giỏi nhưng xác suất quá tự tin (ROC-AUC 0,876) |

**Vì sao đặc trưng mới giúp?**
- Dị thường từ **suy giảm theo luật lũy thừa** của khoảng cách, nên log V làm quan hệ gần tuyến tính hơn.
- S chỉ là **mã** của 6 loại đất. Dùng S như một con số sẽ áp đặt thứ tự giả, nên tách thành độ ẩm và vật liệu.

Riêng việc tinh chỉnh siêu tham số boosting thì **không** giúp: −2,1 điểm.

**Kết quả** (CV lặp 5×3; ngưỡng chọn trên validation và đo trên test tách biệt):

| Chỉ số | Sản phẩm cũ | **Sản phẩm mới** |
|---|:---:|:---:|
| Phân loại 5 lớp | 59,5% | **71,8% ± 4,6** (Wilcoxon p = 0,001) |
| Đúng trong top-2 loại | ~79% | **88,3%** |
| Recall tại ngưỡng ưu tiên | 97,4%* | **99,4% ± 1,0** |
| Số fold giữ recall 100% | 50%* | **75%** |
| Tín hiệu không mìn được xếp cuối | 37%* | 27% |

\* Số cũ đo trên một mô hình nhị phân **khác** với mô hình app thực sự dùng, và chưa áp biên an toàn × 0,8. Bản mới đo đúng mô hình và đúng ngưỡng của sản phẩm.

Ngưỡng ưu tiên khoảng 0,097, bằng trung bình các ngưỡng chọn trên validation **× 0,8** (biên an toàn). Toàn bộ huấn luyện và đánh giá chạy trong khoảng 30 giây trên CPU (`train_final.py` hoặc `08_san_pham_core.ipynb`).

> Lưu ý trung thực: cấu hình được chọn bằng CV nên có thể lạc quan khoảng 1–2 điểm. Bộ chia fold seed 2026, chưa dùng để chọn, cho 69,3%.

### 6.2. Luồng xử lý

```
File tín hiệu cả khu vực (Excel/CSV: V, H, S, [id, x, y])
        │
        ▼
Kiểm tra đầu vào (đủ cột, là số, nằm trong [0,1], S thuộc 6 loại đất)
        │
        ▼
Mô hình phát hiện → P(có mìn)      Mô hình phân loại → loại mìn + loại khả năng thứ 2
        │
        ▼
Mức ưu tiên:  ≥ 0,5 → Rất cao · ≥ ngưỡng → Cao · ≥ ngưỡng/2 → Trung bình · còn lại → Thấp
        │
        ▼
Bảng xếp hạng (1 = đào trước) · Bản đồ rủi ro X–Y · Thống kê · Xuất Excel
```

### 6.3. Các màn hình của ứng dụng (Streamlit)

| Trang | Chức năng |
|---|---|
| Trang chủ | Nguyên tắc an toàn, các chỉ số trung thực của mô hình |
| Dự đoán hàng loạt | Tải file → bảng xếp hạng có tô màu → lọc → tải kết quả Excel |
| Bản đồ rủi ro | Mỗi tín hiệu là một điểm trên lưới khảo sát; màu = mức ưu tiên, kích thước = P(có mìn) |
| Tổng quan | Phân bố mức ưu tiên, loại vật thể, xác suất; thống kê theo loại đất |
| Thử đơn lẻ | Kéo thanh V/H/S để xem mô hình phản ứng (dùng để minh hoạ) |

Chạy thử: `streamlit run Notebook/product_v3/Home.py`, hoặc dùng Docker hay Hugging Face Space.

---

## 7. Ứng dụng của bài toán

### 7.1. Ai dùng, dùng vào lúc nào?

| Người dùng | Công việc | Hệ thống giúp gì |
|---|---|---|
| **Chỉ huy đội rà phá** | Lập kế hoạch đào mỗi ngày | Sắp **thứ tự đào**: xử lý các điểm "Rất cao/Cao" trước, khi người còn tỉnh táo và có đủ trang bị |
| **Kỹ thuật viên** | Chuẩn bị trước khi đào | Biết trước **hai loại mìn có khả năng cao nhất** (đúng trong top-2 khoảng 88%) để chọn quy trình và thiết bị (ví dụ mìn chống tăng thường có khối lượng kim loại lớn) |
| **Cán bộ kiểm soát chất lượng (QA/QC)** | Kiểm tra sau khi rà | Đối chiếu: những điểm mô hình đánh giá "Rất cao" mà đội báo là "không có gì" sẽ được **ưu tiên kiểm tra lại** |
| **Quản lý dự án** | Báo cáo, phân bổ nguồn lực | Bản đồ rủi ro và thống kê theo khu vực, theo loại đất |

### 7.2. Lợi ích kỳ vọng

- **Giảm rủi ro:** các điểm nguy hiểm nhất được xử lý sớm nhất.
- **Tiết kiệm nguồn lực:** khoảng **1/4 số tín hiệu không mìn** được xếp xuống cuối hàng đợi (recall 99,4% tại ngưỡng), nhưng **vẫn phải đào**.
- **Chuẩn hoá và lưu vết:** mỗi khu vực có một file kết quả xếp hạng, phục vụ báo cáo và kiểm tra.
- **Triển khai rẻ:** chỉ cần máy tính CPU thông thường, không cần GPU hay kết nối mạng khi chạy local.

### 7.3. Điều hệ thống **không** làm (giới hạn an toàn)

> ⚠️ Hệ thống **không xác định vùng an toàn** và **không thay thế** quy trình rà phá theo tiêu chuẩn IMAS. **Mọi tín hiệu vẫn phải được đào kiểm tra đầy đủ.** Quyết định cuối cùng thuộc về cán bộ kỹ thuật có thẩm quyền.

---

## 8. Đóng góp của đồ án

1. **Phát hiện và định lượng rò rỉ dữ liệu** trong một bài báo IEEE Access: tối ưu trọng số trên tập test gây thổi phồng +3 đến +7 điểm, có ý nghĩa thống kê.
2. **Phân rã con số công bố** thành phần do fuzzy metric (không kiểm chứng được) và phần do rò rỉ.
3. **Khung đánh giá trung thực** có thể tái sử dụng: nested CV, so sánh 9 mô hình, Friedman/Nemenyi, chọn ngưỡng trên validation.
4. **Góc nhìn chi phí lỗi bất đối xứng:** chứng minh một ngưỡng cố định không đảm bảo 0% bỏ sót, nên định hướng sản phẩm thành **công cụ xếp hạng ưu tiên** thay vì công cụ "loại bỏ tín hiệu".
5. **Giải thích mô hình bằng hai phương pháp độc lập** có đo độ ổn định, và kết quả khớp với cơ sở vật lý.
6. **Nâng phân loại từ 59,5% lên 71,8%** bằng biểu diễn dữ liệu theo vật lý (log V, tách loại đất), đồng thời chỉ ra rằng **mô hình phân loại tốt nhất không phải mô hình xếp hạng tốt nhất**. Vì vậy sản phẩm dùng hai mô hình cho hai bài toán con.
7. **Sản phẩm chạy được**, có kiểm thử tự động (`10_kiem_thu_app.ipynb`), và trung thực về giới hạn ngay trên giao diện.

---

## 9. Giới hạn và hướng phát triển

**Giới hạn**
- Dữ liệu đo trong **phòng thí nghiệm**, chỉ 338 mẫu, không có nhiễu nền từ kim loại vụn như ngoài thực địa.
- Dữ liệu đã được chuẩn hoá sẵn trên toàn tập, nên còn một mức rò rỉ nhỏ không loại bỏ được.
- Các lớp mìn nhỏ (chống bộ binh, M14) khó tách biệt.
- File mẫu `sample_batch` **chính là dữ liệu huấn luyện** gắn thêm toạ độ giả lập, nên chỉ dùng để **minh hoạ giao diện**, không dùng để đánh giá độ chính xác.

**Hướng phát triển**
- Thu **tín hiệu dạng sóng thô** và dùng 1D-CNN; biểu diễn đường cong V theo H.
- **Conformal prediction** thay cho ngưỡng cố định, để có đảm bảo thống kê về tỷ lệ bỏ sót.
- Thu thêm dữ liệu thực địa, đánh giá theo nhóm (ví dụ tách theo loại đất) để kiểm tra khả năng tổng quát.
- Tích hợp với GPS, GIS và IMSMA.

---

## 10. Gợi ý trả lời câu hỏi của hội đồng

**H: Tại sao kết quả của em thấp hơn bài báo mà vẫn gọi là đóng góp?**
Đ: Vì con số của bài báo **không có thật trên dữ liệu mới**: nó được chấm trên chính dữ liệu đã dùng để tối ưu. Trong bài toán an toàn tính mạng, một con số trung thực 90% có giá trị hơn một con số 98% bị thổi phồng. Thí nghiệm A/B và kiểm định Wilcoxon chứng minh mức thổi phồng là có hệ thống.

**H: Sao không dùng deep learning?**
Đ: Chỉ có 338 mẫu và 3 đặc trưng. Kiểm định Nemenyi cho thấy ngay cả các mô hình hiện đại cũng không tách biệt được với nhau. Nút thắt là dữ liệu, không phải mô hình. Deep learning chỉ có ý nghĩa khi có tín hiệu sóng thô (đã nêu trong hướng phát triển).

**H: Vì sao dùng hai mô hình mà không dùng một?**
Đ: Hai bài toán cần hai thứ khác nhau. Phân loại cần đúng nhãn: MLP + đặc trưng tốt nhất (71,8%). Xếp hạng cần xác suất **phân biệt được mức độ**: MLP cho rất nhiều xác suất bằng 1,0 nên không xếp hạng được (ROC-AUC 0,876), còn boosting nhị phân đạt 0,948. Notebook 09 mục 4 đo cả hai theo đúng quy trình ngưỡng của sản phẩm.

**H: Tăng từ 59,5% lên 71,8% có phải do "chọn trúng" trên CV không?**
Đ: Có một phần nhỏ, và đồ án đã kiểm tra. Đồ án chạy lại trên một bộ chia fold mới (seed 2026) chưa dùng để chọn, kết quả vẫn đạt 69,3%. Mức tăng cũng có giải thích vật lý (log V, tách loại đất). Ngoài ra, tinh chỉnh boosting thì không tăng, chứng tỏ cải thiện đến từ cách biểu diễn dữ liệu chứ không phải từ việc dò tham số.

**H: Recall 99,4% có đủ an toàn không?**
Đ: Không, và đồ án nói thẳng điều đó: chỉ 75% số fold giữ được recall 100%. Vì vậy hệ thống chỉ **sắp thứ tự đào**, không cho phép bỏ qua tín hiệu nào.

**H: Rò rỉ dữ liệu là gì, giải thích đơn giản?**
Đ: Giống như cho học sinh xem đề thi để luyện rồi chấm bằng đúng đề đó. Điểm cao nhưng không phản ánh năng lực thật. Cách đúng là luyện trên đề luyện (validation) và chấm trên đề thi chưa từng thấy (test).

---

## Phụ lục — Các lỗi code đã sửa

| # | Vị trí | Lỗi | Cách sửa |
|---|---|---|---|
| 1 | `product_v3/core.py` | `DEFAULT_DATA` trỏ tới `Mine_Dataset.xlsx` nhưng file thật là `.xls`, nên cơ chế tự huấn luyện lại **không bao giờ chạy** | Trỏ đúng `.xls`; báo lỗi rõ ràng nếu vẫn thiếu dữ liệu |
| 2 | `product_v3/requirements.txt` | Thiếu **`xlrd`**: pandas không đọc được `.xls`, nên `train_final.py` và Docker build đều lỗi | Thêm `xlrd==2.0.1` |
| 3 | `requirements.txt` (thư mục gốc) | HF Space cài thư viện từ file ở thư mục gốc nhưng file này không tồn tại | Thêm file, trỏ về requirements của app |
| 4 | `product_v3/train_final.py` | Đường dẫn dữ liệu tương đối, chỉ chạy được khi đứng trong `product_v3/` | Dùng `core.DEFAULT_DATA` (đường dẫn tuyệt đối) |
| 5 | `core.predict_batch` | Không kiểm tra đầu vào: ô trống, chữ, giá trị ngoài [0,1] hay S sai vẫn được dự đoán âm thầm | Kiểm tra và báo lỗi kèm số dòng; chấp nhận tên cột viết thường hoặc có khoảng trắng |
| 6 | `pages/1_Du_doan_hang_loat.py` | Bấm "Dùng file mẫu" rồi đổi bộ lọc thì **mất kết quả** (`st.button` chỉ True trong một lượt chạy) | Lưu nguồn dữ liệu vào `st.session_state` |
| 7 | `pages/1_Du_doan_hang_loat.py` | Nhận diện CSV phân biệt hoa thường (`.CSV` bị đọc như Excel); lỗi đọc file làm sập trang | So sánh `.lower()`; bắt lỗi và hiển thị thông báo |
| 8 | `pages/0_Home_page.py` | `model.pkl` không nằm trong git nên lần chạy đầu luôn báo "Chưa có mô hình", dù app tự huấn luyện được | Gọi `load_bundle()` (tự huấn luyện) rồi mới báo trạng thái |
| 9 | `pages/1…4` | Gọi `st.set_page_config` trùng với `Home.py` khi đang dùng `st.navigation` | Bỏ các lệnh trùng; tiêu đề trang lấy từ `st.Page` |
| 10 | `pages/2_Ban_do_rui_ro.py` | Màu chữ trục cố định `#DDD`, gần như không đọc được ở giao diện sáng | Để Streamlit tự chọn màu theo theme |
| 11 | Notebook 02, 04–07 | Lưu hình vào `../result/figures` (chữ thường), trên Linux sẽ tạo sang một thư mục khác | Thống nhất `../Result/Figures` |
| 12 | Notebook 06 | Bảng tên loại đất ghi "sét", không khớp mô tả dữ liệu và `core.py` ("vôi", *limy*) | Sửa thành "vôi" |
| 13 | Notebook 06 | Dùng `%%` trong f-string nên in ra "100%%" | Sửa thành `%` |
| 14 | Notebook 04 | kNN+DE trong `oof_proba` bỏ qua độ đo đã chọn (luôn dùng Euclid) | Truyền đúng `p` của Minkowski |
| 15 | Notebook 04 | Dòng code chết `... if False else ...`; tính `oof_proba` hai lần cho mỗi mô hình (kNN+DE rất chậm) | Dọn code; lưu kết quả vào `oof_cache` |
| 16 | `product_v3/requirements.txt` | Không ghim `pyarrow`; bản mới cần NumPy ≥ 2 nên **mọi bảng và biểu đồ của app sập** khi cài mới | Ghim `pyarrow==17.0.0` |
| 17 | `product_v3/pages/` | Thư mục tên `pages/` xung đột với `st.navigation` (Streamlit cảnh báo) | Đổi tên thành `app_pages/` |
| 18 | `core.py` | Ngưỡng ưu tiên ước lượng trên mô hình **nhị phân** nhưng áp lên xác suất của mô hình **5 lớp**; thống kê recall đo ở ngưỡng chưa nhân biên an toàn | Ngưỡng và thống kê đo trên đúng mô hình phát hiện và đúng ngưỡng × 0,8 |
| 19 | Môi trường notebook | LightGBM 4.7 bị access violation với NumPy 1.26, làm notebook 04 và 05 sập | Ghim `lightgbm==4.5.0` trong `Notebook/requirements-notebooks.txt` |
