# HNL-206 UXO AI — Hệ thống hỗ trợ phát hiện, phân loại vật thể chôn ngầm

Ứng dụng web xử lý **hàng loạt** tín hiệu rà phá: tải file cả khu vực → xếp
hạng ưu tiên đào → bản đồ rủi ro → tải kết quả về. Không nhập tay từng tín hiệu.

Bài toán dữ liệu bảng nhỏ (338 mẫu, 3 đặc trưng) → chạy CPU, không cần GPU.
Bản notebook từng bước của thư mục này: `../08_san_pham_core.ipynb` (lõi) và
`../10_kiem_thu_app.ipynb` (các trang + kiểm thử).

## Cấu trúc

```
product_v3/
├── core.py               # LÕI: đặc trưng, 2 mô hình, đánh giá CV, ngưỡng, dự đoán hàng loạt
├── train_final.py        # huấn luyện mô hình → models/model.pkl
├── Home.py               # điểm vào, khai báo các trang (st.navigation)
├── app_pages/            # các trang (không đặt tên pages/ để tránh xung đột st.navigation)
│   ├── 0_Home_page.py           # trang chủ + thông tin mô hình
│   ├── 1_Du_doan_hang_loat.py   # tải file → xếp ưu tiên → tải kết quả Excel
│   ├── 2_Ban_do_rui_ro.py       # bản đồ tín hiệu theo tọa độ
│   ├── 3_Tong_quan.py           # dashboard thống kê
│   └── 4_Thu_don_le.py          # thử một tín hiệu
├── models/model.pkl      # mô hình đã huấn luyện (commit sẵn)
├── data/
│   ├── raw/Mine_Dataset.xls     # dữ liệu huấn luyện (cần xlrd)
│   └── sample_batch.xlsx/.csv   # file mẫu (có tọa độ) để thử ngay
├── requirements.txt      # Python 3.11–3.13, dùng cho cả local và Streamlit Cloud
└── Dockerfile
```

## Chạy

```bash
pip install -r requirements.txt
streamlit run Home.py       # mở giao diện, model.pkl có sẵn nên chạy ngay
python train_final.py       # (tuỳ chọn) huấn luyện lại, cần sau khi sửa core.py
```

Vào trang **Dự đoán hàng loạt** → bấm *Dùng file mẫu* → xem xếp hạng, bản đồ,
tổng quan → tải kết quả Excel.

## Định dạng file đầu vào

Cần 3 cột **V, H, S** (đã chuẩn hoá về [0, 1]). Tùy chọn thêm `id`, `x`, `y` để vẽ bản đồ.

Cột kết quả:
- `xac_suat_co_min` — xác suất tín hiệu là vật nổ thật (mô hình phát hiện).
- `loai_du_doan`, `loai_kha_nang_2` — loại vật thể có khả năng nhất và thứ hai (mô hình phân loại).
- `muc_uu_tien` — Rất cao / Cao / Trung bình / Thấp.
- `hang_uu_tien` — thứ tự đào (1 = đào trước).
- `uu_tien_thap` — True: dưới ngưỡng ưu tiên (ước lượng qua CV). Xếp CUỐI hàng
  đợi nhưng VẪN phải đào kiểm tra. Ngưỡng KHÔNG đảm bảo bắt 100% mìn trên dữ
  liệu mới (xem Giai đoạn 6).

## Docker

```bash
docker build -t hnl206-uxo .
docker run -p 8501:8501 hnl206-uxo    # http://localhost:8501
```

## Ghi chú

- Hai mô hình: **MLP + log V + one-hot loại đất** cho loại mìn (~72% CV),
  **HistGradientBoosting nhị phân** cho P(có mìn) và xếp hạng (ROC-AUC ~0,95).
  Phần phân tích học thuật (k-NN+DE, sửa rò rỉ, kiểm định thống kê) nằm ở các
  notebook Giai đoạn 2–7.
- **model.pkl được commit sẵn** (huấn luyện bằng scikit-learn 1.5.2 như trong
  requirements.txt), app chỉ nạp file, không huấn luyện trên máy chủ. Nếu pkl thiếu,
  hỏng hoặc được tạo bằng phiên bản scikit-learn khác, app **tự huấn luyện lại một lần**.
- **Ngưỡng ưu tiên** ước lượng bằng cross-validation (nhất quán Giai đoạn 6),
  KHÔNG đảm bảo bắt 100% mìn trên dữ liệu mới (chỉ giữ recall 100% ở ~75% lần
  thử). Chỉ dùng để sắp thứ tự đào, KHÔNG BAO GIỜ để bỏ qua tín hiệu.
- Muốn đổi thuật toán, sửa `core.py` (`make_pipeline` cho phân loại, `make_detector`
  cho phát hiện) rồi chạy lại `train_final.py`.

## Nguyên tắc an toàn

Hệ thống chỉ **hỗ trợ** sắp xếp ưu tiên và kiểm soát chất lượng. Mọi tín hiệu
vẫn phải được đào kiểm tra đầy đủ. Không dùng kết quả để bỏ qua bất kỳ tín hiệu
nào. Quyết định cuối cùng thuộc về cán bộ kỹ thuật có thẩm quyền.
