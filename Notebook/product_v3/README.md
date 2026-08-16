# HNL-206 UXO AI — Hệ thống hỗ trợ phát hiện, phân loại vật thể chôn ngầm

Ứng dụng web xử lý **hàng loạt** tín hiệu rà phá: tải file cả khu vực → xếp
hạng ưu tiên đào → bản đồ rủi ro → tải kết quả về. Không nhập tay từng tín hiệu.

Bài toán dữ liệu bảng nhỏ (338 mẫu, 3 đặc trưng) → chạy CPU trong tích tắc,
không cần GPU.

## Cấu trúc

```
product/
├── core.py               # LÕI: train, dự đoán hàng loạt, xếp ưu tiên (đã kiểm thử)
├── train_final.py        # huấn luyện mô hình → models/model.pkl
├── Home.py               # trang chính
├── pages/
│   ├── 1_Du_doan_hang_loat.py   # tải file → xếp ưu tiên → tải kết quả Excel
│   ├── 2_Ban_do_rui_ro.py       # bản đồ tín hiệu theo tọa độ
│   ├── 3_Tong_quan.py           # dashboard thống kê
│   └── 4_Thu_don_le.py          # thử một tín hiệu
├── data/
│   ├── raw/Mine_Dataset.xlsx    # dữ liệu huấn luyện
│   └── sample_batch.xlsx/.csv   # file mẫu (có tọa độ) để thử ngay
├── requirements.txt
└── Dockerfile
```

## Chạy

```bash
pip install -r requirements.txt
python train_final.py       # tạo models/model.pkl (chạy trước, bắt buộc)
streamlit run Home.py       # mở giao diện
```

Vào trang **Dự đoán hàng loạt** -> bấm *Dùng file mẫu* -> xem xếp hạng, bản đồ,
tổng quan -> tải kết quả Excel.

## Định dạng file đầu vào

Cần 3 cột **V, H, S**. Tùy chọn thêm `id`, `x` (kinh độ), `y` (vĩ độ) để vẽ bản đồ.

Cột kết quả:
- `xac_suat_co_min` — xác suất tín hiệu là vật nổ thật.
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

- Mô hình sản phẩm dùng HistGradientBoosting (nhanh, ổn định). Phần phân tích
  học thuật (k-NN+DE, sửa rò rỉ, kiểm định thống kê) nằm ở các notebook Giai
  đoạn 2-7, tách khỏi sản phẩm này.
- **model.pkl kèm sẵn** (huấn luyện với phiên bản đã ghim trong requirements.txt)
  nên KHÔNG cần train lại. Nếu môi trường lệch phiên bản khiến pkl không nạp
  được, app **tự huấn luyện lại một lần** từ dữ liệu kèm theo — bạn không phải
  làm gì.
- **Ngưỡng ưu tiên** ước lượng bằng cross-validation (nhất quán Giai đoạn 6),
  KHÔNG đảm bảo bắt 100% mìn trên dữ liệu mới (chỉ giữ recall 100% ở ~50% lần
  thử). Chỉ dùng để sắp thứ tự đào, KHÔNG BAO GIỜ để bỏ qua tín hiệu.
- Muốn đổi thuật toán, sửa `core.py` (hàm `train_and_save`).

## Nguyên tắc an toàn

Hệ thống chỉ **hỗ trợ** sắp xếp ưu tiên và kiểm soát chất lượng. Mọi tín hiệu
vẫn phải được đào kiểm tra đầy đủ. Không dùng kết quả để bỏ qua bất kỳ tín hiệu
nào. Quyết định cuối cùng thuộc về cán bộ kỹ thuật có thẩm quyền.
