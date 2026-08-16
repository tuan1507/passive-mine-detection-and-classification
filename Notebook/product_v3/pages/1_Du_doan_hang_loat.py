# -*- coding: utf-8 -*-
"""Trang dự đoán hàng loạt: tải file → xử lý → xếp ưu tiên → tải kết quả."""
import os
import streamlit as st
import pandas as pd
import io
import core

st.set_page_config(page_title="Dự đoán hàng loạt", page_icon="📤", layout="wide")
st.title("📤 Dự đoán hàng loạt")
st.caption("Tải lên file tín hiệu của cả khu vực (Excel hoặc CSV), "
           "hệ thống trả về bảng kết quả có xếp hạng ưu tiên đào.")

st.markdown("""
**Định dạng file:** phải có 3 cột **V, H, S**. Tùy chọn thêm cột
`id`, `x`, `y` (tọa độ) để dùng cho bản đồ.
""")

up = st.file_uploader("Chọn file tín hiệu", type=["xlsx", "xls", "csv"])

col_a, col_b = st.columns([1, 3])
use_sample = col_a.button("Dùng file mẫu")

df_in = None
if up is not None:
    df_in = pd.read_csv(up) if up.name.endswith("csv") else pd.read_excel(up)
elif use_sample:
    try:
        df_in = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", "data", "sample_batch.xlsx"))
        st.info("Đang dùng file mẫu (338 tín hiệu, có tọa độ giả lập).")
    except Exception:
        st.error("Không tìm thấy file mẫu.")

if df_in is not None:
    try:
        pred, meta = core.predict_batch(df_in)
    except ValueError as e:
        st.error(str(e)); st.stop()

    # lưu vào session để các trang khác dùng
    st.session_state["pred"] = pred

    s = core.summarize(pred)
    st.success(f"Đã xử lý {s['tong_tin_hieu']} tín hiệu.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng tín hiệu", s["tong_tin_hieu"])
    m2.metric("Ưu tiên cao", s["uu_tien_cao"])
    m3.metric("Ưu tiên thấp (xếp cuối)", s["uu_tien_thap"])
    m4.metric("Tỷ lệ ưu tiên thấp", f"{s['ty_le_uu_tien_thap']}%")

    st.divider()

    # bộ lọc
    st.subheader("Bảng kết quả (xếp theo ưu tiên đào)")
    levels = ["Rất cao", "Cao", "Trung bình", "Thấp"]
    chosen = st.multiselect("Lọc theo mức ưu tiên", levels, default=levels)
    view = pred[pred["muc_uu_tien"].isin(chosen)]

    def color_level(val):
        # nền + chữ tối để tương phản rõ trên giao diện tối; đỏ=nguy hiểm nhất
        style = {
            "Rất cao":    "background-color: #e53935; color: #ffffff; font-weight: 700",
            "Cao":        "background-color: #fb8c00; color: #1a1a1a; font-weight: 700",
            "Trung bình": "background-color: #fdd835; color: #1a1a1a; font-weight: 600",
            "Thấp":       "background-color: #43a047; color: #ffffff; font-weight: 600",
        }
        return style.get(val, "color: #1a1a1a")

    show_cols = [c for c in ["id", "V", "H", "S", "xac_suat_co_min",
                 "loai_du_doan", "muc_uu_tien", "hang_uu_tien"] if c in view.columns]
    st.dataframe(
        view[show_cols].style.map(color_level, subset=["muc_uu_tien"]),
        use_container_width=True, hide_index=True, height=420)

    # tải về
    st.divider()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pred.to_excel(w, index=False, sheet_name="ket_qua")
    st.download_button("⬇️ Tải kết quả (Excel)", buf.getvalue(),
        file_name="ket_qua_du_doan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Chưa có dữ liệu. Tải file lên hoặc bấm *Dùng file mẫu*.")
