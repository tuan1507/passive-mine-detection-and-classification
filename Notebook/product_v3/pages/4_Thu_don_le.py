# -*- coding: utf-8 -*-
"""Trang thử một tín hiệu đơn lẻ — để kiểm tra nhanh mô hình."""
import streamlit as st
import pandas as pd
import core

st.set_page_config(page_title="Thử đơn lẻ", page_icon="🔬", layout="centered")
st.title("🔬 Thử một tín hiệu")
st.caption("Nhập thủ công một tín hiệu để xem mô hình dự đoán. "
           "Dùng để minh họa và kiểm thử, không phải luồng chính.")

c1, c2, c3 = st.columns(3)
V = c1.slider("V — điện áp dị thường", 0.0, 1.0, 0.5, 0.01)
H = c2.slider("H — độ cao đầu dò", 0.0, 1.0, 0.5, 0.01)
S = c3.selectbox("S — loại đất", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    format_func=lambda v: core.SOIL_NAMES[v])

if st.button("Dự đoán", type="primary"):
    df = pd.DataFrame([{"V": V, "H": H, "S": S}])
    pred, meta = core.predict_batch(df)
    row = pred.iloc[0]

    st.divider()
    lvl = row["muc_uu_tien"]
    color = {"Rất cao": "🔴", "Cao": "🟠", "Trung bình": "🟡", "Thấp": "🟢"}[lvl]
    st.markdown(f"### Kết quả: **{row['loai_du_doan']}**")
    st.markdown(f"- Xác suất có mìn: **{row['xac_suat_co_min']:.1%}**")
    st.markdown(f"- Mức ưu tiên đào: {color} **{lvl}**")

    # phân bố xác suất từng lớp
    bundle = core.load_bundle()
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(df[["V", "H", "S"]].values)[0]
    classes = pipe.named_steps["clf"].classes_
    dist = pd.DataFrame({
        "Loại": [core.CLASS_NAMES[int(c)] for c in classes],
        "Xác suất": proba,
    }).sort_values("Xác suất", ascending=False)
    st.bar_chart(dist.set_index("Loại"))

    st.warning("Kết quả chỉ để tham khảo. Mọi tín hiệu vẫn phải đào kiểm tra "
               "đầy đủ theo quy trình.")
