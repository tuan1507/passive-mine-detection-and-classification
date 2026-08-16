# -*- coding: utf-8 -*-
"""Trang tổng quan: thống kê và biểu đồ toàn khu vực."""
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Tổng quan", page_icon="📊", layout="wide")
st.title("📊 Tổng quan khu vực")

if "pred" not in st.session_state:
    st.warning("Chưa có dữ liệu. Hãy vào trang *Dự đoán hàng loạt* trước.")
    st.stop()

pred = st.session_state["pred"]

# --- các chỉ số ---
n = len(pred)
n_high = int(pred["muc_uu_tien"].isin(["Rất cao", "Cao"]).sum())
n_safe = int(pred["uu_tien_thap"].sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tổng tín hiệu", n)
m2.metric("Ưu tiên cao / rất cao", n_high)
m3.metric("Ưu tiên thấp", n_safe)
m4.metric("Tỷ lệ ưu tiên thấp", f"{round(100*n_safe/n,1) if n else 0}%")

st.divider()
c1, c2 = st.columns(2)

# --- phân bố mức ưu tiên ---
with c1:
    st.subheader("Phân bố mức ưu tiên")
    order = ["Rất cao", "Cao", "Trung bình", "Thấp"]
    dist = pred["muc_uu_tien"].value_counts().reindex(order).fillna(0).reset_index()
    dist.columns = ["Mức", "Số lượng"]
    chart = alt.Chart(dist).mark_bar().encode(
        x=alt.X("Mức", sort=order),
        y="Số lượng",
        color=alt.Color("Mức", scale=alt.Scale(
            domain=order, range=["#c81e1e", "#e68214", "#e6c828", "#3ca05a"]),
            legend=None),
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

# --- phân bố loại vật thể dự đoán ---
with c2:
    st.subheader("Phân bố loại vật thể dự đoán")
    typ = pred["loai_du_doan"].value_counts().reset_index()
    typ.columns = ["Loại", "Số lượng"]
    chart2 = alt.Chart(typ).mark_arc(innerRadius=60).encode(
        theta="Số lượng", color=alt.Color("Loại", legend=alt.Legend(orient="bottom")),
        tooltip=["Loại", "Số lượng"],
    ).properties(height=320)
    st.altair_chart(chart2, use_container_width=True)

# --- phân bố xác suất có mìn ---
st.subheader("Phân bố xác suất có mìn")
hist = alt.Chart(pred).mark_bar().encode(
    x=alt.X("xac_suat_co_min", bin=alt.Bin(maxbins=20), title="Xác suất có mìn"),
    y=alt.Y("count()", title="Số tín hiệu"),
).properties(height=260)
st.altair_chart(hist, use_container_width=True)

# --- phân tích theo loại đất (nếu có) ---
if "S" in pred.columns:
    st.subheader("Số tín hiệu theo loại đất")
    from core import SOIL_NAMES as soil_names
    tmp = pred.copy()
    tmp["Loại đất"] = tmp["S"].round(1).map(soil_names).fillna(tmp["S"].astype(str))
    soil = tmp.groupby("Loại đất").agg(
        so_tin_hieu=("S", "size"),
        p_min_tb=("xac_suat_co_min", "mean")).reset_index()
    soil["p_min_tb"] = (soil["p_min_tb"] * 100).round(1)
    soil.columns = ["Loại đất", "Số tín hiệu", "P(có mìn) TB (%)"]
    st.dataframe(soil, use_container_width=True, hide_index=True)
