# -*- coding: utf-8 -*-
"""Trang bản đồ rủi ro: hiển thị tín hiệu theo tọa độ khảo sát, màu theo ưu tiên.

Lưu ý: tọa độ (x, y) là tọa độ LƯỚI KHẢO SÁT của khu vực (mét hoặc kinh/vĩ độ),
không nhất thiết là GPS thật. Vì vậy ta vẽ dạng biểu đồ phân tán CÓ HỆ TRỤC
(dễ hiểu, luôn hoạt động), thay vì nền bản đồ Mapbox (cần token và chỉ đúng khi
có GPS thật). File mẫu dùng tọa độ giả lập để minh họa.
"""
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Bản đồ rủi ro", page_icon="🗺️", layout="wide")
st.title("🗺️ Bản đồ rủi ro (lưới khảo sát)")
st.caption("Mỗi điểm là một tín hiệu, đặt theo tọa độ khảo sát X–Y. "
           "Màu = mức ưu tiên đào; kích thước = xác suất có mìn. "
           "Rê chuột vào điểm để xem chi tiết; cuộn để phóng to.")

if "pred" not in st.session_state:
    st.warning("Chưa có dữ liệu. Hãy vào trang *Dự đoán hàng loạt* và xử lý file trước.")
    st.stop()

pred = st.session_state["pred"]
if not {"x", "y"}.issubset(pred.columns):
    st.error("File không có cột tọa độ (x, y) nên không vẽ được bản đồ. "
             "Bạn vẫn xem được bảng kết quả ở trang Dự đoán hàng loạt.")
    st.stop()

plot = pred.dropna(subset=["x", "y"]).copy()

LEVELS = ["Rất cao", "Cao", "Trung bình", "Thấp"]
COLORS = ["#e53935", "#fb8c00", "#fdd835", "#43a047"]  # đỏ→cam→vàng→xanh
st.markdown("**Chú giải mức ưu tiên đào:** "
            "🔴 Rất cao · 🟠 Cao · 🟡 Trung bình · 🟢 Thấp  "
            "— (điểm càng to = xác suất có mìn càng cao)")

chart = (
    alt.Chart(plot)
    .mark_circle(opacity=0.85, stroke="#333", strokeWidth=0.4)
    .encode(
        x=alt.X("x:Q", title="Tọa độ X (lưới khảo sát)",
                scale=alt.Scale(zero=False, nice=True)),
        y=alt.Y("y:Q", title="Tọa độ Y (lưới khảo sát)",
                scale=alt.Scale(zero=False, nice=True)),
        size=alt.Size("xac_suat_co_min:Q", title="P(có mìn)",
                      scale=alt.Scale(range=[40, 500])),
        color=alt.Color("muc_uu_tien:N", title="Mức ưu tiên",
                        scale=alt.Scale(domain=LEVELS, range=COLORS),
                        sort=LEVELS),
        order=alt.Order("xac_suat_co_min:Q", sort="descending"),
        tooltip=[alt.Tooltip("id:N", title="Mã điểm"),
                 alt.Tooltip("loai_du_doan:N", title="Loại dự đoán"),
                 alt.Tooltip("xac_suat_co_min:Q", title="P(có mìn)", format=".3f"),
                 alt.Tooltip("muc_uu_tien:N", title="Ưu tiên"),
                 alt.Tooltip("hang_uu_tien:Q", title="Hạng")],
    )
    .properties(height=560)
    .configure_axis(grid=True, gridOpacity=0.15, labelColor="#DDD", titleColor="#DDD")
    .configure_legend(labelColor="#DDD", titleColor="#DDD")
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Điểm ưu tiên rất cao", int((plot["muc_uu_tien"] == "Rất cao").sum()))
c2.metric("Điểm ưu tiên cao", int((plot["muc_uu_tien"] == "Cao").sum()))
c3.metric("Tổng điểm trên bản đồ", len(plot))

st.caption("Ghi chú: đây là tọa độ lưới khảo sát cục bộ. Khi triển khai thực địa "
           "với GPS thật, có thể chuyển sang bản đồ nền địa lý; bản demo dùng tọa "
           "độ giả lập nên hiển thị dạng lưới X–Y để thấy rõ phân bố rủi ro.")
