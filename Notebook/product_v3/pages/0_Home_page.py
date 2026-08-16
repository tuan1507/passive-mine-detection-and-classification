# -*- coding: utf-8 -*-
"""
Nội dung trang chủ (Home).
File này được gọi qua st.navigation() trong Home.py.
"""
import streamlit as st
import os
import core

# ---- CSS tùy chỉnh cho giao diện chuyên nghiệp ----
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 800; color: #1F3864;
                  margin-bottom: 0; }
    .subtitle  { font-size: 1.05rem; color: #5A6B7B; margin-top: 0; }
    .metric-card { background: #F4F7FB; border: 1px solid #D6E0EE;
                   border-left: 5px solid #2E5496; border-radius: 8px;
                   padding: 1rem 1.2rem; color: #1F3864 !important; min-height: 96px; }
    .metric-card b { color: #16324F; font-size: 1.02rem; }
    .metric-card { line-height: 1.35; }
    .safety-box { background: #FFF3CD; border-left: 6px solid #C55A11;
                  border-radius: 8px; padding: 1rem 1.3rem; color: #664d03;
                  font-weight: 500; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    section[data-testid="stSidebar"] { background: #1F3864; }
    section[data-testid="stSidebar"] * { color: #E8EEF6; }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar bổ sung ----
with st.sidebar:
    st.markdown("### 🛡️ HNL-206 UXO AI")
    st.caption("Hệ thống hỗ trợ phát hiện, phân loại vật thể chôn ngầm")
    st.divider()
    st.markdown("**Quy trình sử dụng**")
    st.markdown("1. Tải dữ liệu tín hiệu (trang *Dự đoán hàng loạt*)\n"
                "2. Xem xếp hạng ưu tiên đào\n"
                "3. Xem bản đồ rủi ro\n"
                "4. Tải kết quả về")
    st.divider()
    if os.path.exists(core.MODEL_PATH):
        st.success("✓ Mô hình đã sẵn sàng")
    else:
        st.error("✗ Chưa có mô hình — chạy train_final.py")

# ---- Nội dung chính ----
st.markdown('<p class="main-title">Hệ thống hỗ trợ phát hiện, phân loại '
            'vật thể chôn ngầm</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ứng dụng học máy trong công tác rà phá '
            'bom mìn, vật nổ — Công ty TNHH MTV Hữu Nghị Nam Lào 206</p>',
            unsafe_allow_html=True)
st.write("")

st.markdown("""
<div class="safety-box">
⚠️ <b>Nguyên tắc an toàn:</b> Hệ thống chỉ đóng vai trò <b>hỗ trợ</b> sắp xếp
thứ tự ưu tiên và kiểm soát chất lượng. Mọi tín hiệu phát hiện được
<b>vẫn phải được đào kiểm tra đầy đủ</b> theo quy trình hiện hành. Kết quả dự
đoán không được dùng làm căn cứ bỏ qua bất kỳ tín hiệu nào. Quyết định cuối
cùng thuộc về cán bộ kỹ thuật có thẩm quyền.
</div>
""", unsafe_allow_html=True)

st.write("")
st.subheader("Các chức năng chính")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><b>📤 Dự đoán hàng loạt</b><br>'
                'Tải file tín hiệu cả khu vực, nhận kết quả xếp hạng ưu tiên đào.'
                '</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><b>🗺️ Bản đồ rủi ro</b><br>'
                'Hiển thị tín hiệu trên bản đồ, tô màu theo mức ưu tiên.'
                '</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><b>📊 Tổng quan</b><br>'
                'Thống kê toàn khu vực: số điểm ưu tiên cao, phân bố mức độ.'
                '</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><b>🔬 Thử đơn lẻ</b><br>'
                'Nhập một tín hiệu để kiểm tra nhanh mô hình dự đoán.'
                '</div>', unsafe_allow_html=True)

st.write("")
st.info("👈 Chọn chức năng ở thanh bên trái để bắt đầu. "
        "Nếu chưa có dữ liệu, dùng file mẫu `data/sample_batch.xlsx`.")

# thông tin mô hình
if os.path.exists(core.MODEL_PATH):
    meta = core.load_bundle()["meta"]
    st.divider()
    st.caption(
        f"Mô hình hiện tại · phân loại 5 lớp (CV): {meta['cv_accuracy']}% · "
        f"recall phát hiện mìn (CV): {meta.get('detect_recall_cv','?')}%")
    st.caption(
        f"Ngưỡng ưu tiên = {meta['safe_threshold']} (ước lượng qua CV). "
        f"Lưu ý: chỉ giữ được recall 100% ở ~{meta.get('hold_100_rate','?')}% lần thử "
        f"→ KHÔNG đảm bảo không bỏ sót; chỉ dùng để sắp thứ tự, không để bỏ qua tín hiệu.")
