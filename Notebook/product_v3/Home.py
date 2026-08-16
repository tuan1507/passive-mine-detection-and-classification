# -*- coding: utf-8 -*-
"""
Entry point. Chạy: streamlit run Home.py
Dùng st.navigation() để hiển thị tên trang tiếng Việt có dấu,
giữ nguyên tên file tiếng Anh theo quy tắc đặt tên chuẩn.
"""
import streamlit as st

st.set_page_config(
    page_title="Hệ thống hỗ trợ rà phá bom mìn",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Định nghĩa các trang với tên tiếng Việt có dấu ----
# title= là nhãn hiển thị trên sidebar, không liên quan đến tên file
pg = st.navigation([
    st.Page("pages/0_Home_page.py",        title="Trang chủ",          icon="🏠"),
    st.Page("pages/1_Du_doan_hang_loat.py", title="Dự đoán hàng loạt", icon="📤"),
    st.Page("pages/2_Ban_do_rui_ro.py",    title="Bản đồ rủi ro",      icon="🗺️"),
    st.Page("pages/3_Tong_quan.py",        title="Tổng quan",           icon="📊"),
    st.Page("pages/4_Thu_don_le.py",       title="Thử đơn lẻ",          icon="🔬"),
])

pg.run()

