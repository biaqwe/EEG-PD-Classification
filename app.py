import streamlit as st

from src.state import init_session_state
from src.ui.styles import inject_styles
from src.ui.sidebar import sidebar_nav
from src.ui.components import render_topbar
from src.ui.pages import (
    render_dashboard,
    render_import,
    render_preprocess,
    render_results,
)

st.set_page_config(
    page_title="EEG Classification (PD vs HC)",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()
inject_styles()
sidebar_nav()
render_topbar()

page = st.session_state.page
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

if page == "Dashboard":
    render_dashboard()
elif page == "Import":
    render_import()
elif page == "Preprocess":
    render_preprocess()
elif page == "Results":
    render_results()
else:
    st.session_state.page = "Dashboard"
    render_dashboard()