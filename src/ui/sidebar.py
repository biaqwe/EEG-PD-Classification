import streamlit as st

from src.actions import (
    run_train_cnn,
    run_train_svm,
    run_train_svm_group_cv,
)
from src.state import log
from src.utils import now_iso


def sidebar_nav():
    st.sidebar.markdown("<div class='sidebar-title'>Menu</div>", unsafe_allow_html=True)

    def nav_button(label: str):
        active = st.session_state.page == label
        cls = "navbtn navbtn-active" if active else "navbtn"

        with st.sidebar.container():
            st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
            clicked = st.button(label, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state.page = label

    nav_button("Dashboard")
    nav_button("Import")
    nav_button("Preprocess")
    nav_button("Results")

    st.sidebar.markdown("<div class='sidebar-title'>Run</div>", unsafe_allow_html=True)

    if st.sidebar.button("Clear logs", use_container_width=True):
        st.session_state.logs = []
        log("Logs cleared.", now_iso)

    st.sidebar.markdown("<div class='sidebar-title'>Models</div>", unsafe_allow_html=True)

    colC, colD = st.sidebar.columns(2)
    with colC:
        if st.button("Train SVM", use_container_width=True):
            run_train_svm()
    with colD:
        if st.button("Train Raw EEG CNN", use_container_width=True):
            run_train_cnn()

    if st.sidebar.button("Run SVM Group CV", use_container_width=True):
        run_train_svm_group_cv()