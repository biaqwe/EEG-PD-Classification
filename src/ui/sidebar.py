import streamlit as st

from src.actions import (
    run_train_cnn,
    run_train_cnn_group_cv,
    run_train_svm,
    run_train_svm_group_cv,
)


def sidebar_nav():
    st.sidebar.markdown("<div class='sidebar-title'>Menu</div>", unsafe_allow_html=True)

    def nav_button(label: str):
        active = st.session_state.page == label
        cls = "navbtn navbtn-active" if active else "navbtn"

        with st.sidebar.container():
            st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
            clicked = st.button(
                label,
                width="stretch",
                type="primary" if active else "secondary",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state.page = label

    nav_button("Dashboard")
    nav_button("Preprocess")
    nav_button("Import")
    nav_button("Results")
    nav_button("Raw EEG Viewer")

    tabular_ok = st.session_state.dataset_df is not None
    raw_ok = bool(st.session_state.raw_file_payloads)

    st.sidebar.markdown("<div class='sidebar-title'>Models</div>", unsafe_allow_html=True)

    if st.sidebar.button("Train SVM", width="stretch", disabled=not tabular_ok):
        run_train_svm()

    if st.sidebar.button("Run SVM Group CV", width="stretch", disabled=not tabular_ok):
        run_train_svm_group_cv()

    if st.sidebar.button("Train CNN", width="stretch", disabled=not raw_ok):
        run_train_cnn()

    if st.sidebar.button("Run CNN Group CV", width="stretch", disabled=not raw_ok):
        run_train_cnn_group_cv()

    st.sidebar.markdown(
        "<div class='small'>Load a CSV/MAT feature table for SVM, or raw EEG files for CNN.</div>",
        unsafe_allow_html=True,
    )