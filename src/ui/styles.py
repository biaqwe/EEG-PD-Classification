import streamlit as st

SIDEBAR_CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stAppViewContainer"] { padding-top: 0rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.97), rgba(15,23,48,0.97)) !important;
  border-right: 1px solid rgba(255,255,255,0.12);
  box-shadow: 8px 0 30px rgba(0,0,0,0.35);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  background: transparent !important;
  padding-bottom: 0.6rem !important;
}

section[data-testid="stSidebar"] .sidebar-title{
  color: rgba(255,255,255,0.55);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin: 0.2rem 0 10px 4px !important;
}

section[data-testid="stSidebar"] button{
  width: 100%;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.04) !important;
  color: rgba(255,255,255,0.92) !important;
  padding: 0.75rem 0.9rem !important;
  font-size: 0.95rem !important;
  transition: all 0.15s ease-in-out;
}

section[data-testid="stSidebar"] button:hover{
  background: rgba(106,166,255,0.12) !important;
  border-color: rgba(106,166,255,0.45) !important;
  transform: translateY(-1px);
}

.navbtn-active > button{
  background: linear-gradient(180deg, rgba(106,166,255,0.18), rgba(106,166,255,0.08)) !important;
  border-color: rgba(106,166,255,0.65) !important;
  box-shadow: 0 0 0 1px rgba(106,166,255,0.25);
}

section[data-testid="stSidebar"] .stButton{ margin-bottom: 8px; }

section[data-testid="stSidebar"] .small{
  color: rgba(255,255,255,0.6);
  font-size: 0.85rem;
  line-height: 1.4;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  background: transparent !important;
  padding-top: 0 !important;
}
</style>
"""

BASE_CSS = """
<style>
  :root{
    color-scheme: dark;
    --bg0:#0b1020;
    --bg1:#0f1730;
    --card:#101a33;
    --card2:#0f1a2f;
    --stroke: rgba(255,255,255,.10);
    --stroke2: rgba(255,255,255,.16);
    --txt: rgba(255,255,255,.90);
    --muted: rgba(255,255,255,.70);
    --muted2: rgba(255,255,255,.55);
    --blue:#6aa6ff;
    --cyan:#4fe3d5;
    --lime:#bafc5a;
    --amber:#ffc857;
    --red:#ff4d6d;
    --purple:#a78bfa;
    --widget-bg: rgba(16,26,51,.82);
    --widget-bg-hover: rgba(20,33,62,.92);
    --widget-border: rgba(255,255,255,.14);
    --widget-border-strong: rgba(106,166,255,.48);
  }

  html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"]{
    background: radial-gradient(1200px 700px at 15% 10%, rgba(106,166,255,.14), transparent 60%),
                radial-gradient(900px 650px at 85% 20%, rgba(79,227,213,.12), transparent 55%),
                radial-gradient(900px 650px at 55% 95%, rgba(167,139,250,.12), transparent 55%),
                linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
    color: var(--txt) !important;
  }

  [data-testid="stAppViewContainer"] > .main,
  [data-testid="stAppViewContainer"] > .main > div,
  [data-testid="stAppViewContainer"] > .main > div > div,
  [data-testid="stAppViewContainer"] > .main > div > div > div{
    background: transparent !important;
  }

  .stApp, p, label, span, div, h1, h2, h3, h4, h5, h6 {
    color: var(--txt);
  }

  .block-container{
    padding-top: 1.0rem;
    padding-bottom: 1.8rem;
    max-width: 1350px;
  }

  h1,h2,h3{
    letter-spacing: -0.4px;
  }

  .topbar{
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(16,26,51,.85), rgba(16,26,51,.55));
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
  }

  .subtle{
    color: var(--muted) !important;
    font-size: 0.95rem;
  }

  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:6px 10px;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,.03);
    color: var(--muted) !important;
    font-size: 0.9rem;
  }

  .card{
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(16,26,51,.75), rgba(16,26,51,.45));
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,.22);
  }

  .card-title{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 12px;
    margin-bottom: 6px;
  }

  .kpis{
    display:grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
  }

  @media (max-width: 1100px){
    .kpis{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }

  .kpi{
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,.03);
    border-radius: 16px;
    padding: 12px 12px;
  }

  .kpi .lbl{ color: var(--muted2) !important; font-size: 0.85rem; }
  .kpi .val{ font-size: 1.25rem; font-weight: 700; margin-top: 4px; }
  .kpi .hint{ color: var(--muted) !important; font-size: 0.88rem; margin-top: 6px; }

  .badge{
    display:inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid var(--stroke2);
    background: rgba(255,255,255,.04);
    font-size: .85rem;
    color: var(--muted) !important;
  }

  .badge-idle{ border-color: rgba(255,200,87,.38); background: rgba(255,200,87,.09); color: rgba(255,230,180,.95) !important; }
  .badge-ok{ border-color: rgba(186,252,90,.32); background: rgba(186,252,90,.10); color: rgba(233,255,205,.95) !important; }
  .badge-warn{ border-color: rgba(255,200,87,.32); background: rgba(255,200,87,.10); color: rgba(255,230,180,.95) !important; }
  .badge-err{ border-color: rgba(255,77,109,.34); background: rgba(255,77,109,.10); color: rgba(255,205,215,.95) !important; }
  .badge-run{ border-color: rgba(79,227,213,.32); background: rgba(79,227,213,.10); color: rgba(205,255,248,.95) !important; }

  .sidebar-title{
    font-size: 0.9rem;
    color: var(--muted2) !important;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin: 6px 0 10px 0;
  }

  .navbtn > button{
    width: 100%;
    border-radius: 14px !important;
    border: 1px solid var(--stroke) !important;
    background: rgba(255,255,255,.03) !important;
    color: var(--txt) !important;
    padding: 0.6rem 0.75rem !important;
    transition: transform .05s ease-in-out, border-color .12s ease-in-out;
  }

  .navbtn > button:hover{
    border-color: rgba(106,166,255,.35) !important;
  }

  .navbtn-active > button{
    border-color: rgba(106,166,255,.55) !important;
    background: rgba(106,166,255,.10) !important;
  }

  .small{
    color: var(--muted2) !important;
    font-size: .88rem;
  }

  .logbox{
    border: 1px dashed var(--stroke2);
    background: rgba(255,255,255,.02);
    border-radius: 14px;
    padding: 10px 12px;
    max-height: 240px;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.86rem;
    color: rgba(255,255,255,.82) !important;
  }

  .status-dot{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid var(--stroke2);
    box-shadow: 0 0 0 3px rgba(255,255,255,.03);
  }

  .dot-idle{ background: rgba(255,200,87,.95); box-shadow: 0 0 12px rgba(255,200,87,.22), 0 0 0 3px rgba(255,200,87,.06); }
  .dot-run{  background: rgba(79,227,213,.95); box-shadow: 0 0 12px rgba(79,227,213,.22), 0 0 0 3px rgba(79,227,213,.06); }
  .dot-ok{   background: rgba(186,252,90,.95); box-shadow: 0 0 12px rgba(186,252,90,.20), 0 0 0 3px rgba(186,252,90,.06); }
  .dot-warn{ background: rgba(255,200,87,.95); box-shadow: 0 0 12px rgba(255,200,87,.18), 0 0 0 3px rgba(255,200,87,.06); }
  .dot-err{  background: rgba(255,77,109,.95); box-shadow: 0 0 12px rgba(255,77,109,.22), 0 0 0 3px rgba(255,77,109,.06); }

  /* Native Streamlit widgets */
  .stAlert, .stJson, .stExpander, .stDataFrame, .stTable, .stMetric,
  [data-testid="stFileUploader"], [data-testid="stNumberInputContainer"],
  [data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
  [data-testid="stSelectbox"], [data-testid="stMultiSelect"],
  [data-testid="stDateInputField"], [data-testid="stTimeInput"],
  [data-testid="stCheckbox"], [data-testid="stRadio"], [data-testid="stTabs"] {
    color: var(--txt) !important;
  }

  .stButton > button,
  .stDownloadButton > button,
  [data-testid="stBaseButton-secondary"],
  [data-testid="stBaseButton-primary"]{
    border-radius: 14px !important;
    border: 1px solid var(--widget-border) !important;
    background: linear-gradient(180deg, rgba(20,33,62,.96), rgba(16,26,51,.96)) !important;
    color: var(--txt) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,.18);
  }

  .stButton > button:hover,
  .stDownloadButton > button:hover,
  [data-testid="stBaseButton-secondary"]:hover,
  [data-testid="stBaseButton-primary"]:hover{
    border-color: var(--widget-border-strong) !important;
    background: linear-gradient(180deg, rgba(28,45,83,.98), rgba(20,33,62,.98)) !important;
    color: white !important;
  }

  .stButton > button:disabled,
  .stDownloadButton > button:disabled{
    opacity: .45 !important;
    color: rgba(255,255,255,.55) !important;
  }

  .stTextInput input,
  .stNumberInput input,
  .stTextArea textarea,
  [data-baseweb="select"] > div,
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stDateInputField"] > div,
  [data-testid="stTimeInput"] > div {
    background: var(--widget-bg) !important;
    color: var(--txt) !important;
    border: 1px solid var(--widget-border) !important;
    border-radius: 14px !important;
  }

  .stTextInput input:focus,
  .stNumberInput input:focus,
  .stTextArea textarea:focus{
    border-color: var(--widget-border-strong) !important;
    box-shadow: 0 0 0 1px rgba(106,166,255,.25) !important;
  }

  .stTextInput label,
  .stNumberInput label,
  .stTextArea label,
  .stSelectbox label,
  .stMultiSelect label,
  .stFileUploader label,
  .stCheckbox label,
  .stRadio label,
  .stMarkdown, .stCaption, small {
    color: var(--txt) !important;
  }

  [data-testid="stFileUploaderDropzone"]:hover {
    background: var(--widget-bg-hover) !important;
    border-color: var(--widget-border-strong) !important;
  }

  [data-testid="stFileUploaderDropzone"] * {
    color: var(--txt) !important;
  }

  .stCheckbox > label,
  .stRadio > label {
    color: var(--txt) !important;
  }

  .stCheckbox [data-testid="stWidgetLabel"] p,
  .stRadio [data-testid="stWidgetLabel"] p,
  [data-testid="stFileUploaderDropzoneInstructions"] span,
  [data-testid="stFileUploaderDropzoneInstructions"] small,
  [data-testid="stFileUploaderFileName"] {
    color: var(--txt) !important;
  }

  [data-baseweb="checkbox"] > div,
  [data-baseweb="radio"] > div {
    background-color: transparent !important;
    border-color: var(--widget-border-strong) !important;
  }

  .stExpander {
    border: 1px solid var(--stroke) !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg, rgba(16,26,51,.72), rgba(16,26,51,.44)) !important;
    overflow: hidden;
  }

  .stExpander summary, .stExpander details, .stExpander p {
    color: var(--txt) !important;
  }

  div[data-testid="stMetric"] {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 16px !important;
    padding: 10px 12px !important;
  }

  div[data-testid="stMetric"] label,
  div[data-testid="stMetric"] [data-testid="stMetricLabel"],
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--txt) !important;
  }

  [data-testid="stDataFrame"] > div,
  .stDataFrame, .stTable {
    background: rgba(16,26,51,.68) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 16px !important;
  }

  [data-testid="stDataFrame"] *, .stTable * {
    color: var(--txt) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent !important;
  }

  .stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    background: rgba(255,255,255,.04) !important;
    color: var(--txt) !important;
    border: 1px solid var(--stroke) !important;
  }

  .stTabs [aria-selected="true"] {
    background: rgba(106,166,255,.12) !important;
    border-color: rgba(106,166,255,.48) !important;
  }

  .stAlert {
    border-radius: 16px !important;
    border: 1px solid var(--stroke) !important;
    background: linear-gradient(180deg, rgba(16,26,51,.82), rgba(16,26,51,.60)) !important;
  }

  .stAlert *, .stJson * {
    color: var(--txt) !important;
  }

  .stJson {
    background: rgba(9,14,28,.90) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 16px !important;
  }
</style>
"""

def inject_styles():
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    st.markdown(BASE_CSS, unsafe_allow_html=True)