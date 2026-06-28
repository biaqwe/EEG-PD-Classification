import streamlit as st

SIDEBAR_CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stAppViewContainer"] { padding-top: 0rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #102946 0%, #153d69 100%) !important;
  border-right: 1px solid rgba(157, 197, 235, 0.24);
  box-shadow: 10px 0 30px rgba(0,0,0,0.28);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  background: transparent !important;
  padding: 1.1rem 0.85rem 1rem !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  background: transparent !important;
  gap: 0.32rem !important;
}

section[data-testid="stSidebar"] .sidebar-title{
  color: rgba(210,229,246,0.82) !important;
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  margin: 0.45rem 0 0.45rem 0.15rem !important;
}

section[data-testid="stSidebar"] .stButton{ margin-bottom: 0.28rem; }

section[data-testid="stSidebar"] button{
  width: 100%;
  min-height: 2.65rem;
  border-radius: 8px !important;
  border: 1px solid rgba(157, 197, 235, 0.24) !important;
  background: rgba(255,255,255,0.055) !important;
  color: rgba(246,251,255,0.96) !important;
  padding: 0.68rem 0.78rem !important;
  font-size: 0.93rem !important;
  font-weight: 700 !important;
  transition: background .14s ease, border-color .14s ease, transform .14s ease;
}

section[data-testid="stSidebar"] button:hover{
  background: rgba(78, 163, 255, 0.18) !important;
  border-color: rgba(78, 163, 255, 0.56) !important;
  transform: translateY(-1px);
}

section[data-testid="stSidebar"] button:disabled{
  opacity: .52 !important;
  cursor: not-allowed !important;
  transform: none !important;
}

.navbtn-active > button{
  background: linear-gradient(180deg, rgba(78, 163, 255, 0.28), rgba(119, 194, 255, 0.18)) !important;
  border-color: rgba(78, 163, 255, 0.72) !important;
  box-shadow: inset 3px 0 0 rgba(78, 163, 255, 0.95), 0 8px 20px rgba(0,0,0,0.18);
}

section[data-testid="stSidebar"] .small{
  color: rgba(210,229,246,0.76);
  font-size: 0.85rem;
  line-height: 1.45;
}
</style>
"""

BASE_CSS = """
<style>
  :root{
    color-scheme: dark;
    --bg0:#10233f;
    --bg1:#153a63;
    --surface:#172f52;
    --surface-2:#1d3c66;
    --surface-3:#244b79;
    --surface-soft:rgba(23,47,82,.80);
    --stroke:rgba(157,197,235,.26);
    --stroke-strong:rgba(78,163,255,.58);
    --txt:#f6fbff;
    --muted:#c7dcf2;
    --muted-2:#9dc5eb;
    --accent:#4ea3ff;
    --accent-2:#77c2ff;
    --good:#7ee3a4;
    --warn:#ffd166;
    --error:#ff7b9c;
    --violet:#a9b8ff;
    --widget-bg:rgba(20,43,75,.94);
    --widget-bg-hover:rgba(28,59,101,.97);
    --shadow:0 14px 34px rgba(8,27,52,.24);
    --radius:8px;
  }

  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"]{
    background:
      radial-gradient(circle at top left, rgba(119,194,255,.20), transparent 32%),
      linear-gradient(180deg, rgba(78,163,255,.10) 0%, rgba(78,163,255,0) 38%),
      linear-gradient(135deg, var(--bg0) 0%, #123058 48%, var(--bg1) 100%) !important;
    color: var(--txt) !important;
  }

  [data-testid="stAppViewContainer"] > .main,
  [data-testid="stAppViewContainer"] > .main > div,
  [data-testid="stAppViewContainer"] > .main > div > div,
  [data-testid="stAppViewContainer"] > .main > div > div > div{
    background: transparent !important;
  }

  .block-container{
    max-width: 1400px;
    padding: 1.05rem 2rem 2.4rem !important;
  }

  .stApp, p, label, span, div, h1, h2, h3, h4, h5, h6 {
    color: var(--txt);
    letter-spacing: 0;
  }

  h1,h2,h3{
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: .35rem;
  }

  p{
    line-height: 1.55;
  }

  a{
    color: var(--accent);
  }

  .spacer-xs{ height: 0.35rem; }
  .spacer-sm{ height: 0.65rem; }
  .spacer-md{ height: 1rem; }
  .spacer-lg{ height: 1.4rem; }

  .app-title{
    margin:0;
    padding:0;
    font-size: clamp(1.25rem, 1.8vw, 1.72rem);
    font-weight: 850;
    color: var(--txt) !important;
  }

  .app-subtitle{
    margin-top: .3rem;
    max-width: 720px;
    color: var(--muted) !important;
    font-size: .96rem;
    line-height: 1.45;
  }

  .topbar{
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(23,47,82,.92), rgba(20,43,75,.84));
    border-radius: var(--radius);
    padding: 1rem 1.05rem;
    box-shadow: var(--shadow);
  }

  .topbar-grid{
    display:grid;
    grid-template-columns: minmax(0, 1fr) minmax(320px, 520px);
    gap: 1rem;
    align-items: stretch;
  }

  .topbar-help-wrap{
    display:flex;
    justify-content:flex-end;
  }

  .topbar-help{
    width: 100%;
    min-height: 7.9rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
  }

  .help-card{
    border: 1px solid rgba(157,197,235,.16);
    background: rgba(255,255,255,.025);
    border-radius: var(--radius);
    padding: .85rem .95rem;
    box-sizing: border-box;
  }

  .help-title{
    color: var(--txt) !important;
    font-weight: 800;
    font-size: .95rem;
    margin-bottom: .4rem;
  }

  .subtle, .small{
    color: var(--muted-2) !important;
    font-size: .9rem;
    line-height: 1.45;
  }

  .pill-row{
    margin-top: .85rem;
    display:flex;
    gap:.55rem;
    flex-wrap:wrap;
    align-items:center;
  }

  .pill{
    display:inline-flex;
    align-items:center;
    gap:.45rem;
    min-height: 2rem;
    padding:.34rem .62rem;
    border-radius: 999px;
    border: 1px solid rgba(157,197,235,.18);
    background: rgba(255,255,255,.035);
    color: var(--muted) !important;
    font-size: .87rem;
    line-height: 1.2;
    max-width: 100%;
  }

  .pill b{
    color: var(--txt) !important;
    font-weight: 800;
    overflow-wrap: anywhere;
  }

  .section-heading{
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap: 1rem;
    margin: .25rem 0 .72rem;
    padding: .1rem .05rem .62rem;
    border-bottom: 1px solid rgba(157,197,235,.14);
  }

  .section-title{
    color: var(--txt) !important;
    font-weight: 850;
    font-size: 1.05rem;
    line-height: 1.25;
  }

  .section-subtitle{
    color: var(--muted-2) !important;
    font-size: .9rem;
    margin-top: .22rem;
    line-height: 1.42;
  }

  .section-aside{
    flex: 0 0 auto;
    display:flex;
    align-items:center;
    gap:.45rem;
  }

  .card{
    border: 1px solid var(--stroke);
    background: linear-gradient(180deg, rgba(23,47,82,.90), rgba(23,47,82,.74));
    border-radius: var(--radius);
    padding: 1.05rem;
    box-shadow: var(--shadow);
  }

  .card-title{
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap: .9rem;
    margin-bottom: .25rem;
  }

  .card-title > div:first-child{
    min-width: 0;
  }

  .kpis{
    display:grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .75rem;
  }

  .kpi{
    min-height: 7.2rem;
    border: 1px solid rgba(157,197,235,.16);
    background: rgba(255,255,255,.032);
    border-radius: var(--radius);
    padding: .85rem;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
    overflow: hidden;
  }

  .kpi .lbl{
    color: var(--muted-2) !important;
    font-size: .78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
  }

  .kpi .val{
    color: var(--txt) !important;
    font-size: clamp(1.08rem, 1.3vw, 1.32rem);
    font-weight: 850;
    margin-top: .55rem;
    line-height: 1.2;
    overflow-wrap: anywhere;
  }

  .kpi .hint{
    color: var(--muted) !important;
    font-size: .84rem;
    margin-top: .55rem;
    line-height: 1.35;
  }

  .badge{
    display:inline-flex;
    align-items:center;
    min-height: 1.5rem;
    padding: .2rem .55rem;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,.04);
    font-size: .8rem;
    font-weight: 800;
    color: var(--muted) !important;
  }

  .badge-idle{ border-color: rgba(246,195,95,.38); background: rgba(246,195,95,.10); color: #fff0c2 !important; }
  .badge-ok{ border-color: rgba(139,212,80,.36); background: rgba(139,212,80,.11); color: #d8ffe6 !important; }
  .badge-warn{ border-color: rgba(246,195,95,.38); background: rgba(246,195,95,.11); color: #fff0c2 !important; }
  .badge-err{ border-color: rgba(255,107,125,.38); background: rgba(255,107,125,.11); color: #ffdce7 !important; }
  .badge-run{ border-color: rgba(78,163,255,.40); background: rgba(78,163,255,.12); color: #d8ecff !important; }

  .status-dot{
    width: .75rem;
    height: .75rem;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid var(--stroke);
  }

  .dot-idle{ background: var(--warn); box-shadow: 0 0 0 3px rgba(246,195,95,.10); }
  .dot-run{  background: var(--accent); box-shadow: 0 0 0 3px rgba(78,163,255,.10); }
  .dot-ok{   background: var(--good); box-shadow: 0 0 0 3px rgba(139,212,80,.10); }
  .dot-warn{ background: var(--warn); box-shadow: 0 0 0 3px rgba(246,195,95,.10); }
  .dot-err{  background: var(--error); box-shadow: 0 0 0 3px rgba(255,107,125,.10); }

  .empty-state{
    border: 1px solid var(--stroke);
    border-left: 3px solid var(--accent-2);
    background: rgba(23,47,82,.74);
    border-radius: var(--radius);
    padding: .95rem 1rem;
    margin: .35rem 0;
  }

  .empty-title{
    color: var(--txt) !important;
    font-weight: 850;
    font-size: .96rem;
    line-height: 1.25;
  }

  .empty-copy{
    color: var(--muted) !important;
    font-size: .9rem;
    line-height: 1.45;
    margin-top: .28rem;
  }

  .empty-warn{ border-left-color: var(--warn); }
  .empty-error{ border-left-color: var(--error); }
  .empty-success{ border-left-color: var(--good); }

  .logbox{
    border: 1px solid rgba(157,197,235,.18);
    background: rgba(12,31,59,.70);
    border-radius: var(--radius);
    padding: .85rem .95rem;
    max-height: 250px;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: .84rem;
    line-height: 1.55;
    color: rgba(246,251,255,.88) !important;
  }

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
    min-height: 2.65rem;
    border-radius: var(--radius) !important;
    border: 1px solid rgba(157,197,235,.20) !important;
    background: linear-gradient(180deg, rgba(30,64,108,.98), rgba(22,49,86,.98)) !important;
    color: var(--txt) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,.18);
    font-weight: 750 !important;
    transition: background .14s ease, border-color .14s ease, transform .14s ease;
  }

  .stButton > button:hover,
  .stDownloadButton > button:hover,
  [data-testid="stBaseButton-secondary"]:hover,
  [data-testid="stBaseButton-primary"]:hover{
    border-color: var(--stroke-strong) !important;
    background: linear-gradient(180deg, rgba(38,78,128,.99), rgba(28,61,105,.99)) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
  }

  [data-testid="stBaseButton-primary"]{
    border-color: rgba(78,163,255,.52) !important;
    background: linear-gradient(180deg, rgba(78,163,255,.24), rgba(119,194,255,.13)) !important;
    color: #ffffff !important;
  }

  .stButton > button:focus,
  .stDownloadButton > button:focus,
  [data-testid="stBaseButton-secondary"]:focus,
  [data-testid="stBaseButton-primary"]:focus{
    box-shadow: 0 0 0 3px rgba(78,163,255,.14), 0 8px 20px rgba(0,0,0,.18) !important;
  }

  .stButton > button:disabled,
  .stDownloadButton > button:disabled{
    opacity: .48 !important;
    color: rgba(246,251,255,.60) !important;
    transform: none;
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
    border: 1px solid rgba(157,197,235,.20) !important;
    border-radius: var(--radius) !important;
  }

  .stTextInput input,
  .stNumberInput input,
  .stTextArea textarea{
    min-height: 2.55rem;
  }

  .stTextInput input:focus,
  .stNumberInput input:focus,
  .stTextArea textarea:focus{
    border-color: var(--stroke-strong) !important;
    box-shadow: 0 0 0 3px rgba(78,163,255,.12) !important;
  }

  .stTextInput label,
  .stNumberInput label,
  .stTextArea label,
  .stSelectbox label,
  .stMultiSelect label,
  .stFileUploader label,
  .stCheckbox label,
  .stRadio label,
  .stSlider label,
  .stMarkdown, .stCaption, small {
    color: var(--txt) !important;
  }

  [data-testid="stWidgetLabel"] p{
    color: var(--muted) !important;
    font-weight: 750;
    font-size: .9rem;
  }

  [data-testid="stFileUploaderDropzone"]{
    min-height: 7rem;
    transition: background .14s ease, border-color .14s ease;
  }

  [data-testid="stFileUploaderDropzone"]:hover {
    background: var(--widget-bg-hover) !important;
    border-color: var(--stroke-strong) !important;
  }

  [data-testid="stFileUploaderDropzone"] * {
    color: var(--txt) !important;
  }

  .stCheckbox [data-testid="stWidgetLabel"] p,
  .stRadio [data-testid="stWidgetLabel"] p,
  [data-testid="stFileUploaderDropzoneInstructions"] span,
  [data-testid="stFileUploaderDropzoneInstructions"] small,
  [data-testid="stFileUploaderFileName"] {
    color: var(--muted) !important;
  }

  [data-baseweb="checkbox"] > div,
  [data-baseweb="radio"] > div {
    background-color: transparent !important;
    border-color: var(--stroke-strong) !important;
  }

  [data-testid="stSlider"] [role="slider"]{
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
  }

  .stExpander {
    border: 1px solid var(--stroke) !important;
    border-radius: var(--radius) !important;
    background: rgba(23,47,82,.76) !important;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,.16);
  }

  .stExpander summary{
    font-weight: 800;
  }

  .stExpander summary, .stExpander details, .stExpander p {
    color: var(--txt) !important;
  }

  div[data-testid="stMetric"] {
    background: rgba(255,255,255,.032) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: var(--radius) !important;
    padding: .75rem .85rem !important;
    min-height: 6.2rem;
  }

  div[data-testid="stMetric"] label,
  div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 750 !important;
  }

  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--txt) !important;
    font-size: 1.75rem !important;
    line-height: 1.12 !important;
    overflow-wrap: anywhere;
  }

  .metric-card{
    min-height: 6.4rem;
    border: 1px solid var(--stroke);
    border-radius: var(--radius);
    background: rgba(255,255,255,.032);
    padding: .9rem .95rem;
    overflow: visible;
  }

  .metric-label{
    color: var(--muted) !important;
    font-size: .9rem;
    font-weight: 800;
    line-height: 1.25;
    display: flex;
    align-items: center;
    gap: .45rem;
    position: relative;
  }

  .metric-tooltip{
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 999px;
    border: 1px solid rgba(78,163,255,.45);
    background: rgba(78,163,255,.12);
    color: rgba(246,251,255,.96) !important;
    font-size: .72rem;
    font-weight: 850;
    cursor: help;
    user-select: none;
  }

  .metric-tooltip-text{
    position: absolute;
    z-index: 1000;
    left: 50%;
    bottom: calc(100% + .6rem);
    transform: translateX(-50%) translateY(.25rem);
    width: min(280px, 70vw);
    padding: .6rem .7rem;
    border-radius: var(--radius);
    border: 1px solid rgba(78,163,255,.36);
    background: rgba(12,31,59,.98);
    box-shadow: 0 14px 30px rgba(0,0,0,.38);
    color: rgba(246,251,255,.92) !important;
    font-size: .82rem;
    font-weight: 550;
    line-height: 1.35;
    text-align: left;
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transition: opacity .12s ease, transform .12s ease, visibility .12s ease;
  }

  .metric-tooltip:hover .metric-tooltip-text,
  .metric-tooltip:focus .metric-tooltip-text{
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0);
  }

  .metric-value{
    color: var(--txt) !important;
    font-size: clamp(1.7rem, 2.3vw, 2.1rem);
    line-height: 1.12;
    margin-top: .85rem;
    font-weight: 850;
    overflow-wrap: anywhere;
  }

  [data-testid="stDataFrame"] > div,
  .stDataFrame, .stTable {
    background: rgba(23,47,82,.78) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
  }

  [data-testid="stDataFrame"] *, .stTable * {
    color: var(--txt) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: .55rem;
    background: transparent !important;
    width: 100%;
  }

  .stTabs [data-baseweb="tab"] {
    width: 100%;
    min-height: 3rem;
    justify-content: center;
    border-radius: var(--radius) !important;
    background: rgba(255,255,255,.035) !important;
    color: var(--muted) !important;
    border: 1px solid var(--stroke) !important;
    padding: .7rem .65rem !important;
    font-size: .92rem !important;
    font-weight: 800 !important;
    text-align: center;
    white-space: normal;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(78,163,255,.15), rgba(119,194,255,.10)) !important;
    border-color: rgba(78,163,255,.48) !important;
    color: var(--txt) !important;
  }

  .stTabs [data-baseweb="tab"] p {
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    line-height: 1.15 !important;
    text-align: center;
  }

  .stTabs [data-baseweb="tab-highlight"]{
    display:none !important;
  }

  .stTabs [data-baseweb="tab-panel"]{
    padding-top: 1rem;
  }

  .stAlert {
    border-radius: var(--radius) !important;
    border: 1px solid var(--stroke) !important;
    background: rgba(23,47,82,.86) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.14);
  }

  .stAlert *, .stJson * {
    color: var(--txt) !important;
  }

  .stJson {
    background: rgba(12,31,59,.76) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: var(--radius) !important;
  }

  @media (max-width: 1100px){
    .topbar-grid{
      grid-template-columns: 1fr;
    }

    .topbar-help-wrap{
      justify-content:flex-start;
    }

    .topbar-help{
      min-height: auto;
      overflow: visible;
    }

    .kpis{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 760px){
    .block-container{
      padding-left: 1rem !important;
      padding-right: 1rem !important;
      padding-bottom: 1.8rem !important;
    }

    .topbar,
    .card{
      padding: .9rem;
    }

    .section-heading{
      align-items:flex-start;
      flex-direction:column;
      gap:.45rem;
    }

    .section-aside{
      width:100%;
      justify-content:flex-start;
    }

    .kpis{
      grid-template-columns: 1fr;
    }

    .stTabs [data-baseweb="tab-list"]{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .pill{
      width: 100%;
      justify-content: space-between;
    }
  }
</style>
"""


def inject_styles():
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    st.markdown(BASE_CSS, unsafe_allow_html=True)