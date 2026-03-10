import streamlit as st

SIDEBAR_CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stAppViewContainer"] { padding-top: 0rem; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.95), rgba(15,23,48,0.95)) !important;
  border-right: 1px solid rgba(255,255,255,0.12);
  box-shadow: 8px 0 30px rgba(0,0,0,0.35);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  padding-top: -1.2rem !important;
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
  }

  html, body, [data-testid="stAppViewContainer"]{
    background: radial-gradient(1200px 700px at 15% 10%, rgba(106,166,255,.14), transparent 60%),
                radial-gradient(900px 650px at 85% 20%, rgba(79,227,213,.12), transparent 55%),
                radial-gradient(900px 650px at 55% 95%, rgba(167,139,250,.12), transparent 55%),
                linear-gradient(180deg, var(--bg0), var(--bg1));
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
    color: var(--muted);
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
    color: var(--muted);
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

  .kpi .lbl{ color: var(--muted2); font-size: 0.85rem; }
  .kpi .val{ font-size: 1.25rem; font-weight: 700; margin-top: 4px; }
  .kpi .hint{ color: var(--muted); font-size: 0.88rem; margin-top: 6px; }

  .badge{
    display:inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid var(--stroke2);
    background: rgba(255,255,255,.04);
    font-size: .85rem;
    color: var(--muted);
  }

  .badge-idle{ border-color: rgba(255,200,87,.38); background: rgba(255,200,87,.09); color: rgba(255,230,180,.95); }
  .badge-ok{ border-color: rgba(186,252,90,.32); background: rgba(186,252,90,.10); color: rgba(233,255,205,.95); }
  .badge-warn{ border-color: rgba(255,200,87,.32); background: rgba(255,200,87,.10); color: rgba(255,230,180,.95); }
  .badge-err{ border-color: rgba(255,77,109,.34); background: rgba(255,77,109,.10); color: rgba(255,205,215,.95); }
  .badge-run{ border-color: rgba(79,227,213,.32); background: rgba(79,227,213,.10); color: rgba(205,255,248,.95); }

  .sidebar-title{
    font-size: 0.9rem;
    color: var(--muted2);
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
    color: var(--muted2);
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
    color: rgba(255,255,255,.82);
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
</style>
"""


def inject_styles():
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    st.markdown(BASE_CSS, unsafe_allow_html=True)