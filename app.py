# app.py  —  UTC Factbook: State of Residence (single term)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ============== Branding, maps, helpers ==============
UTC_BLUE = "#112E51"
UTC_GOLD = "#FDB736"
BLACK    = "#000000"

NAME_TO_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District Of Columbia":"DC","Florida":"FL",
    "Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS",
    "Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI",
    "Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
    "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI",
    "South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","Puerto Rico":"PR"
}
CENTROIDS = {
    "AL":(-86.9,32.3),"AK":(-150.0,64.0),"AZ":(-111.7,34.3),"AR":(-92.3,35.1),"CA":(-119.6,36.5),
    "CO":(-105.6,39.0),"CT":(-72.7,41.6),"DE":(-75.5,39.0),"FL":(-81.5,27.7),"GA":(-83.4,32.7),
    "HI":(-157.5,20.9),"ID":(-114.1,44.2),"IL":(-89.3,40.0),"IN":(-86.1,39.9),"IA":(-93.3,42.1),
    "KS":(-98.4,38.5),"KY":(-85.3,37.5),"LA":(-91.9,31.0),"ME":(-69.2,45.4),"MD":(-76.6,39.0),
    "MA":(-71.8,42.3),"MI":(-84.6,44.7),"MN":(-94.6,46.3),"MS":(-89.7,32.7),"MO":(-92.5,38.5),
    "MT":(-110.4,46.9),"NE":(-99.9,41.5),"NV":(-116.7,39.6),"NH":(-71.5,43.9),"NJ":(-74.4,40.1),
    "NM":(-106.1,34.4),"NY":(-75.5,42.9),"NC":(-79.3,35.5),"ND":(-100.5,47.5),"OH":(-82.8,40.3),
    "OK":(-97.3,35.6),"OR":(-120.5,44.1),"PA":(-77.6,40.9),"RI":(-71.5,41.7),"SC":(-80.9,33.9),
    "SD":(-100.0,44.4),"TN":(-86.7,35.7),"TX":(-99.4,31.5),"UT":(-111.7,39.3),"VT":(-72.7,44.1),
    "VA":(-78.8,37.5),"WA":(-120.4,47.4),"WV":(-80.6,38.6),"WI":(-89.6,44.6),"WY":(-107.6,43.0),
    "DC":(-77.0,38.9),"PR":(-66.5,18.2)
}
TERM_MAP = {"Spring":20,"Summer":30,"Fall":40}
INV_TERM = {20:"Spring",30:"Summer",40:"Fall"}

def clean_state_name(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().title().replace(" Of ", " of ")
    return "District Of Columbia" if s == "District Of Columbia" else s

def to_abbr(name: str) -> str:
    return NAME_TO_ABBR.get(clean_state_name(name), "")

def parse_termstring_to_code(s):
    # "Spring 2025" -> 202520
    if not isinstance(s,str): return None
    parts = s.strip().split()
    if len(parts)!=2: return None
    term, year = parts[0].title(), parts[1]
    if term not in TERM_MAP: return None
    try: year = int(year)
    except: return None
    return year*100 + TERM_MAP[term]

def normalize_termyear(df: pd.DataFrame) -> pd.DataFrame:
    # if numeric, keep; if string like "Spring 2025", convert; or derive from AcademicYear+Term
    if "TermYear" in df.columns:
        if df["TermYear"].dtype.kind in "iu":
            return df
        df["TermYear"] = df["TermYear"].apply(lambda v: parse_termstring_to_code(v) if isinstance(v,str) else pd.to_numeric(v, errors="coerce"))
    if df.get("TermYear", pd.Series([None])).isna().all() and {"AcademicYear","Term"}.issubset(df.columns):
        df["TermYear"] = (pd.to_numeric(df["AcademicYear"], errors="coerce")*100 +
                          df["Term"].astype(str).str.title().map(TERM_MAP)).astype("Int64")
    return df

def term_label_from_code(code:int)->str:
    return f"{INV_TERM.get(code%100, code%100)} {code//100}"

# ============== Streamlit app ==============
st.set_page_config(page_title="UTC Factbook: State of Residence", layout="wide")

st.title("UTC Factbook: State of Residence (single term)")
st.caption("Select a term, render a Plotly choropleth with UTC branding, and export as HTML/PNG.")

# Remember defaults
if "default_path" not in st.session_state:
    st.session_state.default_path = "State_or_Territory_of_Residence.xlsx"
if "last_term" not in st.session_state:
    st.session_state.last_term = None

# --- File input row ---
left, right = st.columns([2,1])
with left:
    uploaded = st.file_uploader("Upload State_or_Territory_of_Residence.xlsx", type=["xlsx"])
with right:
    default_path = st.text_input("...or use a local path", st.session_state.default_path)
    use_local = st.checkbox("Use local path", value=False)

# --- Sidebar controls ---
st.sidebar.header("Controls")
show_labels = st.sidebar.checkbox("Show labels on map", value=True)
label_size  = st.sidebar.slider("Label size", 8, 24, 12)
contrast_th = st.sidebar.slider("Auto-contrast threshold", 0.40, 0.90, 0.60, 0.01)
min_label   = st.sidebar.number_input("Hide labels under count", min_value=0, max_value=10000, value=0, step=10)
exclude_geo = st.sidebar.multiselect("Exclude geographies", ["AK","HI","PR"], default=[])

topn = st.sidebar.slider("Top N table", 5, 25, 10)

@st.cache_data(show_spinner=False)
def load_xlsx(path_or_buf):
    return pd.read_excel(path_or_buf, sheet_name=0, engine="openpyxl")

# --- Read data ---
if uploaded:
    df = load_xlsx(uploaded)
    source = uploaded.name
elif use_local and Path(default_path).exists():
    df = load_xlsx(default_path)
    source = default_path
else:
    st.info("Upload the Excel or toggle ‘Use local path’ and point to an existing file.")
    st.stop()

required_any = [{"State_Name","Count","TermYear"},
                {"State_Name","Count","AcademicYear","Term"}]
if not any(req.issubset(df.columns) for req in required_any):
    st.error("File is missing required columns. Expect either "
             "`State_Name, Count, TermYear` or `State_Name, Count, AcademicYear, Term`.")
    st.stop()
st.toast("File loaded ✓", icon="✅")

# Normalize + coerce
df = normalize_termyear(df)
df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)

# Term selection
available_terms = sorted([int(x) for x in df["TermYear"].dropna().unique()])
if not available_terms:
    st.error("No valid TermYear values detected.")
    st.stop()

default_idx = available_terms.index(st.session_state.last_term) if st.session_state.last_term in available_terms else len(available_terms)-1
termyear = st.selectbox(
    "Select TermYear",
    options=available_terms,
    index=default_idx,
    format_func=lambda x: f"{x}  –  {term_label_from_code(int(x))}"
)
st.session_state.last_term = int(termyear)
st.session_state.default_path = default_path

st.divider()

# Filter + aggregate
cur = df[df["TermYear"] == int(termyear)].copy()
if cur.empty:
    st.warning("No rows for the selected term. Try another term.")
    st.stop()

cur["StateCode"] = cur["State_Name"].apply(to_abbr)
cur = cur[(cur["StateCode"]!="") & (~cur["StateCode"].isin(exclude_geo))].copy()
cur = cur.groupby("StateCode", as_index=False)["Count"].sum()
cur["lon"] = cur["StateCode"].map(lambda s: CENTROIDS.get(s,(None,None))[0])
cur["lat"] = cur["StateCode"].map(lambda s: CENTROIDS.get(s,(None,None))[1])
cur = cur.dropna(subset=["lon","lat"]).copy()

# Label formatting + adaptive contrast
z = cur["Count"]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
cur["LabelColor"] = ["white" if v > contrast_th else "black" for v in z_norm]
cur["Label"] = cur["StateCode"] + "<br>" + cur["Count"].map("{:,}".format)

# Plot
utc_colorscale = [[0.0, UTC_GOLD], [1.0, UTC_BLUE]]
fig = go.Figure()
chor = go.Choropleth(
    locations=cur["StateCode"],
    z=cur["Count"],
    locationmode="USA-states",
    colorscale=utc_colorscale,
    colorbar_title="Headcount",
    marker_line_color="white",
    marker_line_width=0.6,
    hovertemplate="<b>%{location}</b><br>Headcount: %{z:,}<extra></extra>"
)
fig.add_trace(chor)

if show_labels:
    cur_lbl = cur[cur["Count"] >= int(min_label)].copy()
    fig.add_trace(go.Scattergeo(
        lon=cur_lbl["lon"], lat=cur_lbl["lat"], text=cur_lbl["Label"], mode="text",
        textfont=dict(size=int(label_size)), textfont_color=cur_lbl["LabelColor"], hoverinfo="skip"
    ))

fig.update_layout(
    title=dict(text=f"<b>State of Residence — {term_label_from_code(int(termyear))}</b>",
               x=0.5, xanchor="center", font=dict(color=UTC_BLUE)),
    geo=dict(scope="usa", projection_type="albers usa", showcountries=False, showlakes=False, bgcolor="white"),
    margin=dict(l=0, r=0, t=50, b=0),
    paper_bgcolor="white", plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# Quick insights
st.subheader("Top States")
st.dataframe(cur[["StateCode","Count"]].sort_values("Count", ascending=False).head(int(topn)),
             use_container_width=True)

# Exports
out_html = f"factbook_state_map_{termyear}.html"
st.download_button("Download HTML", data=fig.to_html(include_plotlyjs="cdn"),
                   file_name=out_html, mime="text/html")

# PNG export (requires kaleido)
try:
    png_bytes = fig.to_image(format="png", scale=2)  # high-res
    st.download_button("Download PNG", data=png_bytes,
                       file_name=out_html.replace(".html", ".png"),
                       mime="image/png")
except Exception:
    st.caption("Tip: install `kaleido` for PNG export (already included above).")
