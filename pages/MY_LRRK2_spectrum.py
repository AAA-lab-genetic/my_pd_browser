import re
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="LRRK2 Spectrum", page_icon="🧪", layout="wide")
st.title("🧪 LRRK2 Mutational Spectrum in Malaysia")
st.sidebar.title("🇲🇾 MY LRRK2 spectrum")

# -------------------------
# Google Sheets (public CSV)
# -------------------------
def load_public_csv(sheet_id: str, gid: str = "0") -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url)

# === Your sheet IDs (first tab gid="0") ===
SHEET_ID_VARIANT  = "1IVhDBzDEWdaHKxwIPDL8kRn6c6VtYfJzw5DvzBzcZbE"  # variant sheet you gave
SHEET_ID_ANNOVAR  = "1KtO_SG2fR6F8CcCTXAIn5OIooH-nbX7NOe2GgJ0rOUo"
SHEET_ID_CARRIERS = "1mdD-MYXV4-iF56tcLXYkxEWH_czJwb1ohsVgcrhJw2Q"
SHEET_ID_MAF      = "1MZ5RW_3UhOdu6M7z7GxMUyId66g8SIsVv8FzH1hy4xI"

if st.button("🔄 Refresh Google Sheets"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=300)
def load_variants():
    df = load_public_csv(SHEET_ID_VARIANT, "0")
    df.columns = [c.strip() for c in df.columns]
    # Expect: variant_id, aa_pos, impact(optional), count(optional)
    if "aa_pos" not in df.columns:
        # fallback: parse position out of variant_id or aa_change
        def parse_aa_pos(x):
            if not isinstance(x, str): return None
            m = re.search(r"(\d+)", x)
            return int(m.group(1)) if m else None
        col = "aa_change" if "aa_change" in df.columns else "variant_id"
        df["aa_pos"] = df[col].apply(parse_aa_pos)
    df = df.dropna(subset=["variant_id", "aa_pos"]).copy()
    df["aa_pos"] = df["aa_pos"].astype(int)
    if "impact" not in df.columns:
        df["impact"] = "missense"
    if "count" not in df.columns:
        df["count"] = 1
    return df

@st.cache_data(ttl=300)
def load_maf():
    df = load_public_csv(SHEET_ID_MAF, "0")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(ttl=300)
def load_annovar():
    df = load_public_csv(SHEET_ID_ANNOVAR, "0")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(ttl=300)
def load_carriers():
    df = load_public_csv(SHEET_ID_CARRIERS, "0")
    df.columns = [c.strip() for c in df.columns]
    return df

variant_df  = load_variants()
maf_df      = load_maf()
annovar_df  = load_annovar()
carriers_df = load_carriers()

# -------------------------
# Friendly schema checks
# -------------------------
def need(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"**{label}** missing: {missing}")
    return missing

problems = []
problems += need(variant_df, ["variant_id","aa_pos"], "Variant sheet")
problems += need(maf_df,     ["variant_id","ancestry","phenotype","maf"], "MAF sheet")
problems += need(annovar_df, ["variant_id"], "ANNOVAR sheet")
if not (("IID" in carriers_df.columns) or ("UMID" in carriers_df.columns)):
    st.error("**Carriers** needs at least 'IID' or 'UMID'.")
problems += need(carriers_df, ["variant_id"], "Carriers sheet")
if problems:
    st.stop()

# -------------------------
# Domain definitions (your exact spec)
# -------------------------
DOMAINS = [
    ("ARM",    12,   704,  "#264D27"),
    ("ANK",    705,  795,  "#D5E7C0"),
    ("LRR",    800,  1305, "#3F48CC"),
    ("ROC",    1335, 1510, "#FFFF66"),
    ("COR",    1511, 1878, "#F5B041"),
    ("MAPKKK", 1879, 2138, "#D6DBDF"),
    ("WD40",   2142, 2497, "#E74C3C"),
]

# ---------- Dash Bio–like needle style renderer ----------
def render_needleplot(
    variant_df: pd.DataFrame,
    domains: list[tuple[str, int, int, str]],
    needleStyle: dict | None = None,
    domainStyle: dict | None = None,
    y_max: float | None = None,
):
    """
    variant_df must have: variant_id (str), aa_pos (int)
    optional: count (numeric), impact (str) to map symbols/colors
    """
    import plotly.graph_objects as go
    from itertools import cycle

    needleStyle = needleStyle or {}
    domainStyle = domainStyle or {}

    # --- Needle style
    head_size      = float(needleStyle.get("headSize", 8))
    stem_thickness = float(needleStyle.get("stemThickness", 1.2))
    stem_color     = str(needleStyle.get("stemColor", "#222"))
    stem_const     = bool(needleStyle.get("stemConstHeight", False))
    head_symbol_in = needleStyle.get("headSymbol", "circle")   # str OR list
    head_color_in  = needleStyle.get("headColor",  "#1f77b4")  # str OR list

    # --- Domain style
    display_minor         = bool(domainStyle.get("displayMinorDomains", True))
    minor_width_threshold = int(domainStyle.get("minorWidthThreshold", 10))
    rangeslider           = bool(domainStyle.get("rangeSlider", False))
    xlabel                = domainStyle.get("xlabel", "Amino-acid position")
    ylabel                = domainStyle.get("ylabel", "# of Mutations")
    band                  = float(domainStyle.get("bandHeight", 0.12))  # fixed band at bottom

    # --- Working df / heights
    df = variant_df.copy()
    if "count" not in df.columns:
        df["count"] = 1.0
    if stem_const:
        const_height = y_max if y_max is not None else max(1.0, float(df["count"].max()))
        df["_height"] = const_height
    else:
        df["_height"] = df["count"].astype(float)
    y_max_eff = y_max if y_max is not None else max(float(df["_height"].max()), 5.0)

    # --- Figure
    fig = go.Figure()

    # Domains as fixed band (paper coords)
    for name, start, end, color in domains:
        width = end - start + 1
        if not display_minor and width < minor_width_threshold:
            continue
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=start, x1=end, y0=0.0, y1=band,
            fillcolor=color, opacity=0.95,
            line=dict(color="black", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=(start + end) / 2, y=band / 2,
            xref="x", yref="paper",
            text=name, showarrow=False,
            font=dict(size=12, color="black"),
        )

    # Stems (data coords)
    for _, r in df.iterrows():
        fig.add_shape(
            type="line",
            x0=int(r["aa_pos"]), x1=int(r["aa_pos"]),
            y0=0, y1=float(r["_height"]),
            line=dict(color=stem_color, width=stem_thickness),
            layer="below",
        )

    # Map symbols/colors:
    def map_values(values, default, series: pd.Series | None):
        if isinstance(values, list):
            if series is not None and series.nunique() > 1:
                cats = sorted(series.astype(str).unique())
                lut = {c: values[i % len(values)] for i, c in enumerate(cats)}
                return [lut[str(v)] for v in series.astype(str)]
            # only one category → cycle across rows so you still see variety
            cyc = cycle(values)
            return [next(cyc) for _ in range(len(df))]
        return values  # single value

    impact_series = df["impact"] if "impact" in df.columns else None
    marker_symbol = map_values(head_symbol_in, "circle", impact_series)
    marker_color  = map_values(head_color_in,  "#1f77b4", impact_series)

    # Heads (markers)
    fig.add_trace(go.Scatter(
        x=df["aa_pos"],
        y=df["_height"],
        mode="markers",
        marker=dict(size=head_size, symbol=marker_symbol, color=marker_color,
                    line=dict(width=1, color="white")),
        text=df["variant_id"],
        hovertemplate="Variant: %{text}<br>AA pos: %{x}<br>Height: %{y}<extra></extra>",
        name="Variants",
        showlegend=False,
    ))

    # Axes/layout
    xmax = max([end for *_, end, _ in [(d[0], d[1], d[2], d[3]) for d in domains]] + [int(df["aa_pos"].max())])
    xmin = min([start for _, start, _, _ in domains] + [int(df["aa_pos"].min())])
    fig.update_xaxes(title=xlabel, range=[xmin - 10, xmax + 10], showgrid=False,
                     rangeslider=dict(visible=rangeslider))
    fig.update_yaxes(title=ylabel, range=[0, y_max_eff], zeroline=True)
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=10))
    return fig



# -------------------------
# Controls
# -------------------------
left, right = st.columns([3, 2])
with left:
    st.markdown("##### Variant selector (optional)")
    # Sort variants by numeric pos then label
    def pos_key(v):
        m = re.search(r"\d+", str(v))
        return (m is None, int(m.group()) if m else 0, str(v))
    variant_choices = sorted(variant_df["variant_id"].unique().tolist(), key=pos_key)
    selected_variant = st.selectbox("Pick a variant (or click a dot below)", options=variant_choices, index=0 if variant_choices else None)
with right:
    ymax_default = float(max(variant_df["count"].max(), 5))
    y_max = st.number_input("Needle height max (y-axis)", value=ymax_default, min_value=1.0, step=1.0)

# -------------------------
# NeedlePlot-style figure with Dash Bio–like styling
# -------------------------
needleStyle = {
    "headSize": 10,
    "stemThickness": 3,
    "stemColor": "#CCC",
    "stemConstHeight": False,  # set True to make all needles same height
    "headSymbol": ["circle", "square", "triangle-up", "diamond"],  # cycles by 'impact'
    # "headColor": ["#1f77b4", "purple"],  # optional list; cycles by 'impact'
}
domainStyle = {
    "displayMinorDomains": True,
    "minorWidthThreshold": 10,     # used only if displayMinorDomains=False
    "rangeSlider": False,
    "xlabel": "Sequence of the protein",
    "ylabel": "# of Mutations",
}

fig = render_needleplot(
    variant_df=variant_df,
    domains=DOMAINS,
    needleStyle={
        "headSize": 10,
        "stemThickness": 3,
        "stemColor": "#CCC",
        "stemConstHeight": False,
        "headSymbol": ["circle", "square", "triangle-up", "diamond"],  # will map/cycle
        # "headColor": ["#1f77b4", "purple"],  # optional
    },
    domainStyle={
        "displayMinorDomains": True,
        "minorWidthThreshold": 10,
        "rangeSlider": False,
        "xlabel": "Sequence of the protein",
        "ylabel": "# of Mutations",
        "bandHeight": 0.12,  # domain strip height
    },
    y_max=y_max,  # from your number_input
)

st.plotly_chart(fig, use_container_width=True)
st.markdown(f"**Selected variant:** {selected_variant}")  # from your dropdown
st.markdown("---")

# -------------------------
# Panels
# -------------------------

# 1) MAF by ancestry & phenotype
st.markdown("#### 1) Minor Allele Frequency (by ancestry & phenotype)")
maf_sub = maf_df[maf_df["variant_id"] == selected_variant].copy()
if not maf_sub.empty:
    pheno_order = ["Control","Case","Total"]
    maf_sub["phenotype"] = pd.Categorical(maf_sub["phenotype"], categories=pheno_order, ordered=True)
    maf_sub.sort_values(["ancestry","phenotype"], inplace=True)
    fig_maf = px.bar(maf_sub, x="ancestry", y="maf", color="phenotype",
                     barmode="group", labels={"maf":"MAF","ancestry":"Ancestry","phenotype":"Group"})
    fig_maf.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_maf, use_container_width=True)
    with st.expander("Show MAF table"):
        st.dataframe(maf_sub.reset_index(drop=True), use_container_width=True)
else:
    st.info("No MAF rows found for this variant.")

# 2) ANNOVAR annotation
st.markdown("#### 2) ANNOVAR annotation")
anno_row = annovar_df[annovar_df["variant_id"] == selected_variant]
if not anno_row.empty:
    col_left  = [c for c in ["gene","exonicfunc","aa_change","clinvar_sig"] if c in anno_row.columns]
    col_right = [c for c in ["chr","pos","ref","alt","sift_pred","polyphen_pred","cadd_phred","gnomad_af"] if c in anno_row.columns]
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Core**")
        st.table(anno_row[col_left].T.rename(columns={anno_row.index[0]:""}))
    with a2:
        st.markdown("**Additional**")
        st.table(anno_row[col_right].T.rename(columns={anno_row.index[0]:""}))
    with st.expander("View full ANNOVAR row"):
        st.dataframe(anno_row, use_container_width=True)
else:
    st.info("No ANNOVAR annotation found for this variant.")

# 3) Carriers
st.markdown("#### 3) Carriers")
carriers_sub = carriers_df[carriers_df["variant_id"] == selected_variant].copy()
if not carriers_sub.empty:
    id_cols = [c for c in ["IID","UMID"] if c in carriers_sub.columns]
    show_cols = id_cols + [c for c in ["ancestry","phenotype"] if c in carriers_sub.columns]
    show_cols = show_cols or carriers_sub.columns.tolist()
    st.dataframe(carriers_sub[show_cols], use_container_width=True)
    st.download_button(
        "Download carriers (CSV)",
        data=carriers_sub.to_csv(index=False),
        file_name=f"{selected_variant}_carriers.csv",
        mime="text/csv",
    )
else:
    st.info("No carriers found for this variant.")

