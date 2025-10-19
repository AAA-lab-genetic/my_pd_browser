import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import re

st.set_page_config(page_title="PD Browser", page_icon="🇲🇾", layout="wide")
st.title("🇲🇾 PCA plot of Malaysian PD samples")
st.sidebar.title("🇲🇾 MY PCA Plot")

# --- Information & user options ---
st.markdown("""
### 🧬 PD Browser: PCA Sample Search

You can:
- **Search samples** by:
  - 🧾 **IID** (e.g. UMKLM_002012_s1)
  - 🧍‍♂️ **UMID** (e.g. PD-0023, the local ID)
  - 🧠 **NAME** (participant name)

- **Color the PCA plot** by:
  - 🌏 `SG10K_LABEL` (Ancestry predicted using SG10K panel)
  - 👨‍👩‍👧‍👦 `RACE` (Self-reported race)

Simply select your preferred search type and color scheme below.        
""")

# --- Data ---
df_pred = pd.read_csv("pca_plot_fin.txt", sep="\t")

# --- User input for search type and color ---
c0, c1, c2, c3 = st.columns([2, 3, 2, 1])

with c0:
    search_field = st.selectbox(
        "Search by:",
        options=["IID", "UMID", "NAME"],
        index = 0,
        help="Choose which field to match your search terms against."
    )
    
with c1:
    color_field = st.selectbox(
        "Color PCA by:",
        options=["SG10K_LABEL", "RACE"],
        index=0,
        help="Choose which variable to color-code the PCA plot."
    )
    
with c2:
    
    exact = st.toggle("Exact match only", value=False, help="Off = partial match")
    
with c3:
    clear_highlight = st.button("Clear highlights")
    
    
# --- Search bar ---
search_raw = st.text_input(
    f"Enter {search_field}(s), comma-separated",
    placeholder=f"e.g. UMKLM_002012_s1 or partial {search_field}",
)

# --- PCA plot ---
fig = px.scatter_3d(
    df_pred,
    x="PC1", y="PC2", z="PC3",
    color=color_field,
    hover_name="IID",
    title=f"3D PCA Plot colored by {color_field}",
)
fig.update_traces(marker=dict(size=2))
fig.update_layout(
    legend=dict(itemsizing="constant", font=dict(size=16)),
    margin=dict(l=10, r=10, t=60, b=10),
)

# --- Highlight logic (dynamic search field) ---
def find_matches(terms, field, exact=False):
    if not terms or field not in df_pred:
        return df_pred.iloc[0:0]
    if exact:
        lower = set(t.lower() for t in terms)
        return df_pred[df_pred[field].str.lower().isin(lower)]
    pattern = "|".join(re.escape(t) for t in terms)
    matches = df_pred[df_pred[field].str.contains(pattern, case=False, na=False)]
    st.write(matches[["IID", "UMID", "NAME", "IC", "SG10K_LABEL", "RACE"]])
    return matches

highlight = pd.DataFrame()
if not clear_highlight and search_raw:
    terms = [t.strip() for t in search_raw.split(",") if t.strip()]
    highlight = find_matches(terms, search_field, exact=exact)
    
# --- Add highlight trace ---
if not highlight.empty:
    fig.add_scatter3d(
        x=highlight["PC1"], y=highlight["PC2"], z=highlight["PC3"],
        mode="markers+text",
        marker=dict(size=8, color="red", line=dict(width=1)),
        text=highlight["IID"],
        textposition="top center",
        name=f"Highlighted ({len(highlight)})",
        hoverinfo="skip",
        showlegend=True,
    )
    st.success(f"Highlighted {len(highlight)} sample(s) by `{search_field}`. Others remain visible.")
    
    
    

elif search_raw and not clear_highlight:
    st.warning(f"No matches found for `{search_field}`.")

# Render and remember camera after render
out = st.plotly_chart(fig, use_container_width=True)
# NOTE: Streamlit doesn't expose camera directly post-render; but if you
# later capture relayout events (via Plotly events), set st.session_state["camera"].

# Optional: allow download of current view as standalone HTML
html_str = pio.to_html(fig, include_plotlyjs=True, full_html=True)
st.download_button(
    "Download interactive HTML",
    data=html_str,
    file_name="pca_search.html",
    mime="text/html",
)
