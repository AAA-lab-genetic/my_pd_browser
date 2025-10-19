import pandas as pd
import streamlit as st
import re
from streamlit_gsheets import GSheetsConnection

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="MY Patient Database", page_icon="👨‍⚕️", layout="wide")
st.title("MY Patient Database")
st.sidebar.title("👨‍⚕️ MY Patient Database")

# --- Information & user options ---
st.markdown("""
### 🧬 Patient's genetic browser:

You can:
- **Search samples** by:
  - 🧾 **IID** (e.g. UMKLM_002012_s1)
  - 🧍‍♂️ **UMID** (e.g. PD-0023, the local ID)
  - 🧠 **NAME** (participant name)

- **The genotype output is denoted** by:
  - 0: Homozygote for reference allele
  - 1: Heterozygote
  - 2: Homozygote for alternate allele
  - NA: No genotyping data

Simply select your preferred search type below.        
""")

# -------------------------
# Google Sheets
# -------------------------
url = "https://docs.google.com/spreadsheets/d/1_dHirW4xjxC5K3KplGXOajsY5jjlspfgt80xPp9nWhk/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=300)
def load_sheet(u: str) -> pd.DataFrame:
    return conn.read(spreadsheet=u)

df = load_sheet(url)

if st.button("🔄 Refresh Google Sheets"):
    st.cache_data.clear()
    st.rerun()


# --- Define the gene list ---

gene_list = ["All gene", "PINK1", "LRRK2", "SNCA"]


# --- Controls ---
c0, c1, c2, c3 = st.columns([2, 3, 2, 2])
with c0:
    search_field = st.selectbox("Search by:", ["IID", "UMID", "NAME"], index=0)
with c1:
    gene_select = st.selectbox("Select gene to view:", gene_list,  index=0)
with c2:
    exact = st.toggle("Exact match only", value=False, help="Off = partial match")
with c3:
    clear_selection = st.button("Clear selection")

search_raw = st.text_input(
    f"Enter {search_field}(s), comma-separated",
    placeholder=f"e.g. UMKLM_002012_s1 or partial {search_field}",
)

# --- Helper: safe column subset (avoid KeyError if columns are missing) ---
def safe_subset(frame: pd.DataFrame, want: list[str]) -> pd.DataFrame:
    have = [c for c in want if c in frame.columns]
    return frame[have] if have else frame

# --- Matching logic ---
def find_matches(terms, field, exact=False) -> pd.DataFrame:
    # Guard: empty terms or missing column
    if not terms or field not in df.columns:
        return df.iloc[0:0]

    col = df[field].astype(str)  # ensure string ops
    if exact:
        lower = {t.lower() for t in terms if t}
        out = df[col.str.lower().isin(lower)]
    else:
        pattern = "|".join(re.escape(t) for t in terms if t)
        if not pattern:
            return df.iloc[0:0]
        out = df[col.str.contains(pattern, case=False, na=False)]

    # Optional gene filter
    if gene_select != "All gene" and "GENE" in df.columns:
        out = out[out["GENE"] == gene_select]

    return out

# --- Run search ---
selected = pd.DataFrame()
if clear_selection:
    st.info("Selection cleared.")
elif search_raw:
    terms = [t.strip() for t in search_raw.split(",") if t.strip()]
    selected = find_matches(terms, search_field, exact=exact)

# --- Display ---
if not selected.empty:
    st.success(f"Selected {len(selected)} sample(s) by `{search_field}`"
               + (f" · GENE={gene_select}" if gene_select != "All gene" else ""))
    cols_to_show = ["IID", "UMID", "NAME", "IC", "GENE", "VARIANT", "GENOTYPE", "SOURCES", "VALIDATION"]
    st.dataframe(safe_subset(selected, cols_to_show), use_container_width=True)

    # Optional: download selected
    csv = selected.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download selected as CSV", data=csv, file_name="selected_rows.csv", mime="text/csv")
else:
    if search_raw and not clear_selection:
        st.warning(f"No matches found for `{search_field}`"
                   + (f" with GENE={gene_select}" if gene_select != "All gene" else "."))
