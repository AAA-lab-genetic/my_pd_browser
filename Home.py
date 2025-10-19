# Home.py
import streamlit as st

st.set_page_config(page_title="PD Browser", page_icon="🇲🇾", layout="wide")
st.title("🇲🇾 PD Browser")
st.markdown("""
Welcome! Use the sidebar to navigate:
- **PCA**: Explore sample clustering and highlight participants.
- **Genetic DB**: Look up clinical/genetic fields per participant.
- **LRRK2 Spectrum**: Visualize mutational landscape for LRRK2.
""")
st.sidebar.success("Choose a page on the left.")
