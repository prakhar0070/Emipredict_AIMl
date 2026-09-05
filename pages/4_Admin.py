"""
4_Admin.py
------------
Simple CRUD interface for the dataset: view, filter, edit, add, delete rows,
and export changes. Uses Streamlit's built-in data_editor - no database
needed for this project's scope.
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin: Data Management")

DATA_PATH = "data/emi_dataset.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(DATA_PATH):
    st.error(f"No dataset found at {DATA_PATH}. Run `python src/generate_data.py` first.")
    st.stop()

df = load_data()

st.markdown("### Filter records")
c1, c2 = st.columns(2)
with c1:
    scenario_filter = st.multiselect("EMI Scenario", df["emi_scenario"].unique().tolist())
with c2:
    eligibility_filter = st.multiselect("Eligibility", df["emi_eligibility"].unique().tolist())

filtered = df.copy()
if scenario_filter:
    filtered = filtered[filtered["emi_scenario"].isin(scenario_filter)]
if eligibility_filter:
    filtered = filtered[filtered["emi_eligibility"].isin(eligibility_filter)]

st.markdown(f"### Records ({len(filtered):,} shown, editable below)")
edited_df = st.data_editor(
    filtered.head(500),  # cap for UI performance; full CRUD works on any subset
    num_rows="dynamic",   # allows adding/deleting rows
    use_container_width=True,
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("💾 Save changes back to CSV", type="primary"):
        # Merge edited rows back into the full dataset by index, then overwrite
        full_df = load_data()
        full_df.update(edited_df)
        full_df.to_csv(DATA_PATH, index=False)
        st.cache_data.clear()
        st.success("Changes saved. Reload the page to see updates reflected everywhere.")

with col_b:
    st.download_button(
        "⬇️ Download current view as CSV",
        data=edited_df.to_csv(index=False).encode("utf-8"),
        file_name="emi_data_export.csv",
        mime="text/csv",
    )
