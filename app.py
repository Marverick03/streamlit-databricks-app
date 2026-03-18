import streamlit as st
import pandas as pd
from backend.db import get_all_cases

st.set_page_config(page_title="Case Management System", layout="wide")

st.title("📋 Case Management System")
st.divider()

# -------------------------------
# Create New Case Button
# -------------------------------
col1, col2 = st.columns([8, 2])

with col2:
    if st.button("➕ Create New Case"):
        st.session_state.page = "create"

# -------------------------------
# Case List Table
# -------------------------------
st.subheader("All Cases")

cases = get_all_cases()

if cases:
    df = pd.DataFrame(
        cases,
        columns=["Case ID", "Title", "Status", "Created Date"]
    )
    st.dataframe(df, use_container_width=True)
else:
    st.info("No cases found.")
