import streamlit as st
import pymupdf
from src.helper import PrintCost
import altair as alt
import pandas as pd

st.title("Print Cost", text_alignment="center")
st.text("A simple app to calculate or predict printing prices", text_alignment="center")

uploaded_files = st.file_uploader(
    "Upload your PDF file here",
    type=["pdf"],
    accept_multiple_files=False,
)

if st.button("Proceed"):
    if uploaded_files is not None:
        pdf_bytes = uploaded_files.read()

        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pc = PrintCost(pdf, "models/xgboost_98.64_cmy_k_cmyk_7_dpi.pkl")
        result = pc.predict(dpi=7)

        st.divider()

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Total Pages", result["total_pages"])
        col2.metric("Total Price", f"IDR {result['total_price']}", format="accounting")

        st.subheader("Pages by Category")
        df = pd.DataFrame(result["details"])

        color_map = {
            "Mono Print": "#FFF2EF",
            "Color Light": "#FFDBB6",
            "Color Standard": "#F7A5A5",
            "Color Heavy": "#5D688A",
            "Full Color – Dark & Mixed": "#88527F",
        }

        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(
                theta=alt.Theta(
                    field="pages", type="quantitative", title="Total Pages"
                ),
                color=alt.Color(
                    field="category",
                    type="nominal",
                    scale=alt.Scale(
                        domain=list(color_map.keys()),
                        range=list(color_map.values()),
                    ),
                    legend=alt.Legend(title="Print Category"),
                ),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("pages:Q", title="Pages"),
                    alt.Tooltip("subtotal:Q", title="Subtotal"),
                ],
            )
        )

        st.altair_chart(chart, width="stretch")

    else:
        st.warning("Please upload the file")
