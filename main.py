# main.py
# ------------------------------------------------------------
# Streamlit UI for running movie classification using
# Naive Bayes (Multinomial + Gaussian) and visualizing results.
# ------------------------------------------------------------

import streamlit as st
from compare_and_visualize import run_full_evaluation

st.title("🎬 Movie Review Classification using Bayesian Methods")
st.write("This app uses IMDB dataset and applies Naive Bayes models.")

st.subheader("Run Evaluation")

if st.button("Start Analysis"):
    with st.spinner("Processing dataset and running models..."):
        results = run_full_evaluation()

    st.success("Analysis Completed!")

    # Show accuracy of both models
    st.subheader("Model Performance Summary")
    st.write(results["summary_df"])

    # Show charts
    st.subheader("Performance Comparison Charts")
    st.image("comparison_chart.png")
    st.image("confusion_matrix_multinomial.png")
    st.image("confusion_matrix_gaussian.png")

st.info("Make sure your IMDB_Dataset.csv is in the same folder as these files.")
