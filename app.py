import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

st.title("Sanity Check")
st.write("If you see this, the app works.")
st.write(f"Matplotlib Backend: {matplotlib.get_backend()}")
