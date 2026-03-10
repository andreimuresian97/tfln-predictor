import streamlit as st

# --- PAGE CONFIGURATION ---
# This must be the first Streamlit command. It sets the browser tab title and layout.
st.set_page_config(
    page_title="TFLN Design Hub",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- HERO SECTION ---
st.title("⚡ TFLN Engineering Dashboard")
st.markdown("**Advanced Surrogate Modeling & Optimization for Thin Film Lithium Niobate**")
st.markdown("---")

# --- EXECUTIVE SUMMARY ---
st.markdown("""
Welcome to the TFLN internal design platform. This suite replaces computationally expensive 
3D electromagnetic simulations (CST) with high-speed, machine-learning-driven surrogate models. 

By leveraging Gaussian Process regression, this tool enables instant performance inference, 
global geometric optimization, and inverse design, drastically reducing engineering cycles.

### 🧰 Available Modules
Please select a tool from the sidebar on the left to begin:
""")

# --- MODULE DESCRIPTIONS (2x2 Grid) ---
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.subheader("🔮 Predictor")
    st.markdown("""
    **Instant Inference.** Input T-rail geometry parameters to instantly predict the fundamental Figures of Merit: 
    * Half-wave Voltage ($V_\pi$)
    * Characteristic Impedance ($Z_0$)
    * Effective Index ($n_m$)
    """)

with row1_col2:
    st.subheader("🎯 Optimizer")
    st.markdown("""
    **Global Search.** Define your physical constraints and let the algorithm sweep millions of design combinations to 
    recommend the absolute optimal geometry for peak device performance.
    """)

# Adds a bit of vertical spacing between the rows
st.write("") 

with row2_col1:
    st.subheader("🔍 Inverse Designer")
    st.markdown("""
    **Goal-Seeking Engine.** Input your exact target FOMs and allowable tolerances. The model will scan the entire design space to back-calculate the specific physical geometries required to achieve your desired physics.
    """)

with row2_col2:
    st.subheader("📈 Plotter")
    st.markdown("""
    **Visual Analytics.** Generate interactive parametric sweeps and trade-off curves to visualize how sensitive the modulator's 
    performance is to specific manufacturing tolerances and geometric shifts.
    """)

st.markdown("---")

# --- FOOTER / SPECS ---
st.info("""
**System Note:** All current surrogate models are trained for a central operating wavelength of **1330 nm** and assume a fixed pitch of **200 µm**.
""")
