import streamlit as st

# --- PAGE CONFIGURATION ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Ultimate TFLN Hub",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR LARGER LINKS ---
# This forces the st.page_link text to act like a subheader
st.markdown("""
<style>
[data-testid="stPageLink-NavLink"] p {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1f77b4 !important; /* Gives it a nice clickable blue tint */
}
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.title("⚡ Ultimate TFLN Engineering Dashboard")
st.markdown("**10-DOF Broadband Surrogate Modeling & RF Physics Cascade**")
st.markdown("---")

# --- EXECUTIVE SUMMARY ---
st.markdown("""
Welcome to Phase 2 of the TFLN internal design platform. This upgraded suite moves beyond raw electrostatics 
to simulate the full broadband microwave photonic response. 

By calculating the complete RF physics cascade—including velocity walk-off, impedance peaking reflections, and high-frequency 
skin-effect attenuation—this tool accurately predicts true operational performance up to 150 GHz.

### 🧰 Available Modules
Click on the module below (or use the sidebar) to begin:
""")
st.write("") # Little bit of spacing

# --- MODULE DESCRIPTIONS ---
# Make sure your file in pages1 is named exactly like this!
st.page_link("pages1/1_🔮_Ultimate_Predictor.py", label="🔮 Ultimate Predictor", icon="▶️")
st.markdown("""
**Full-Physics Inference.** Input the complete 10 Degrees of Freedom (Geometry + Device Length + Test/Telecom Impedances) to instantly predict: 
* Pad-Referenced Half-wave Voltage ($V_\pi$) at 60 GHz
* Electro-Optic Bandwidth (up to 150 GHz)
* Characteristic Impedance ($Z_c$) & Microwave Index ($n_m$)
* Dynamic $R_t$ Safety Floor (Ensuring $S_{11} \le -10$ dB)
""")

st.markdown("---")

# --- FOOTER / SPECS ---
st.info("""
**System Note:** * Optical predictions are strictly trained for a central operating wavelength of **1330 nm**.
* Physical geometries assume a fixed pitch of **200 µm**, WG of **70 µm**, and CAP_H of **1.4 µm**.
* Real-world telecom driver impedance is explicitly modeled at **65 Ω** for reflection limits.
""")
