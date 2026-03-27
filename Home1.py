import streamlit as st

# =====================================================================
# 1. DEFINE THE HOME PAGE CONTENT
# =====================================================================
def home_page_content():
    # --- CUSTOM CSS FOR LARGER LINKS ---
    st.markdown("""
    <style>
    [data-testid="stPageLink-NavLink"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- HERO SECTION ---
    st.title("⚡ Ultimate TFLN Engineering Dashboard")
    st.markdown("**10-DOF Broadband Surrogate Modeling & RF Physics Cascade**")
    st.markdown("---")

    # --- EXECUTIVE SUMMARY ---
    st.markdown(r"""
    Welcome to Phase 2 of the TFLN internal design platform. This upgraded suite moves beyond raw electrostatics 
    to simulate the full broadband microwave photonic response. 

    By calculating the complete RF physics cascade—including velocity walk-off, impedance peaking reflections, and high-frequency 
    skin-effect attenuation—this tool accurately predicts true operational performance up to 150 GHz.

    ### 🧰 Available Modules
    Click on the module below (or use the sidebar) to begin:
    """)
    st.write("") 

    # --- MODULE DESCRIPTIONS ---
    # We safely link to the page object we define below, bypassing folder rules!
    st.page_link(ultimate_predictor_page, label="🔮 Ultimate Predictor", icon="▶️")

    st.markdown(r"""
    **Full-Physics Inference.** Input the complete 10 Degrees of Freedom to instantly predict: 
    * Pad-Referenced Half-wave Voltage (Vpi) at 60 GHz
    * Electro-Optic Bandwidth (up to 150 GHz)
    * Characteristic Impedance (Zc) & Microwave Index (nm)
    * Dynamic Rt Safety Floor (Ensuring S11 <= -10 dB)
    """)

    st.markdown("---")

    # --- FOOTER / SPECS ---
    st.info("""
    **System Note:** * Optical predictions are strictly trained for a central operating wavelength of **1330 nm**.
    * Physical geometries assume a fixed pitch of **200 µm**, WG of **70 µm**, and CAP_H of **1.4 µm**.
    * Real-world telecom driver impedance is explicitly modeled at **65 Ω** for reflection limits.
    """)

# =====================================================================
# 2. REGISTER PAGES & FORCE CUSTOM ROUTING
# =====================================================================
# We explicitly tell Streamlit where the files are, ignoring the default "pages/" folder
home_page = st.Page(home_page_content, title="Home", icon="🏠", default=True)
ultimate_predictor_page = st.Page("pages1/Ultimate_Predictor.py", title="Ultimate Predictor", icon="🔮")

# Initialize the router with ONLY these two pages
pg = st.navigation([home_page, ultimate_predictor_page])

# =====================================================================
# 3. RUN THE APP
# =====================================================================
st.set_page_config(
    page_title="Ultimate TFLN Hub",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Execute the isolated navigation
pg.run()
