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
    col1, col2 = st.columns(2)

    with col1:
        st.page_link(ultimate_predictor_page, label="🔮 Ultimate Predictor", icon="▶️")
        st.markdown(r"""
        **Full-Physics Inference.** Input the complete 10 Degrees of Freedom to instantly predict: 
        * Pad-Referenced Half-wave Voltage (Vpi) at 60 GHz
        * Electro-Optic Bandwidth (up to 150 GHz)
        * Characteristic Impedance (Zc) & Microwave Index (nm)
        * Dynamic Rt Safety Floor (Ensuring S11 <= -10 dB)
        """)

    with col2:
        st.page_link(ultimate_optimizer_page, label="🎯 Ultimate Optimizer", icon="▶️")
        st.markdown(r"""
        **Memetic Global Search.** Define your physical bounds and performance targets. 
        * Explores 10-DOF space using NSGA-II Genetic Algorithm.
        * Maps the Pareto Front (Bandwidth vs. Vpi).
        * Uses SLSQP Gradient Polishing to maximize Zc and perfectly lock velocity matching.
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
ultimate_optimizer_page = st.Page("pages1/Ultimate_Optimizer.py", title="Ultimate Optimizer", icon="🎯")

# Initialize the router with ONLY these pages
pg = st.navigation([home_page, ultimate_predictor_page, ultimate_optimizer_page])

# =====================================================================
# 3. RUN THE APP
# =====================================================================
st.set_page_config(
    page_title="Ultimate TFLN Hub",
    page_icon="⚡",
    layout="wide",  # "wide" looks great with the 2-column layout!
    initial_sidebar_state="expanded"
)

# Execute the isolated navigation
pg.run()
