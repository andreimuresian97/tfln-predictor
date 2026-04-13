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
    st.title("⚡ Ultimate TFLN Engineering Dashboard (Phase 3)")
    st.markdown("**11-DOF Broadband Surrogate Modeling & 3D RF Physics Cascade**")
    st.markdown("---")

    # --- EXECUTIVE SUMMARY ---
    st.markdown(r"""
    Welcome to Phase 3 of the TFLN internal design platform. This upgraded suite moves beyond legacy 2D approximations 
    to simulate the full broadband microwave photonic response, including flawlessly decoupled 3D periodic scattering and Ohmic relief. 

    By calculating the complete RF physics cascade—including exact velocity walk-off, impedance peaking reflections, 
    local Bayesian Epistemic Uncertainty, and L1 duty-cycle optical phase scaling—this tool accurately predicts true operational performance up to 150 GHz.

    ### 🧰 Available Modules
    Click on a module below (or use the sidebar) to begin:
    """)
    st.write("") 

    # --- MODULE DESCRIPTIONS (2x2 Grid) ---
    col1, col2 = st.columns(2)

    with col1:
        st.page_link(ultimate_predictor_page, label="🔮 Ultimate Predictor", icon="▶️")
        st.markdown(r"""
        **Full-Physics Bayesian Inference.** Input the complete 11 Degrees of Freedom to instantly predict: 
        * Pad-Referenced Half-wave Voltage (Vpi) at 60 GHz with L1 Duty-Cycle corrections
        * Electro-Optic Bandwidth (up to 150 GHz)
        * Characteristic Impedance (Zc) & Microwave Index (nm)
        * 3D RF Attenuation (Decoupled 2D Heat, 3D Ohmic Relief, and 3D Pure Radiation)
        * Rigorous Bayesian 95% Confidence Intervals & Propagated MAE
        """)
        
        st.write("")
        st.write("")

        st.page_link(ultimate_inverse_page, label="🔍 Inverse Synthesizer", icon="▶️")
        st.markdown(r"""
        **Goal-Seeking Engine.** Input your exact target broadband FOMs and physical tolerances. 
        * Floods the 11-DOF space using Quasi-Monte Carlo Sobol sequences.
        * Sifts through 250,000 extreme combinations via machine learning.
        * Uses SLSQP Gradient Polishing to back-calculate the exact optimal 3D geometries.
        """)

    with col2:
        st.page_link(ultimate_optimizer_page, label="🎯 Ultimate Optimizer", icon="▶️")
        st.markdown(r"""
        **Memetic Global Search.** Define your physical boundary limits and performance targets. 
        * Explores the 11-DOF space using the NSGA-II Genetic Algorithm.
        * Maps the fully-constrained Pareto Front (Bandwidth vs. Vpi).
        * Uses SLSQP Gradient Polishing to maximize Zc and perfectly lock velocity matching to the optical mode.
        """)
        
        st.write("")
        st.write("")

        st.page_link(ultimate_curves_page, label="📈 Parametric Curves Plotter", icon="▶️")
        st.markdown(r"""
        **Interactive Sweep Visualizer.** Instantly see how sweeping any geometric parameter shifts your figures of merit. 
        * Isolates individual DOFs while holding your baseline machine constant.
        * Dynamically calculates the full RF cascade and non-linear physics across the sweep.
        * Renders interactive Plotly curves with shaded local Bayesian 95% Confidence Intervals.
        """)

    st.markdown("---")

    # --- FOOTER / SPECS ---
    st.info("""
    **System Note:** * Optical predictions are strictly trained for a central operating wavelength of **1330 nm**.
    * Physical geometries assume a fixed pitch of **200 µm**, WG of **70 µm**, and CAP_H of **1.4 µm**.
    * Real-world telecom driver impedance is explicitly modeled at **65 Ω** for reflection limits.
    * `ETCH_DEPTH` routing is now natively integrated into all optical and RF attenuation surrogates.
    """)

# =====================================================================
# 2. REGISTER PAGES & FORCE CUSTOM ROUTING (Updated to pages2)
# =====================================================================
# We explicitly tell Streamlit where the files are, mapping to the "pages2/" folder
home_page = st.Page(home_page_content, title="Home", icon="🏠", default=True)
ultimate_predictor_page = st.Page("pages2/Ultimate_Predictor.py", title="Ultimate Predictor", icon="🔮")
ultimate_optimizer_page = st.Page("pages2/Ultimate_Optimizer.py", title="Ultimate Optimizer", icon="🎯")
ultimate_inverse_page = st.Page("pages2/Ultimate_Inverse_Designer.py", title="Ultimate Inverse Designer", icon="🔍")
ultimate_curves_page = st.Page("pages2/Ultimate_Curves_Plotter.py", title="Ultimate Curves Plotter", icon="📈")

# Initialize the router with ALL pages
pg = st.navigation([
    home_page, 
    ultimate_predictor_page, 
    ultimate_optimizer_page, 
    ultimate_inverse_page, 
    ultimate_curves_page
])

# =====================================================================
# 3. RUN THE APP
# =====================================================================
st.set_page_config(
    page_title="Ultimate TFLN Hub",
    page_icon="⚡",
    layout="wide",  
    initial_sidebar_state="expanded"
)

# Execute the isolated navigation
pg.run()
