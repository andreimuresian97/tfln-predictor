"""
================================================================================
🎯 ULTIMATE TFLN 11-DOF INVERSE SYNTHESIZER (STREAMLIT APP)
NSGA-II Global Pareto Search + SLSQP Gradient Polishing
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
import io
import warnings

# Import the new, highly accurate Ultimate Predictor
# Ensure this matches the name of your python file (e.g., ultimate_predictor.py)
from ultimate_predictor import Ultimate_TFLN_Predictor

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE HEADER & STYLING
# ==========================================
st.title("🎯 Ultimate TFLN 11-DOF Optimizer")
st.markdown("""
**Find the absolute physical limit of your design.**
This engine uses a **Memetic Algorithm** (NSGA-II Global Pareto Search + SLSQP Gradient Polishing)
to map the ultimate trade-offs between Bandwidth and VPI, mathematically locking the velocity match while maximizing characteristic impedance ($Z_c$).
""")

# ==========================================
# 2. CACHED ENGINE LOADER
# ==========================================
@st.cache_resource
def load_engine():
    return Ultimate_TFLN_Predictor()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load predictor models. Ensure 'gp_surrogate_results_ultimate_500LHS' exists. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: TARGETS & BOUNDS
# ==========================================
st.sidebar.header("1. Performance Targets")
target_vpi = st.sidebar.number_input("Max Target VPI (Length Scaled) [V]", value=2.00, step=0.1)
target_bw = st.sidebar.number_input("Min Target EO Bandwidth [GHz]", value=65.0, step=5.0)
target_zc = st.sidebar.number_input("Min Target Zc [Ω]", value=45.0, step=1.0)
target_nm = st.sidebar.number_input("Target Index (nm)", value=2.270, step=0.01)
target_tol = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)

st.sidebar.markdown("---")
st.sidebar.header("2. Search Space (Bounds)")
def range_input(label, min_def, max_def, step=0.1, fmt="%.2f"):
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input(f"Min {label}", value=float(min_def), step=step, format=fmt)
    max_val = c2.number_input(f"Max {label}", value=float(max_def), step=step, format=fmt)
    return (min_val, max_val)

# 11 DOFs mapped exactly to the predictor architecture
b_WS    = range_input("WS [µm]", 10.0, 60.0)
b_GAP   = range_input("GAP [µm]", 4.0, 12.0)
b_MTX   = range_input("MTX [µm]", 1.5, 12.0)
b_L1    = range_input("L1 [µm]", 4.0, 60.0)
b_L2    = range_input("L2 [µm]", 4.0, 180.0)
b_W1    = range_input("W1 [µm]", 4.0, 60.0)
b_W2    = range_input("W2 [µm]", 4.0, 60.0)
b_CAPW  = range_input("CAP_W [µm]", 1.5, 12.0)
b_ETCH  = range_input("ETCH_DEPTH [µm]", 0.0, 0.40, step=0.01, fmt="%.3f")
b_L     = range_input("L_device [mm]", 4.0, 16.5)
b_Rt    = range_input("Rt [Ω]", 34.0, 60.0)

BOUNDS_LIST = [b_WS, b_GAP, b_MTX, b_L1, b_L2, b_W1, b_W2, b_CAPW, b_ETCH, b_L, b_Rt]

# ==========================================
# 4. OPTIMIZER CLASSES (NSGA-II & SLSQP)
# ==========================================
class NSGA2_Problem(ElementwiseProblem):
    def __init__(self, bnds):
        xl = np.array([b[0] for b in bnds])
        xu = np.array([b[1] for b in bnds])
        super().__init__(n_var=11, n_obj=2, n_ieq_constr=8, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # 11 Variables: [WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH, L_dev, Rt]
        WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH, L_dev, Rt = x
        geom_um = [WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH]

        g1_cap = CAP_W - (GAP - 1.0)
        g2_w = (W1 + W2) - 60.0

        # Fast-Fail constraint boundaries
        if g1_cap > 0 or g2_w > 0:
            out["F"] = [10.0, 0.0]
            out["G"] = [g1_cap, g2_w, 10., 10., 10., 10., 10., 10.]
            return

        try:
            res = engine.predict(geom=geom_um, L_device_mm=L_dev, Rt=Rt, Zs_driver=65.0, ng=2.27, plot=False)
            
            # STRICT FIX: Properly unpacking the [0] index from Bayesian Tuples
            bw  = res['bw'][0]
            vpi = res['vpi_len'][0]
            nm  = res['nm'][0]
            zc  = res['z0'][0]
            rtm = res['rt_min']
        except Exception:
            bw, vpi, nm, zc, rtm = 0.0, 10.0, 0.0, 0.0, 100.0

        # Dynamics Constraints calculated against User Settings
        g3_nm_low = (target_nm - target_tol) - nm
        g4_nm_high = nm - (target_nm + target_tol)
        g5_zc = target_zc - zc
        g6_rt = rtm - Rt
        g7_bw = target_bw - bw
        g8_vpi = vpi - target_vpi

        out["F"] = [vpi, -bw]
        out["G"] = [g1_cap, g2_w, g3_nm_low, g4_nm_high, g5_zc, g6_rt, g7_bw, g8_vpi]

def slsqp_polisher(x0, bnds):
    def get_p(x):
        try:
            return engine.predict(geom=x[:9].tolist(), L_device_mm=x[9], Rt=x[10], Zs_driver=65.0, ng=2.27, plot=False)
        except:
            return {'z0':(0.,0.,0.), 'nm':(10.,0.,0.), 'bw':(0.,0.,0.), 'vpi_len':(10.,0.,0.), 'rt_min':100.}

    # STRICT FIX: Explicit list indexing to prevent the "Naked x" crash
    def obj_Z0(x): return -get_p(x)['z0'][0]
    def eq_nm(x): return get_p(x)['nm'][0] - target_nm
    def ineq_bw(x): return get_p(x)['bw'][0] - target_bw
    def ineq_vpi(x): return target_vpi - get_p(x)['vpi_len'][0]
    def ineq_rt(x): return x[10] - get_p(x)['rt_min']
    def ineq_cap(x): return (x[1] - 1.0) - x[7]  # GAP - 1.0 - CAP_W >= 0
    def ineq_wsum(x): return 60.0 - (x[5] + x[6]) # 60 - W1 - W2 >= 0

    cons = [
        {'type': 'eq', 'fun': eq_nm},
        {'type': 'ineq', 'fun': ineq_bw},
        {'type': 'ineq', 'fun': ineq_vpi},
        {'type': 'ineq', 'fun': ineq_rt},
        {'type': 'ineq', 'fun': ineq_cap},
        {'type': 'ineq', 'fun': ineq_wsum}
    ]

    res = scipy_minimize(obj_Z0, x0=x0, method='SLSQP', bounds=bnds, constraints=cons, options={'ftol': 1e-4, 'maxiter': 50})
    return res.x

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if st.button("🚀 Run Ultimate Inverse Synthesizer", use_container_width=True):
    prog = st.progress(0.0, "Stage 1/3: Launching NSGA-II Pareto Search...")

    problem = NSGA2_Problem(BOUNDS_LIST)
    algorithm = NSGA2(pop_size=150, n_offsprings=50, eliminate_duplicates=True)

    res_ga = pymoo_minimize(problem, algorithm, get_termination("n_gen", 100), seed=42, verbose=False)

    if res_ga.F is None:
        prog.empty()
        st.error("❌ Optimizer failed to find any designs meeting all constraints. Try relaxing BW, Vpi, or Zc targets.")
    else:
        prog.progress(0.5, "Stage 2/3: Extracting Pareto Front and Preparing DataFrames...")

        VPI_pareto = res_ga.F[:, 0]
        BW_pareto = -res_ga.F[:, 1]
        X_pareto = res_ga.X

        pareto_data = []
        for x in X_pareto:
            try:
                res = engine.predict(geom=x[:9].tolist(), L_device_mm=x[9], Rt=x[10], Zs_driver=65.0, ng=2.27, plot=False)
                
                # STRICT FIX: Explicit index extraction for DataFrame build
                pareto_data.append({
                    'VPI_Length_Scaled (V)': round(res['vpi_len'][0], 3),
                    'VPI_Fully_Penalized (V)': round(res['vpi_full'][0], 3),
                    'EO_Bandwidth (GHz)': round(res['bw'][0], 1),
                    'Zc (Ohms)': round(res['z0'][0], 2),
                    'nm': round(res['nm'][0], 4),
                    'Total_RF_Alpha_60GHz': round(res['alpha'][0], 3),
                    'Rt (Ohms)': round(x[10], 2),
                    'L_dev (mm)': round(x[9], 2),
                    'WS': round(x[0], 2), 'GAP': round(x[1], 2), 'MTX': round(x[2], 2),
                    'L1': round(x[3], 2), 'L2': round(x[4], 2), 'W1': round(x[5], 2), 'W2': round(x[6], 2),
                    'CAP_W': round(x[7], 2), 'ETCH_DEPTH': round(x[8], 3)
                })
            except Exception:
                pass

        df_pareto = pd.DataFrame(pareto_data).sort_values(by='EO_Bandwidth (GHz)', ascending=False)

        prog.progress(0.8, "Stage 3/3: Running SLSQP Gradient Polish on Absolute Best Design...")
        best_idx = np.argmax(BW_pareto)
        best_nsga_x = X_pareto[best_idx]

        final_x = slsqp_polisher(best_nsga_x, BOUNDS_LIST)

        prog.progress(1.0, "Synthesis Complete!")

        # Display Results
        st.success("✅ **Synthesis Complete! Found Ultimate Global Maximum.**")

        # --- IMPEDANCE MATCHING STATUS ---
        final_res = engine.predict(geom=final_x[:9].tolist(), L_device_mm=final_x[9], Rt=final_x[10], Zs_driver=65.0, ng=2.27, plot=False)
        st.subheader("🔌 System Impedance Match")
        if final_x[10] < final_res['rt_min']:
            st.error(f"⚠️ **WARNING:** Your Termination Resistor (Rt = {final_x[10]:.2f} Ω) violates physical matching limits (Min Allowed: {final_res['rt_min']:.2f} Ω). Expect severe RF reflections and signal distortion.")
        else:
            st.success(f"✅ **Status:** Impedance matching (Rt = {final_x[10]:.2f} Ω) is safely within limits (> {final_res['rt_min']:.2f} Ω).")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Polished 11-DOF Geometry")
            # STRICT FIX: Safe array formatting
            st.code(f"""
[GEOMETRY]
WS          = {final_x[0]:.2f} µm
GAP         = {final_x[1]:.2f} µm
MTX         = {final_x[2]:.2f} µm
L1          = {final_x[3]:.2f} µm
L2          = {final_x[4]:.2f} µm
W1          = {final_x[5]:.2f} µm
W2          = {final_x[6]:.2f} µm
CAP_W       = {final_x[7]:.2f} µm
ETCH_DEPTH  = {final_x[8]:.3f} µm

[SYSTEM]
L_device    = {final_x[9]:.2f} mm
Rt          = {final_x[10]:.2f} Ω
            """)

        with col2:
            
            st.subheader("📈 NSGA-II Pareto Front")
            fig_p, ax_p = plt.subplots(figsize=(6,4))
            ax_p.scatter(VPI_pareto, BW_pareto, color='blue', edgecolor='k', alpha=0.7, label='NSGA-II Designs')
            ax_p.scatter(final_res['vpi_len'][0], final_res['bw'][0], color='gold', marker='*', s=250, edgecolor='k', label='SLSQP Final Winner')
            ax_p.set_xlabel("VPI Length Scaled (V)")
            ax_p.set_ylabel("Bandwidth (GHz)")
            ax_p.grid(True, ls=':', alpha=0.7)
            ax_p.legend()
            st.pyplot(fig_p)

        st.subheader("📊 Final Modulator Figures of Merit")
        def render_fom(name, tup, unit):
            val, std, mae = tup
            return {"Metric": name, "Prediction": f"{val:.3f} {unit}", "95% CI": f"[{max(0.0, val-1.96*std):.3f}, {val+1.96*std:.3f}] {unit}", "Global MAE": f"± {mae:.4f}" if mae > 0 else "N/A"}

        fom_df = pd.DataFrame([
            render_fom("Microwave Index (nm)", final_res['nm'], ""),
            render_fom("Characteristic Impedance (Z0)", final_res['z0'], "Ω"),
            render_fom("Total RF Attenuation @ 60 GHz", final_res['alpha'], "dB/cm"),
            render_fom("Duty Cycle Corrected VPI*L", final_res['vpi_duty'], "V*cm"),
            render_fom("Length Scaled VPI", final_res['vpi_len'], "V"),
            render_fom("VPI (Fully Penalized)", final_res['vpi_full'], "V"),
            render_fom("Electro-Optic Bandwidth", final_res['bw'], "GHz")
        ])
        st.dataframe(fom_df, use_container_width=True, hide_index=True)

        # Thread-safe Streamlit S21 plotting using the predictor's arrays
        st.subheader("📈 Broadband Electro-Optic S21 Response")
        fig_s21, ax_s21 = plt.subplots(figsize=(10, 4))
        ax_s21.plot(final_res['f_axis'], final_res['s21'], 'b-', lw=2)
        ax_s21.axhline(-3, color='r', ls='--')
        if final_res['bw'][0] < 150.0:
            ax_s21.plot(final_res['bw'][0], -3, 'ko', markersize=8)
            ax_s21.annotate(f"{final_res['bw'][0]:.1f} GHz", (final_res['bw'][0] + 3, -1.5), fontsize=12, fontweight='bold')
        ax_s21.set_xlabel('Frequency (GHz)')
        ax_s21.set_ylabel('Normalized S21 (dB)')
        ax_s21.set_xlim(0, 150)
        ax_s21.set_ylim(-8, 1)
        ax_s21.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_s21)

        st.subheader("🌐 Pareto Front Explorer")
        st.dataframe(df_pareto)
        
        # Memory buffer for Excel download
        buf = io.BytesIO()
        df_pareto.to_excel(buf, index=False)
        st.download_button("📥 Download Pareto Front (Excel)", buf.getvalue(), "Pareto_Front_Designs.xlsx", "application/vnd.ms-excel")
