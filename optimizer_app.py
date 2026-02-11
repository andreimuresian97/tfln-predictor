import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.quasirandom import SobolEngine
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="TFLN Optimizer", layout="wide")
st.title("🚀 AI-Driven TFLN Modulator Optimizer")
st.markdown("""
**Find the perfect geometry.**
This tool uses a **Hybrid Bayesian Optimization** engine (Sobol Sampling + Gradient Polishing) 
to find the optimal TFLN cross-section for your specific constraints.
""")

# ==========================================
# 2. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("1. Geometry Constraints")
st.sidebar.markdown("Define the allowable search range.")

# Helper for sidebar ranges
def range_input(label, min_def, max_def, step=0.1, key_prefix=""):
    col1, col2 = st.sidebar.columns(2)
    min_val = col1.number_input(f"Min {label}", value=float(min_def), step=step, key=f"{key_prefix}_min_{label}")
    max_val = col2.number_input(f"Max {label}", value=float(max_def), step=step, key=f"{key_prefix}_max_{label}")
    return (min_val, max_val)

bounds = {}
bounds['WS']    = range_input("Signal Width (WS)", 35.0, 70.0)
bounds['GAP']   = range_input("Gap", 3.0, 15.0)
bounds['MTX']   = range_input("Metal Thickness (MTX)", 1.8, 15.0)
bounds['CAP_W'] = range_input("Cap Width", 1.4, 15.0)
bounds['L1']    = range_input("L1 (T-Rail)", 3.0, 60.0)
bounds['L2']    = range_input("L2 (T-Rail)", 3.0, 190.0)
bounds['W1']    = range_input("W1 (T-Rail)", 3.0, 60.0)
bounds['W2']    = range_input("W2 (T-Rail)", 3.0, 60.0)

st.sidebar.header("2. Performance Targets")
const_vpi = st.sidebar.number_input("Max Vpi·L (V·cm)", value=2.0, step=0.1)
const_s21 = st.sidebar.number_input("Min S21 (dB)", value=-2.0, step=0.1)
const_nm_target = st.sidebar.number_input("Target Index (nm)", value=2.27, step=0.01)
const_nm_tol = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)
beta = st.sidebar.slider("Optimization Aggressiveness (Beta)", 0.0, 2.0, 0.5, help="0.0 = Safe, 1.0 = Curious")

# Fixed list for ordering
VAR_NAMES = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2']
BOUNDS_LIST = [bounds[name] for name in VAR_NAMES]

# ==========================================
# 3. CORE FUNCTIONS (Memory Optimized)
# ==========================================

# UPDATED: Matches your repo folder name
MODEL_DIR = "gp_surrogate_results_199_8var_fixed"

@st.cache_resource
def load_scalers():
    """Load scalers only once."""
    scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler file not found at {scaler_path}")
        return None
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def load_single_model(name, scalers):
    """Load one model, use it, and return the model object."""
    # Map friendly name to file name
    file_map = {
        'VPI': 'gp_model_VPI.pkl',
        'nm':  'gp_model_nm.pkl',
        'Z0':  'gp_model_Z0.pkl',
        'S21': 'gp_model_S21.pkl'
    }
    
    # Map friendly name to Scaler key
    scaler_key_map = {
        'VPI': 'VPI (duty cycle)',
        'nm':  'nm',
        'Z0':  'Z0',
        'S21': 'S21'
    }
    
    path = os.path.join(MODEL_DIR, file_map[name])
    if not os.path.exists(path):
        st.error(f"❌ Model {name} not found at {path}")
        return None, None

    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    scaler_y = scalers['y'][scaler_key_map[name]]
    return model, scaler_y

def predict_batch_memory_safe(X_norm, scalers):
    """
    Loads models sequentially to save RAM.
    Returns a dictionary of predictions.
    """
    preds = {}
    model_names = ['VPI', 'nm', 'Z0', 'S21']
    
    # Progress bar for prediction phase
    prog_bar = st.progress(0, text="Predicting...")
    
    for i, name in enumerate(model_names):
        model, y_scaler = load_single_model(name, scalers)
        if model is None: return {}
        
        # Predict
        mean_norm, std_norm = model.predict(X_norm, return_std=True)
        
        # Inverse Transform
        mean_phys = y_scaler.inverse_transform(mean_norm.reshape(-1, 1)).ravel()
        if hasattr(y_scaler, 'scale_'):
            std_phys = std_norm * y_scaler.scale_[0]
        else:
            std_phys = std_norm
            
        if name == 'VPI':
            mean_phys = 10 ** mean_phys
            # Approximate uncertainty for 10^x
            # sigma_y = y * ln(10) * sigma_x
            std_phys = mean_phys * np.log(10) * std_phys
            
        preds[name] = {'mean': mean_phys, 'std': std_phys}
        
        # DELETE MODEL FROM MEMORY
        del model
        gc.collect()
        
        prog_bar.progress((i + 1) / 4, text=f"Predicted {name}")
    
    prog_bar.empty()
    return preds

# ==========================================
# 4. OPTIMIZATION LOGIC
# ==========================================

def run_optimization():
    scalers = load_scalers()
    if not scalers: return

    # 1. Generate Candidates
    N_CANDIDATES = 200000 
    
    with st.spinner(f"Generating {N_CANDIDATES} Monte Carlo samples..."):
        sobol = SobolEngine(dimension=8, scramble=True, seed=42)
        X_unit = sobol.draw(N_CANDIDATES).to(dtype=torch.float64)
        
        mins = torch.tensor([b[0] for b in BOUNDS_LIST], dtype=torch.float64)
        maxs = torch.tensor([b[1] for b in BOUNDS_LIST], dtype=torch.float64)
        X_phys = mins + X_unit * (maxs - mins)
        X_phys_np = X_phys.numpy()
        
        # 2. Geometric Filtering
        # CAP_W < GAP - 1
        # W1 + W2 < 65
        gap_col = 1; cap_col = 3; w1_col = 6; w2_col = 7
        mask_geom = (X_phys_np[:, cap_col] < (X_phys_np[:, gap_col] - 1.0)) & \
                    ((X_phys_np[:, w1_col] + X_phys_np[:, w2_col]) < 65.0)
        
        X_valid = X_phys_np[mask_geom]
    
    if len(X_valid) == 0:
        st.error("No candidates match geometric constraints!")
        return
    
    st.caption(f"Valid Geometries: {len(X_valid)} / {N_CANDIDATES}")

    # 3. Batch Predict
    scaler_X = scalers['X']['input']
    X_norm = scaler_X.transform(X_valid)
    
    preds = predict_batch_memory_safe(X_norm, scalers)
    if not preds: return
    
    # 4. Apply Performance Constraints
    mask_perf = (preds['VPI']['mean'] < const_vpi) & \
                (np.abs(preds['nm']['mean'] - const_nm_target) < const_nm_tol) & \
                (preds['S21']['mean'] > const_s21)
                
    valid_indices = np.where(mask_perf)[0]
    
    if len(valid_indices) == 0:
        st.warning(f"No designs met the constraints (Vpi<{const_vpi}, S21>{const_s21}, nm≈{const_nm_target}). Try relaxing them.")
        return

    # 5. Select Discrete Winner
    z0_mean = preds['Z0']['mean'][valid_indices]
    z0_std = preds['Z0']['std'][valid_indices]
    scores = z0_mean + (beta * z0_std)
    
    best_idx_local = np.argmax(scores)
    best_idx_global = valid_indices[best_idx_local]
    
    winner_x = X_valid[best_idx_global]
    
    # 6. Polishing (SLSQP)
    perform_polishing(winner_x, scalers)

def perform_polishing(x0, scalers):
    st.info("🛠 Polishing geometry for maximum Z0...")
    
    # Load all models for Scipy (Fits in RAM since we don't store big arrays)
    models = {}
    model_names = ['VPI', 'nm', 'Z0', 'S21']
    for name in model_names:
        m, s = load_single_model(name, scalers)
        models[name] = {'model': m, 'scaler': s}
        
    scaler_X = scalers['X']['input']

    def predict_one(x):
        x_n = scaler_X.transform(x.reshape(1, -1))
        res = {}
        for k, v in models.items():
            m_val = v['model'].predict(x_n)[0]
            val = v['scaler'].inverse_transform(m_val.reshape(-1,1)).item()
            if k == 'VPI': val = 10**val
            res[k] = val
        return res

    def obj(x): return -1.0 * predict_one(x)['Z0']
    
    # Constraints for Scipy
    cons = [
        {'type': 'ineq', 'fun': lambda x: const_vpi - predict_one(x)['VPI']},
        {'type': 'ineq', 'fun': lambda x: predict_one(x)['S21'] - const_s21},
        {'type': 'ineq', 'fun': lambda x: (const_nm_target + const_nm_tol) - predict_one(x)['nm']},
        {'type': 'ineq', 'fun': lambda x: predict_one(x)['nm'] - (const_nm_target - const_nm_tol)},
        {'type': 'ineq', 'fun': lambda x: (x[1]-1.0) - x[3]}, # GAP-1 > CAP
        {'type': 'ineq', 'fun': lambda x: 65.0 - (x[6] + x[7])} # W < 65
    ]
    
    # Run Optimization
    res = minimize(obj, x0, method='SLSQP', bounds=BOUNDS_LIST, constraints=cons)
    
    final_x = res.x
    final_preds = predict_one(final_x)
    
    # Save to Session State to display
    st.session_state['result'] = {
        'x': final_x,
        'preds': final_preds
    }
    
    # Cleanup
    del models
    gc.collect()

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_geometry(x):
    WS, GAP, MTX, CAP_W, L1, L2, W1, W2 = x
    
    # Color Scheme matching your style
    ELECTRODE_COLOR = '#F5BD02' 
    SUBSTRATE_COLOR = '#00FFFF' 
    CAP_COLOR = '#00FFFF'       
    LINE_COLOR = 'black'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- TOP VIEW (Simplified) ---
    ax1.set_title("Top View (Optimized)", fontsize=14)
    # Left Electrode (WS)
    ax1.add_patch(patches.Rectangle((-WS - GAP/2, -50), WS, 100, fc=ELECTRODE_COLOR, ec=LINE_COLOR))
    # Right Electrode (WG) - Just showing the gap edge
    ax1.add_patch(patches.Rectangle((GAP/2, -50), 30, 100, fc=ELECTRODE_COLOR, ec=LINE_COLOR))
    # Annotations
    ax1.text(0, 0, f"Gap\n{GAP:.2f}", ha='center', va='center')
    ax1.text(-WS/2 - GAP/2, 0, f"WS\n{WS:.2f}", ha='center', va='center')
    ax1.set_xlim(-WS - GAP - 10, GAP + 40)
    ax1.set_ylim(-60, 60)
    ax1.axis('off')

    # --- CROSS SECTION ---
    ax2.set_title("Cross Section", fontsize=14)
    # Substrate
    ax2.add_patch(patches.Rectangle((-50, -5), 100, 5, fc=SUBSTRATE_COLOR))
    # Black base layer
    ax2.add_patch(patches.Rectangle((-50, 0), 100, 0.23, fc='black'))
    
    # Electrodes
    ax2.add_patch(patches.Rectangle((-WS/2, 0.23), WS, MTX, fc=ELECTRODE_COLOR, ec=LINE_COLOR)) # WS
    ax2.add_patch(patches.Rectangle((WS/2 + GAP, 0.23), 20, MTX, fc=ELECTRODE_COLOR, ec=LINE_COLOR)) # Right WG
    ax2.add_patch(patches.Rectangle((-WS/2 - GAP - 20, 0.23), 20, MTX, fc=ELECTRODE_COLOR, ec=LINE_COLOR)) # Left WG
    
    # Caps
    ax2.add_patch(patches.Rectangle((WS/2 + GAP/2 - CAP_W/2, 0.23), CAP_W, 1.4, fc=CAP_COLOR, ec=LINE_COLOR))
    ax2.add_patch(patches.Rectangle((-WS/2 - GAP/2 - CAP_W/2, 0.23), CAP_W, 1.4, fc=CAP_COLOR, ec=LINE_COLOR))
    
    ax2.set_xlim(-WS - GAP - 10, WS + GAP + 10)
    ax2.set_ylim(-2, max(MTX, 2) + 5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    return fig

# ==========================================
# 6. MAIN UI EXECUTION
# ==========================================

if st.button("RUN OPTIMIZATION", type="primary"):
    run_optimization()

if 'result' in st.session_state:
    res = st.session_state['result']
    preds = res['preds']
    
    st.markdown("---")
    st.header("🏆 Optimal Geometry Found")
    
    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Z0 (Impedance)", f"{preds['Z0']:.2f} Ω", delta="Maximize")
    c2.metric("S21 (Loss)", f"{preds['S21']:.2f} dB", delta_color="normal")
    c3.metric("Vpi·L", f"{preds['VPI']:.2f} V·cm", delta_color="inverse")
    c4.metric("Index (nm)", f"{preds['nm']:.4f}", help="Target: 2.27")
    
    # Plot
    st.pyplot(plot_geometry(res['x']))
    
    # Parameter Table
    st.subheader("Optimal Parameters")
    df_params = pd.DataFrame([res['x']], columns=VAR_NAMES)
    st.table(df_params.style.format("{:.3f}"))
