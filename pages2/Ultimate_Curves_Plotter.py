"""
================================================================================
📈 ULTIMATE TFLN 11-DOF PARAMETRIC SWEEP VISUALIZER
Interactive Plotly visualizer with Decoupled Alphas and Base Tracking.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
from pathlib import Path
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.title("📈 Ultimate TFLN Parametric Sweep Visualizer")
st.markdown("""
Instantly visualize the isolated physics of your modulator using the 11-DOF GP Surrogate + Physics Cascade.  
Explore how any geometric parameter (including Etch Depth) affects Bandwidth, VPI, or individual components of the RF Attenuation.
""")

# =====================================================================
# 1. THE ULTIMATE PREDICTOR ENGINE (Embedded for Streamlit Cache)
# =====================================================================
class Ultimate_TFLN_Predictor:
    def __init__(self, dir_vpi="gp_surrogate_results_ultimate_500LHS/GP_files_Vpi", 
                 dir_nm_z0="gp_surrogate_results_ultimate_500LHS/GP_files_nm_Z0", 
                 dir_alpha="gp_surrogate_results_ultimate_500LHS/GP_files_RF_alpha"):
        
        self.dirs = {'VPI': Path(dir_vpi), 'NM_Z0': Path(dir_nm_z0), 'ALPHA': Path(dir_alpha)}
        self.models, self.scalers = {}, {}
        self.c0 = 299792458.0
        self.L_cell = 200e-6
        self._load_system()

    def _load_system(self):
        with open(self.dirs['VPI'] / "scalers_COMSOL.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['VPI_X'], self.scalers['VPI_y'] = d['X'], d['y']['VPI_L [V*cm]']
        with open(self.dirs['VPI'] / "gp_model_VPI_L_V_cm.pkl", 'rb') as f: self.models['VPI'] = pickle.load(f)

        with open(self.dirs['NM_Z0'] / "scalers_COMSOL.pkl", 'rb') as f: self.scalers['NMZ0_C'] = pickle.load(f)
        with open(self.dirs['NM_Z0'] / "scalers_CST.pkl", 'rb') as f: self.scalers['NMZ0_CST'] = pickle.load(f)
        for m in [('RN_NM', 'gp_model_RN_NM.pkl'), ('Z0', 'gp_model_Z0_Ω.pkl'), ('dL', 'gp_model_Delta_L_lumped.pkl'), ('dC', 'gp_model_Delta_C_lumped.pkl')]:
            with open(self.dirs['NM_Z0'] / m[1], 'rb') as f: self.models[m[0]] = pickle.load(f)

        with open(self.dirs['ALPHA'] / "scalers_COMSOL.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['AL_C_X'], self.scalers['AL_C_y'] = d['X'], d['y']['RF ATT [dB/cm]']
        with open(self.dirs['ALPHA'] / "scalers_Net_Ohmic_Penalty.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['AL_O_X'], self.scalers['AL_O_y'] = d['scaler_X'], d['scaler_y']
        with open(self.dirs['ALPHA'] / "scalers_Pure_Radiation.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['AL_R_X'], self.scalers['AL_R_y'] = d['scaler_X'], d['scaler_y']
        for m in [('AL_C', 'gp_model_RF_ATT_dB_cm_COMSOL.pkl'), ('AL_O', 'gp_model_Net_Ohmic_Penalty.pkl'), ('AL_R', 'gp_model_Pure_Radiation.pkl')]:
            with open(self.dirs['ALPHA'] / m[1], 'rb') as f: self.models[m[0]] = pickle.load(f)

    def _predict_log10(self, model_key, scaler_X, scaler_y, X_input):
        X_scaled = scaler_X.transform(X_input)
        y_norm, y_std = self.models[model_key].predict(X_scaled, return_std=True)
        y_log = scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()[0]
        std_log = y_std[0] * scaler_y.scale_[0]
        val = (10 ** y_log) - 1e-15
        low = max(0.0, (10 ** (y_log - 1.96 * std_log)) - 1e-15)
        high = (10 ** (y_log + 1.96 * std_log)) - 1e-15
        return val, (high - low) / (2 * 1.96)

    def _predict_lin(self, model_key, scaler_X, scaler_y, X_input, limit_zero=False):
        X_scaled = scaler_X.transform(X_input)
        y_norm, y_std = self.models[model_key].predict(X_scaled, return_std=True)
        if isinstance(scaler_y, dict): pass
        val = scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()[0]
        std = y_std[0] * scaler_y.scale_[0]
        if limit_zero: val = max(0.0, val)
        return val, std

    def calc_eo_response(self, f_GHz, a_cond, a_rad, nm, zc, L_cm, ng, Rt, Zs):
        L_m = L_cm / 100.0
        f_ratio = np.maximum(f_GHz, 1e-9) / 60.0
        alpha_dB_cm = a_cond * np.sqrt(f_ratio) + a_rad * (f_ratio**3)
        alpha_np_m = alpha_dB_cm * (100.0 / 8.686)
        
        beta_rf = 2 * np.pi * f_GHz * 1e9 * nm / self.c0
        gamma_m = alpha_np_m + 1j * beta_rf
        gamma_o = 1j * 2 * np.pi * f_GHz * 1e9 * ng / self.c0

        Gamma_L = (Rt - zc) / (Rt + zc)
        Gamma_S = (Zs - zc) / (Zs + zc)
        denom = 1 - Gamma_S * Gamma_L * np.exp(-2 * gamma_m * L_m)
        delta1 = gamma_o - gamma_m
        delta2 = gamma_o + gamma_m

        int1 = np.where(np.abs(delta1) < 1e-12, L_m, (np.exp(delta1 * L_m) - 1) / delta1)
        int2 = np.where(np.abs(delta2) < 1e-12, L_m * np.exp(-2 * gamma_m * L_m), np.exp(-2 * gamma_m * L_m) * (np.exp(delta2 * L_m) - 1) / delta2)

        S21_eo = (int1 + Gamma_L * int2) / denom
        S21_mag = np.abs(S21_eo)
        idx_1GHz = np.argmin(np.abs(f_GHz - 1.0))
        return 20 * np.log10(S21_mag / S21_mag[idx_1GHz]), S21_mag

    def get_bandwidth(self, f_GHz, s21_db):
        for i in range(len(s21_db)):
            if s21_db[i] <= -3.0:
                if i > 0:
                    f1, f2 = f_GHz[i-1], f_GHz[i]
                    s1, s2 = s21_db[i-1], s21_db[i]
                    return f1 + (f2 - f1) * (-3.0 - s1) / (s2 - s1)
                return f_GHz[i]
        return f_GHz[-1]

    def predict(self, geom, L_device_mm, Rt, Zs_driver=65.0, ng=2.27):
        WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH = geom
        L_cm = L_device_mm / 10.0
        SLAB_H = 0.460 - ETCH_DEPTH
        Perimeter = L1 + L2 + W1 + W2
        Bragg_Proxy = Perimeter * (WS / GAP)

        x_c_5  = np.array([[WS, GAP, MTX, CAP_W, ETCH_DEPTH]])
        x_c_8  = np.array([[WS, GAP, MTX, L1, L2, W1, W2, ETCH_DEPTH]])
        x_c_9  = np.array([[WS, GAP, MTX, L1, L2, W1, W2, ETCH_DEPTH, Bragg_Proxy]])
        x_lc_9 = np.array([[WS, GAP, MTX, L1, L2, W1, W2, SLAB_H, Perimeter]])

        vpi_b, vpi_b_std = self._predict_log10('VPI', self.scalers['VPI_X'], self.scalers['VPI_y'], x_c_5)
        nm_2d, nm_2d_std = self._predict_lin('RN_NM', self.scalers['NMZ0_C']['X'], self.scalers['NMZ0_C']['y']['RN NM'], x_c_5)
        z0_2d, z0_2d_std = self._predict_lin('Z0', self.scalers['NMZ0_C']['X'], self.scalers['NMZ0_C']['y']['Z0 [Ω]'], x_c_5)
        dL, dL_std = self._predict_lin('dL', self.scalers['NMZ0_CST']['X'], self.scalers['NMZ0_CST']['y']['Delta_L_lumped'], x_lc_9)
        dC, dC_std = self._predict_lin('dC', self.scalers['NMZ0_CST']['X'], self.scalers['NMZ0_CST']['y']['Delta_C_lumped'], x_lc_9)
        dL, dL_std, dC, dC_std = dL/1e12, dL_std/1e12, dC/1e15, dC_std/1e15

        al_b, al_b_std = self._predict_log10('AL_C', self.scalers['AL_C_X'], self.scalers['AL_C_y'], x_c_5)
        al_o, al_o_std = self._predict_lin('AL_O', self.scalers['AL_O_X'], self.scalers['AL_O_y'], x_c_8)
        al_r, al_r_std = self._predict_lin('AL_R', self.scalers['AL_R_X'], self.scalers['AL_R_y'], x_c_9, limit_zero=True)

        L_dist_3d = max(1e-15, (z0_2d * nm_2d / self.c0) + (dL / self.L_cell))
        C_dist_3d = max(1e-15, (nm_2d / (z0_2d * self.c0)) + (dC / self.L_cell))
        nm_3d = self.c0 * math.sqrt(L_dist_3d * C_dist_3d)
        z0_3d = math.sqrt(L_dist_3d / C_dist_3d)
        
        al_cond = al_b + al_o
        al_total_60 = al_cond + al_r

        duty_cycle = L1 / 200.0
        vpi_duty = vpi_b / (1.0 - duty_cycle)
        vpi_length = vpi_duty / L_cm

        nm_3d_std = math.sqrt(nm_2d_std**2 + (dL_std/L_dist_3d * 0.5 * nm_3d)**2 + (dC_std/C_dist_3d * 0.5 * nm_3d)**2)
        z0_3d_std = math.sqrt(z0_2d_std**2 + (dL_std/L_dist_3d * 0.5 * z0_3d)**2 + (dC_std/C_dist_3d * 0.5 * z0_3d)**2)
        al_cond_std = math.sqrt(al_b_std**2 + al_o_std**2)
        al_total_std = math.sqrt(al_b_std**2 + al_o_std**2 + al_r_std**2)

        f_axis = np.linspace(0.001, 150.0, 500)
        idx_60 = np.argmin(np.abs(f_axis - 60.0))

        def compute_foms(n_val, z_val, a_c_val, a_r_val):
            s21_dB_ll, s21_mag_ll = self.calc_eo_response(f_axis, 1e-8, 1e-8, n_val, z_val, L_cm, ng, Rt, Zs_driver)
            s21_dB_full, s21_mag_full = self.calc_eo_response(f_axis, a_c_val, a_r_val, n_val, z_val, L_cm, ng, Rt, Zs_driver)
            bw = self.get_bandwidth(f_axis, s21_dB_full)
            vpi_full = vpi_length / (s21_mag_full[idx_60] / s21_mag_full[0])
            return bw, vpi_full

        bw_nom, vfull_nom = compute_foms(nm_3d, z0_3d, al_cond, al_r)

        d_nm = 0.01 * nm_3d; bw_nm, vfull_nm = compute_foms(nm_3d + d_nm, z0_3d, al_cond, al_r)
        d_z0 = 0.01 * z0_3d; bw_z0, vfull_z0 = compute_foms(nm_3d, z0_3d + d_z0, al_cond, al_r)
        d_ac = 0.01 * al_cond; bw_ac, vfull_ac = compute_foms(nm_3d, z0_3d, al_cond + d_ac, al_r)
        d_ar = max(0.01 * al_r, 1e-4); bw_ar, vfull_ar = compute_foms(nm_3d, z0_3d, al_cond, al_r + d_ar)

        def propagate(f_nom, f_n, f_z, f_c, f_r, uncert_n, uncert_z, uncert_c, uncert_r):
            return math.sqrt( (((f_n - f_nom)/d_nm)*uncert_n)**2 + (((f_z - f_nom)/d_z0)*uncert_z)**2 +
                              (((f_c - f_nom)/d_ac)*uncert_c)**2 + (((f_r - f_nom)/d_ar)*uncert_r)**2 )

        bw_std = propagate(bw_nom, bw_nm, bw_z0, bw_ac, bw_ar, nm_3d_std, z0_3d_std, al_cond_std, al_r_std)
        vfull_std = propagate(vfull_nom, vfull_nm, vfull_z0, vfull_ac, vfull_ar, nm_3d_std, z0_3d_std, al_cond_std, al_r_std)

        # Output payload explicitly provides decomposed Alphas for the sweep
        return {
            'nm': (nm_3d, nm_3d_std, 0.0), 'z0': (z0_3d, z0_3d_std, 0.0),
            'alpha_total': (al_total_60, al_total_std, 0.0),
            'alpha_base': (al_b, al_b_std, 0.0),
            'alpha_ohmic': (al_o, al_o_std, 0.0),
            'alpha_rad': (al_r, al_r_std, 0.0),
            'vpi_duty': (vpi_duty, vpi_b_std/(1-duty_cycle), 0.0), 
            'vpi_len': (vpi_length, (vpi_b_std/(1-duty_cycle))/L_cm, 0.0),
            'vpi_full': (vfull_nom, vfull_std, 0.0),
            'bw': (bw_nom, bw_std, 0.0)
        }

@st.cache_resource
def load_plotter_engine(): return Ultimate_TFLN_Predictor()

try:
    engine = load_plotter_engine()
except Exception as e:
    st.error(f"Failed to load predictor models. Error: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR: BASELINE INPUTS
# ==========================================
st.sidebar.header("Base Operating Point")
st.sidebar.caption("These values are held constant for all non-swept parameters.")

base_geom = {
    'WS': st.sidebar.number_input("WS [µm]", value=18.0, step=0.1),
    'GAP': st.sidebar.number_input("GAP [µm]", value=6.5, step=0.1),
    'MTX': st.sidebar.number_input("MTX [µm]", value=2.5, step=0.1),
    'L1': st.sidebar.number_input("L1 [µm]", value=24.0, step=0.1),
    'L2': st.sidebar.number_input("L2 [µm]", value=72.0, step=0.1),
    'W1': st.sidebar.number_input("W1 [µm]", value=8.0, step=0.1),
    'W2': st.sidebar.number_input("W2 [µm]", value=19.0, step=0.1),
    'CAP_W': st.sidebar.number_input("CAP_W [µm]", value=3.0, step=0.1),
    'ETCH_DEPTH': st.sidebar.number_input("ETCH_DEPTH [µm]", value=0.16, step=0.01, format="%.3f"),
    'L_dev': st.sidebar.number_input("L_device [mm]", value=16.5, step=0.5),
    'Rt': st.sidebar.number_input("Rt [Ω]", value=45.0, step=1.0)
}

# ==========================================
# 3. MAIN UI: SWEEP CONFIGURATION
# ==========================================
col1, col2 = st.columns(2)

with col1:
    fom_options = [
        'Electro-Optic Bandwidth [GHz]',
        'Total RF Attenuation [dB/cm]',
        'COMSOL RF Alpha Baseline [dB/cm]',
        'Net Ohmic Relief [dB/cm]',
        'Pure Radiative Alpha [dB/cm]',
        'Microwave Index (nm)',
        'Impedance (Zc) [Ω]',
        'Length Scaled VPI [V]',
        'Fully Penalized VPI [V]'
    ]
    target_fom = st.selectbox("1. Select Figure of Merit (Y-Axis)", options=fom_options)

with col2:
    if target_fom in ['Microwave Index (nm)', 'Impedance (Zc) [Ω]', 'COMSOL RF Alpha Baseline [dB/cm]', 'Net Ohmic Relief [dB/cm]', 'Pure Radiative Alpha [dB/cm]']:
        sweep_options = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2', 'ETCH_DEPTH']
    elif 'VPI' in target_fom or 'Attenuation' in target_fom:
        sweep_options = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2', 'ETCH_DEPTH', 'L_dev']
    else:
        sweep_options = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2', 'ETCH_DEPTH', 'L_dev', 'Rt']

    sweep_param = st.selectbox("2. Select Parameter to Sweep (X-Axis)", options=sweep_options)

# Dynamic limits based on selected parameter
default_bounds = {
    'WS': (10.0, 60.0), 'GAP': (4.0, 12.0), 'MTX': (1.5, 12.0), 'CAP_W': (1.5, 12.0),
    'L1': (4.0, 60.0), 'L2': (4.0, 180.0), 'W1': (4.0, 60.0), 'W2': (4.0, 60.0),
    'ETCH_DEPTH': (0.0, 0.40), 'L_dev': (4.0, 20.0), 'Rt': (34.0, 60.0)
}
def_min, def_max = default_bounds.get(sweep_param, (1.0, 100.0))

col3, col4 = st.columns(2)
with col3:
    min_val = st.number_input(f"Sweep Minimum ({sweep_param})", value=def_min, step=0.1)
with col4:
    max_val = st.number_input(f"Sweep Maximum ({sweep_param})", value=def_max, step=0.1)

unit_param = "mm" if sweep_param == 'L_dev' else "Ω" if sweep_param == 'Rt' else "µm"

# ==========================================
# 4. INSTANT LOOP EXECUTION
# ==========================================
def get_sweep_data(base_geom, param_to_sweep, bounds, fom):
    # FIXED: Proper extraction of bounds to prevent the linspace tuple dimension crash
    x_vals = np.linspace(bounds[0], bounds[1], 100)
    y_mean, y_std = [], []
    cap_w_adjusted = False
    
    for x in x_vals:
        cg = base_geom.copy()
        cg[param_to_sweep] = x
        
        # Enforce physical validity silently (CAP_W must fit inside GAP)
        if cg['CAP_W'] >= cg['GAP'] - 1.0:
            cg['CAP_W'] = max(1.0, cg['GAP'] - 1.0)
            cap_w_adjusted = True
            
        geom_list = [cg['WS'], cg['GAP'], cg['MTX'], cg['L1'], cg['L2'], cg['W1'], cg['W2'], cg['CAP_W'], cg['ETCH_DEPTH']]
        
        # Inference using the natively embedded Predictor
        res = engine.predict(geom_list, cg['L_dev'], cg['Rt'])
        
        if fom == 'Electro-Optic Bandwidth [GHz]':           out = res['bw']
        elif fom == 'Microwave Index (nm)':                  out = res['nm']
        elif fom == 'Impedance (Zc) [Ω]':                    out = res['z0']
        elif fom == 'Length Scaled VPI [V]':                 out = res['vpi_len']
        elif fom == 'Fully Penalized VPI [V]':               out = res['vpi_full']
        elif fom == 'Total RF Attenuation [dB/cm]':          out = res['alpha_total']
        elif fom == 'COMSOL RF Alpha Baseline [dB/cm]':      out = res['alpha_base']
        elif fom == 'Net Ohmic Relief [dB/cm]':              out = res['alpha_ohmic']
        elif fom == 'Pure Radiative Alpha [dB/cm]':          out = res['alpha_rad']
        else:                                                out = (0.0, 0.0, 0.0)
        
        y_mean.append(out[0])
        y_std.append(out[1])

    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    
    # Determine Physical Bounds
    if fom == 'Net Ohmic Relief [dB/cm]':
        # Ohmic Relief can mathematically and physically be negative
        y_low = y_mean - 1.96 * y_std
    else:
        # All other metrics are capped at a physical 0.0 minimum
        y_low = np.maximum(0.0, y_mean - 1.96 * y_std)
        
    y_up = y_mean + 1.96 * y_std
    
    return x_vals, y_mean, y_low, y_up, cap_w_adjusted

# ==========================================
# 5. INTERACTIVE PLOT GENERATION
# ==========================================
st.markdown("---")
if st.button("🚀 Generate Interactive Sweep", type="primary", use_container_width=True):
    with st.spinner(f"Querying GP Surrogate & Physics Cascade for {target_fom} vs {sweep_param}..."):
        
        x_vals, y_mean, y_low, y_up, cap_adjusted = get_sweep_data(base_geom, sweep_param, (min_val, max_val), target_fom)
        
        if x_vals is not None:
            fig = go.Figure()

            # Add 95% Confidence Interval (Shaded Region)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_vals, x_vals[::-1]]),
                y=np.concatenate([y_up, y_low[::-1]]),
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='95% Bayesian Confidence Interval'
            ))

            # Add Main Prediction Line
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_mean,
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                name=f'Predicted {target_fom}',
                hovertemplate=f"<b>{sweep_param}</b>: %{{x:.3f}} {unit_param}<br><b>{target_fom}</b>: %{{y:.3f}}<extra></extra>"
            ))
            
            # --- THE OTHER CHATBOT'S UX WIN: Baseline Tracking Marker ---
            base_x = base_geom[sweep_param]
            if min_val <= base_x <= max_val:
                idx_closest = np.argmin(np.abs(x_vals - base_x))
                fig.add_trace(go.Scatter(
                    x=[x_vals[idx_closest]], y=[y_mean[idx_closest]],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Current Base Setting'
                ))

            # Customize Layout
            fig.update_layout(
                title=f"<b>{target_fom} vs {sweep_param}</b>",
                xaxis_title=f"{sweep_param} [{unit_param}]",
                yaxis_title=target_fom,
                hovermode="x unified", 
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- PHYSICS NOTICES ---
            if cap_adjusted and sweep_param == 'GAP':
                st.warning("⚠️ **Physics Note:** At tight `GAP` dimensions, `CAP_W` was dynamically constrained to fit inside the gap to maintain geometric validity.")
                
            if target_fom == 'Net Ohmic Relief [dB/cm]':
                st.info("💡 **Physics Note:** Negative values for Net Ohmic Relief represent attenuation that was successfully *cancelled* by the massive surface area of the T-rail structure compared to a flat CPW baseline.")
