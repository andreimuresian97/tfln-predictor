import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import gc
import base64
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

st.title("⚡ Ultimate TFLN Modulator Predictor")
st.markdown("""
Instant physics prediction mapping **Raw Electrostatics** to **Full Broadband RF Performance**.  
All predictions are performed at **1330 nm**, featuring exact decoupled 3D RF Attenuation and Bayesian Uncertainty.
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
        
        self.maes = {
            'VPI_L': 0.0866, 'RN_NM': 5.77651e-04, 'Z0': 2.51204e-02,
            'Delta_L': 2.59073e-13, 'Delta_C': 6.83945e-17,
            'COMSOL_RF_ATT': 3.49594e-03, 'Net_Ohmic': 0.0563, 'Pure_Radiation': 0.1612
        }

        self.c0 = 299792458.0
        self.L_cell = 200e-6
        self._load_system()

    def _load_system(self):
        # 1. LOAD VPI
        with open(self.dirs['VPI'] / "scalers_COMSOL.pkl", 'rb') as f:
            d = pickle.load(f)
            self.scalers['VPI_X'], self.scalers['VPI_y'] = d['X'], d['y']['VPI_L [V*cm]']
        with open(self.dirs['VPI'] / "gp_model_VPI_L_V_cm.pkl", 'rb') as f:
            self.models['VPI'] = pickle.load(f)

        # 2. LOAD NM & Z0
        with open(self.dirs['NM_Z0'] / "scalers_COMSOL.pkl", 'rb') as f:
            self.scalers['NMZ0_C'] = pickle.load(f)
        with open(self.dirs['NM_Z0'] / "scalers_CST.pkl", 'rb') as f:
            self.scalers['NMZ0_CST'] = pickle.load(f)
        
        for m in [('RN_NM', 'gp_model_RN_NM.pkl'), ('Z0', 'gp_model_Z0_Ω.pkl'), 
                  ('dL', 'gp_model_Delta_L_lumped.pkl'), ('dC', 'gp_model_Delta_C_lumped.pkl')]:
            with open(self.dirs['NM_Z0'] / m[1], 'rb') as f:
                self.models[m[0]] = pickle.load(f)

        # 3. LOAD ALPHA
        with open(self.dirs['ALPHA'] / "scalers_COMSOL.pkl", 'rb') as f:
            d = pickle.load(f)
            self.scalers['AL_C_X'], self.scalers['AL_C_y'] = d['X'], d['y']['RF ATT [dB/cm]']
        with open(self.dirs['ALPHA'] / "scalers_Net_Ohmic_Penalty.pkl", 'rb') as f:
            d = pickle.load(f)
            self.scalers['AL_O_X'], self.scalers['AL_O_y'] = d['scaler_X'], d['scaler_y']
        with open(self.dirs['ALPHA'] / "scalers_Pure_Radiation.pkl", 'rb') as f:
            d = pickle.load(f)
            self.scalers['AL_R_X'], self.scalers['AL_R_y'] = d['scaler_X'], d['scaler_y']

        for m in [('AL_C', 'gp_model_RF_ATT_dB_cm_COMSOL.pkl'), 
                  ('AL_O', 'gp_model_Net_Ohmic_Penalty.pkl'), 
                  ('AL_R', 'gp_model_Pure_Radiation.pkl')]:
            with open(self.dirs['ALPHA'] / m[1], 'rb') as f:
                self.models[m[0]] = pickle.load(f)

    def _predict_log10(self, model_key, scaler_X, scaler_y, X_input):
        X_scaled = scaler_X.transform(X_input)
        y_norm, y_std = self.models[model_key].predict(X_scaled, return_std=True)
        y_log = scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()[0]
        std_log = y_std[0] * scaler_y.scale_[0]

        val = (10 ** y_log) - 1e-15
        low = max(0.0, (10 ** (y_log - 1.96 * std_log)) - 1e-15)
        high = (10 ** (y_log + 1.96 * std_log)) - 1e-15
        std_linear = (high - low) / (2 * 1.96)
        return val, std_linear

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
        int2 = np.where(np.abs(delta2) < 1e-12, L_m * np.exp(-2 * gamma_m * L_m),
                        np.exp(-2 * gamma_m * L_m) * (np.exp(delta2 * L_m) - 1) / delta2)

        S21_eo = (int1 + Gamma_L * int2) / denom
        S21_mag = np.abs(S21_eo)
        idx_1GHz = np.argmin(np.abs(f_GHz - 1.0))
        S21_dB = 20 * np.log10(S21_mag / S21_mag[idx_1GHz])
        return S21_dB, S21_mag

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

        # ML Features
        x_c_5  = np.array([[WS, GAP, MTX, CAP_W, ETCH_DEPTH]])
        x_c_8  = np.array([[WS, GAP, MTX, L1, L2, W1, W2, ETCH_DEPTH]])
        x_c_9  = np.array([[WS, GAP, MTX, L1, L2, W1, W2, ETCH_DEPTH, Bragg_Proxy]])
        x_lc_9 = np.array([[WS, GAP, MTX, L1, L2, W1, W2, SLAB_H, Perimeter]])

        # 1. DIRECT ML PREDICTIONS
        vpi_b, vpi_b_std = self._predict_log10('VPI', self.scalers['VPI_X'], self.scalers['VPI_y'], x_c_5)
        nm_2d, nm_2d_std = self._predict_lin('RN_NM', self.scalers['NMZ0_C']['X'], self.scalers['NMZ0_C']['y']['RN NM'], x_c_5)
        z0_2d, z0_2d_std = self._predict_lin('Z0', self.scalers['NMZ0_C']['X'], self.scalers['NMZ0_C']['y']['Z0 [Ω]'], x_c_5)
        dL, dL_std = self._predict_lin('dL', self.scalers['NMZ0_CST']['X'], self.scalers['NMZ0_CST']['y']['Delta_L_lumped'], x_lc_9)
        dC, dC_std = self._predict_lin('dC', self.scalers['NMZ0_CST']['X'], self.scalers['NMZ0_CST']['y']['Delta_C_lumped'], x_lc_9)
        dL, dL_std, dC, dC_std = dL/1e12, dL_std/1e12, dC/1e15, dC_std/1e15

        al_b, al_b_std = self._predict_log10('AL_C', self.scalers['AL_C_X'], self.scalers['AL_C_y'], x_c_5)
        al_o, al_o_std = self._predict_lin('AL_O', self.scalers['AL_O_X'], self.scalers['AL_O_y'], x_c_8)
        al_r, al_r_std = self._predict_lin('AL_R', self.scalers['AL_R_X'], self.scalers['AL_R_y'], x_c_9, limit_zero=True)

        # 2. PHYSICS MERGE
        L_dist_3d = max(1e-15, (z0_2d * nm_2d / self.c0) + (dL / self.L_cell))
        C_dist_3d = max(1e-15, (nm_2d / (z0_2d * self.c0)) + (dC / self.L_cell))

        nm_3d = self.c0 * math.sqrt(L_dist_3d * C_dist_3d)
        z0_3d = math.sqrt(L_dist_3d / C_dist_3d)
        
        al_cond = al_b + al_o
        al_total_60 = al_cond + al_r

        duty_cycle = L1 / 200.0
        vpi_duty = vpi_b / (1.0 - duty_cycle)
        vpi_length = vpi_duty / L_cm

        # Primary Uncertainties (Quadrature)
        nm_3d_std = math.sqrt(nm_2d_std**2 + (dL_std/L_dist_3d * 0.5 * nm_3d)**2 + (dC_std/C_dist_3d * 0.5 * nm_3d)**2)
        z0_3d_std = math.sqrt(z0_2d_std**2 + (dL_std/L_dist_3d * 0.5 * z0_3d)**2 + (dC_std/C_dist_3d * 0.5 * z0_3d)**2)
        al_cond_std = math.sqrt(al_b_std**2 + al_o_std**2)
        al_total_std = math.sqrt(al_b_std**2 + al_o_std**2 + al_r_std**2)

        nm_3d_mae = math.sqrt(self.maes['RN_NM']**2 + (self.maes['Delta_L']/L_dist_3d * 0.5 * nm_3d)**2 + (self.maes['Delta_C']/C_dist_3d * 0.5 * nm_3d)**2)
        z0_3d_mae = math.sqrt(self.maes['Z0']**2 + (self.maes['Delta_L']/L_dist_3d * 0.5 * z0_3d)**2 + (self.maes['Delta_C']/C_dist_3d * 0.5 * z0_3d)**2)
        al_cond_mae = math.sqrt(self.maes['COMSOL_RF_ATT']**2 + self.maes['Net_Ohmic']**2)
        al_total_mae = math.sqrt(al_cond_mae**2 + self.maes['Pure_Radiation']**2)

        # 3. DERIVED FOMs (Jacobian)
        f_axis = np.linspace(0.001, 150.0, 500)
        idx_60 = np.argmin(np.abs(f_axis - 60.0))

        def compute_foms(n_val, z_val, a_c_val, a_r_val):
            s21_dB_ll, s21_mag_ll = self.calc_eo_response(f_axis, 1e-8, 1e-8, n_val, z_val, L_cm, ng, Rt, Zs_driver)
            s21_dB_full, s21_mag_full = self.calc_eo_response(f_axis, a_c_val, a_r_val, n_val, z_val, L_cm, ng, Rt, Zs_driver)
            
            bw = self.get_bandwidth(f_axis, s21_dB_full)
            vpi_ll = vpi_length / (s21_mag_ll[idx_60] / s21_mag_ll[0])
            vpi_full = vpi_length / (s21_mag_full[idx_60] / s21_mag_full[0])
            return bw, vpi_ll, vpi_full, s21_dB_full

        bw_nom, vll_nom, vfull_nom, s21_nom_curve = compute_foms(nm_3d, z0_3d, al_cond, al_r)

        d_nm = 0.01 * nm_3d; bw_nm, vll_nm, vfull_nm, _ = compute_foms(nm_3d + d_nm, z0_3d, al_cond, al_r)
        d_z0 = 0.01 * z0_3d; bw_z0, vll_z0, vfull_z0, _ = compute_foms(nm_3d, z0_3d + d_z0, al_cond, al_r)
        d_ac = 0.01 * al_cond; bw_ac, vll_ac, vfull_ac, _ = compute_foms(nm_3d, z0_3d, al_cond + d_ac, al_r)
        d_ar = max(0.01 * al_r, 1e-4); bw_ar, vll_ar, vfull_ar, _ = compute_foms(nm_3d, z0_3d, al_cond, al_r + d_ar)

        def propagate(f_nom, f_n, f_z, f_c, f_r, uncert_n, uncert_z, uncert_c, uncert_r):
            return math.sqrt( (((f_n - f_nom)/d_nm)*uncert_n)**2 +
                              (((f_z - f_nom)/d_z0)*uncert_z)**2 +
                              (((f_c - f_nom)/d_ac)*uncert_c)**2 +
                              (((f_r - f_nom)/d_ar)*uncert_r)**2 )

        bw_std = propagate(bw_nom, bw_nm, bw_z0, bw_ac, bw_ar, nm_3d_std, z0_3d_std, al_cond_std, al_r_std)
        bw_mae = propagate(bw_nom, bw_nm, bw_z0, bw_ac, bw_ar, nm_3d_mae, z0_3d_mae, al_cond_mae, self.maes['Pure_Radiation'])

        vll_std = propagate(vll_nom, vll_nm, vll_z0, vll_nom, vll_nom, nm_3d_std, z0_3d_std, 0, 0)
        vfull_std = propagate(vfull_nom, vfull_nm, vfull_z0, vfull_ac, vfull_ar, nm_3d_std, z0_3d_std, al_cond_std, al_r_std)

        # 4. EXPORT ARRAYS FOR STREAMLIT PLOTTING
        f_ratio = np.maximum(f_axis, 1e-9) / 60.0
        alpha_curve_nom = al_cond * np.sqrt(f_ratio) + al_r * (f_ratio**3)
        
        ac_bc = max(0, al_cond - 1.96*al_cond_std)
        ar_bc = max(0, al_r - 1.96*al_r_std)
        alpha_curve_bc = ac_bc * np.sqrt(f_ratio) + ar_bc * (f_ratio**3)
        
        ac_wc = al_cond + 1.96*al_cond_std
        ar_wc = al_r + 1.96*al_r_std
        alpha_curve_wc = ac_wc * np.sqrt(f_ratio) + ar_wc * (f_ratio**3)

        gamma_limit = 10 ** (-10.0 / 20.0) 
        rt_min_zs = Zs_driver * (1 - gamma_limit) / (1 + gamma_limit) 
        rt_min_zc = z0_3d * (1 - gamma_limit) / (1 + gamma_limit) 

        return {
            'nm': (nm_3d, nm_3d_std, nm_3d_mae),
            'z0': (z0_3d, z0_3d_std, z0_3d_mae),
            'alpha': (al_total_60, al_total_std, al_total_mae),
            'vpi_base': (vpi_b, vpi_b_std, self.maes['VPI_L']),
            'vpi_duty': (vpi_duty, vpi_b_std/(1-duty_cycle), self.maes['VPI_L']/(1-duty_cycle)),
            'vpi_len': (vpi_length, (vpi_b_std/(1-duty_cycle))/L_cm, (self.maes['VPI_L']/(1-duty_cycle))/L_cm),
            'vpi_ll': (vll_nom, vll_std, 0.0), 
            'vpi_full': (vfull_nom, vfull_std, 0.0), 
            'bw': (bw_nom, bw_std, bw_mae),
            'rt_min': max(rt_min_zs, rt_min_zc),
            # Plot arrays
            'f_axis': f_axis,
            's21': s21_nom_curve,
            'alpha_curve_nom': alpha_curve_nom,
            'alpha_curve_bc': alpha_curve_bc,
            'alpha_curve_wc': alpha_curve_wc
        }

@st.cache_resource
def load_engine():
    return Ultimate_TFLN_Predictor()

try:
    predictor = load_engine()
except Exception as e:
    st.error(f"Error loading models. Please verify the `gp_surrogate_results_ultimate_500LHS` directory exists. Error: {e}")
    st.stop()


# =====================================================================
# 2. SIDEBAR: GEOMETRY & SYSTEM INPUTS
# =====================================================================
st.sidebar.header("Geometry Parameters")

def user_input_features():
    st.sidebar.subheader("Global Device Parameters")
    length_cm = st.sidebar.number_input("Device Length (L) [cm]", value=1.65, min_value=0.4, max_value=10.0, step=0.1, format="%.2f")
    
    st.sidebar.subheader("Active Region")
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=10.69, step=0.01, format="%.2f")
    gap = st.sidebar.number_input("GAP [µm]", value=7.36, step=0.01, format="%.2f")
    mtx = st.sidebar.number_input("MTX (Metal Thickness) [µm]", value=1.50, step=0.01, format="%.2f")
    cap_w = st.sidebar.number_input("CAP_W (Cap Width) [µm]", value=6.25, step=0.01, format="%.2f")
    etch_depth = st.sidebar.number_input("ETCH_DEPTH [µm]", value=0.23, step=0.01, format="%.2f")
    
    st.sidebar.subheader("T-Structure Dimensions")
    l1 = st.sidebar.number_input("L1 (Inner Length) [µm]", value=4.0, step=0.1, format="%.1f")
    l2 = st.sidebar.number_input("L2 (Outer Length) [µm]", value=57.38, step=0.1, format="%.1f")
    w1 = st.sidebar.number_input("W1 (Inner Width) [µm]", value=10.69, step=0.1, format="%.1f")
    w2 = st.sidebar.number_input("W2 (Outer Width) [µm]", value=35.92, step=0.1, format="%.1f")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Electrical Environment")
    Zs_driver = st.sidebar.number_input("Zs_driver (Telecom Driver) [Ω]", value=65.0, step=1.0)
    Rt = st.sidebar.number_input("Rt (Termination Resistor) [Ω]", value=37.16, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Fixed Parameters")
    st.sidebar.info("**Pitch:** 200 µm  \n**WG:** 70 µm  \n**CAP_H:** 1.4 µm  \n**ng:** 2.27")
    
    # 9-DOF EXACT Ordering mapped to the predictor
    geom_list = [ws, gap, mtx, l1, l2, w1, w2, cap_w, etch_depth]
    params = {"WS": ws, "GAP": gap, "MTX": mtx, "L1": l1, "L2": l2, "W1": w1, "W2": w2, "CAP_W": cap_w, "ETCH_DEPTH": etch_depth}
    
    return length_cm, geom_list, params, Rt, Zs_driver

length_cm, geometry_list, params, Rt, Zs_driver = user_input_features()

# =====================================================================
# 3. SVG DRAWING FUNCTIONS
# =====================================================================
def render_svg(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W, ETCH_DEPTH = p["WS"], p["GAP"], p["MTX"], p["CAP_W"], p["ETCH_DEPTH"]
    WG = 70.0; BOTTOM_LAYER_H = 0.46 - ETCH_DEPTH; RIDGE_W = 0.8; RIDGE_H = ETCH_DEPTH; CAP_HEIGHT = 1.4
    C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'
    
    CV_W = 800; CV_H = 600; CX = CV_W / 2
    
    def svg_arrow(x1, y1, x2, y2, text, text_loc="top", offset=10, font_size=14):
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1.5" marker-start="url(#arrow_start)" marker-end="url(#arrow_end)" />'
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        tx, ty = mx, my
        anchor = "middle"; dominant = "middle"
        if text_loc == "top": ty -= offset; dominant = "auto"
        elif text_loc == "bottom": ty += offset; dominant = "hanging"
        elif text_loc == "left": tx -= offset; anchor = "end"
        elif text_loc == "right": tx += offset; anchor = "start"
        return line + f'<text x="{tx}" y="{ty}" fill="{C_LINE}" font-family="sans-serif" font-size="{font_size}" font-weight="bold" text-anchor="{anchor}" dominant-baseline="{dominant}">{text}</text>'

    # TOP-DOWN
    SCALE_TOP = 3.5; CV_H_TOP = 700; CY_TOP = 380 
    def to_top(x, y): return CX + x*SCALE_TOP, CY_TOP - y*SCALE_TOP
    pts = [(GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
           (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
           (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
           (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)]
    poly_str = " ".join([f"{to_top(x,y)[0]},{to_top(x,y)[1]}" for x, y in pts])
    ws_x1, ws_y1 = to_top(-(GAP/2 + WS), 100)
    
    arrows_top = ""
    arrows_top += svg_arrow(*to_top(-(GAP/2 + WS), 120), *to_top(-GAP/2, 120), "WS", "top", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, 120), *to_top(GAP/2 + WG, 120), "WG", "top", 10)
    arrows_top += svg_arrow(*to_top(-GAP/2, -120), *to_top(GAP/2, -120), "GAP", "bottom", 10)
    l1_x = GAP/2 - 20; arrows_top += svg_arrow(*to_top(l1_x, -L1/2), *to_top(l1_x, L1/2), "L1", "left", 10)
    l2_x = GAP/2 + W1 + W2 + 20; arrows_top += svg_arrow(*to_top(l2_x, -L2/2), *to_top(l2_x, L2/2), "L2", "right", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, -L1/2 - 20), *to_top(GAP/2 + W1, -L1/2 - 20), "W1", "bottom", 10)
    arrows_top += svg_arrow(*to_top(GAP/2 + W1, L2/2 + 20), *to_top(GAP/2 + W1 + W2, L2/2 + 20), "W2", "top", 10)

    svg_top = f"""<svg width="{CV_W}" height="{CV_H_TOP}" viewBox="0 0 {CV_W} {CV_H_TOP}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="20" y="50" text-anchor="start" font-family="sans-serif" font-size="24" font-weight="bold">Top-Down View</text>
        <rect x="{ws_x1}" y="{ws_y1}" width="{WS*SCALE_TOP}" height="{200*SCALE_TOP}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />
        <polygon points="{poly_str}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />
        {arrows_top}
    </svg>"""

    # CROSS-SECTION
    SCALE_CS = 700.0 / ((WS/2 + GAP) + 1.0) / 2; CY_CS = 350 
    def to_cs(x, y): return CX + x*SCALE_CS, CY_CS - y*SCALE_CS
    
    sub_rect = f'<rect x="0" y="{to_cs(0, 0)[1]}" width="{CV_W}" height="{CV_H}" fill="{C_SUB}" />'
    bl_rect = f'<rect x="0" y="{to_cs(0, BOTTOM_LAYER_H)[1]}" width="{CV_W}" height="{BOTTOM_LAYER_H*SCALE_CS}" fill="black" />'
    ws_rect = f'<rect x="{to_cs(-WS/2, BOTTOM_LAYER_H + MTX)[0]}" y="{to_cs(-WS/2, BOTTOM_LAYER_H + MTX)[1]}" width="{WS*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    rwg_rect = f'<rect x="{to_cs(WS/2 + GAP, BOTTOM_LAYER_H + MTX)[0]}" y="{to_cs(WS/2 + GAP, BOTTOM_LAYER_H + MTX)[1]}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    lwg_rect = f'<rect x="{to_cs(-(WS/2 + GAP + 500), BOTTOM_LAYER_H + MTX)[0]}" y="{to_cs(-(WS/2 + GAP + 500), BOTTOM_LAYER_H + MTX)[1]}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    
    caps_svg = ""
    for cx in [WS/2 + GAP/2, -WS/2 - GAP/2]:
        caps_svg += f'<rect x="{to_cs(cx - CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT)[0]}" y="{to_cs(cx - CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT)[1]}" width="{CAP_W*SCALE_CS}" height="{CAP_HEIGHT*SCALE_CS}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5" />'
        caps_svg += f'<rect x="{to_cs(cx - RIDGE_W/2, BOTTOM_LAYER_H + RIDGE_H)[0]}" y="{to_cs(cx - RIDGE_W/2, BOTTOM_LAYER_H + RIDGE_H)[1]}" width="{RIDGE_W*SCALE_CS}" height="{RIDGE_H*SCALE_CS}" fill="black" />'

    arr_y = BOTTOM_LAYER_H + max(MTX, CAP_HEIGHT) + (3.0 if MTX < 5 else 0.5 * MTX)
    arrows_cs = ""
    arrows_cs += svg_arrow(*to_cs(-WS/2, arr_y), *to_cs(WS/2, arr_y), "WS", "top", 15)
    arrows_cs += svg_arrow(*to_cs(WS/2, arr_y), *to_cs(WS/2 + GAP, arr_y), "GAP", "top", 15)
    arrows_cs += svg_arrow(*to_cs((-WS/2 - GAP/2) - CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT + 1.0), *to_cs((-WS/2 - GAP/2) + CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT + 1.0), "CAP_W", "top", 15)
    arrows_cs += svg_arrow(*to_cs(-WS/2 + 2.0, BOTTOM_LAYER_H), *to_cs(-WS/2 + 2.0, BOTTOM_LAYER_H + MTX), "MTX", "right", 10)

    svg_cross = f"""<svg width="{CV_W}" height="{CV_H}" viewBox="0 0 {CV_W} {CV_H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="20" y="50" text-anchor="start" font-family="sans-serif" font-size="24" font-weight="bold">Cross-Section View</text>
        {sub_rect} {bl_rect} {ws_rect} {rwg_rect} {lwg_rect} {caps_svg} {arrows_cs}
    </svg>"""
    return svg_top, svg_cross

st.subheader("1. Geometry Visualization")
st.caption("Updated automatically based on sidebar inputs.")
col1, col2 = st.columns([1, 1])
svg_t, svg_c = generate_exact_svg(params)
with col1: st.markdown(render_svg(svg_t), unsafe_allow_html=True)
with col2: st.markdown(render_svg(svg_c), unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 4. RESULTS DISPLAY
# =====================================================================
st.subheader("2. Performance Prediction")

if st.button("🚀 Predict Performance", type="primary", use_container_width=True):
    
    with st.spinner("Extracting 3D Scattering and Decoupling RF Mechanics..."):
        res = predictor.predict(geometry_list, length_cm * 10.0, Rt, Zs_driver=Zs_driver)
    
    if res:
        # Impedance Warning Block
        st.markdown("### 🔍 Impedance Matching Conditions (S11 ≤ -10 dB)")
        rt_min = res['rt_min']
        st.write(f"Minimum safe **Rt** (accounting for Driver Zs={Zs_driver}Ω and Line Zc={res['z0'][0]:.1f}Ω) : **{rt_min:.2f} Ω**")
        
        if Rt < rt_min:
            st.error(f"⚠️ **WARNING:** Your specified Rt ({Rt} Ω) violates physical reflection limits. Expect heavy signal distortion.")
        else:
            st.success(f"✅ **Status:** Rt ({Rt} Ω) is safely within reflection limits.")
        
        st.markdown("---")
        
        # Top Row Metrics 
        col1, col2, col3 = st.columns(3)
        col1.metric("EO Bandwidth", f"{res['bw'][0]:.1f} GHz")
        col2.metric("Impedance (Zc)", f"{res['z0'][0]:.1f} Ω")
        col3.metric("Index (nm)", f"{res['nm'][0]:.4f}")
        
        # Comprehensive FOM Table
        st.markdown("### 📊 Detailed Figures of Merit")
        
        def fmt_row(name, tup, unit):
            val, std, mae = tup
            ci_l = max(0.0, val - 1.96*std) # Physical boundary cap
            ci_u = val + 1.96*std
            return {
                "Figure of Merit": name,
                "Prediction": f"{val:.3f} {unit}",
                "95% Confidence Interval": f"[{ci_l:.3f}, {ci_u:.3f}] {unit}",
                "Global MAE": f"± {mae:.4f}" if mae > 0 else "N/A (Derived)"
            }

        table_data = [
            fmt_row("1. Microwave Index (nm)", res['nm'], ""),
            fmt_row("2. Characteristic Impedance (Z0)", res['z0'], "Ω"),
            fmt_row("3. Total RF Attenuation @ 60 GHz", res['alpha'], "dB/cm"),
            fmt_row("4. Baseline VPI*L (COMSOL)", res['vpi_base'], "V*cm"),
            fmt_row("5. Duty Cycle Corrected VPI*L", res['vpi_duty'], "V*cm"),
            fmt_row("6. Pure Length Scaled VPI", res['vpi_len'], "V"),
            fmt_row("7. VPI (Walk-off + Term. Mismatch)", res['vpi_ll'], "V"),
            fmt_row("8. VPI (Fully Penalized)", res['vpi_full'], "V"),
            fmt_row("9. Electro-Optic Bandwidth (-3 dBe)", res['bw'], "GHz")
        ]

        df_results = pd.DataFrame(table_data)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # --- PLOT 1: S21 Bandwidth ---
        st.markdown("### 📈 Broadband RF Response")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res['f_axis'], res['s21'], 'b-', lw=2.5, label=f'EO Response (L={length_cm*10:.1f} mm, Rt={Rt} Ω)')
        ax.axhline(-3, color='r', linestyle='--', lw=2)
        
        if res['bw'][0] < 150.0:
            ax.plot(res['bw'][0], -3, 'ko', markersize=8)
            ax.annotate(f"{res['bw'][0]:.1f} GHz", (res['bw'][0] + 3, -1.5), fontsize=12, fontweight='bold')
            
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Normalized S21 (dB)')
        ax.set_ylim(-8, 1)
        ax.set_xlim(0, 150)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # --- PLOT 2: RF Attenuation Curve ---
        st.markdown("### 📉 RF Attenuation Profile (Decoupled Physics)")
        fig_alpha, ax_alpha = plt.subplots(figsize=(8, 4))
        
        # Plot the nominal fit
        ax_alpha.plot(res['f_axis'], res['alpha_curve_nom'], 'g-', lw=2.5, label='Predicted Total Attenuation')
        
        # Fill the 95% Confidence Interval between the best-case (lowest loss) and worst-case (highest loss)
        ax_alpha.fill_between(res['f_axis'], res['alpha_curve_bc'], res['alpha_curve_wc'], color='green', alpha=0.2, label='95% Bayesian Confidence Interval')

        ax_alpha.set_xlabel('Frequency (GHz)')
        ax_alpha.set_ylabel(r'Attenuation $\alpha$ (dB/cm)')
        ax_alpha.set_xlim(0, 150)
        ax_alpha.set_ylim(bottom=0)
        ax_alpha.grid(True, linestyle=':', alpha=0.7)
        ax_alpha.legend()
        plt.tight_layout()
        st.pyplot(fig_alpha)
