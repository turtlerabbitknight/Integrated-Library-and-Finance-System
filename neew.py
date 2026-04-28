import pandas as pd
import kagglehub
import os
import glob
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

st.set_page_config(page_title="Integrated Library and Finance System", layout="wide")

@st.cache_data
def get_data():
    try:
        path = kagglehub.dataset_download("ziya07/library-transaction-dataset")
        file = (os.path.join(path, "*.csv"))[0]
        #file = glob.glob(os.path.join(path, "*.csv"))[0]
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()
        df['role'] = df['user_role'].str.upper().fillna('STUDENT')
        return df
    except:
        return None

df = get_data()


def draw_pair(labels, values, title, ylabel, color, p_type="standard"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    y = np.array(values, dtype=float)
    
    # Calculations
    ax1.bar(labels, y, color=color, alpha=0.8)
    ax1.set_title(f"CURRENT: {title}", fontweight='bold')
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(0, max(y) * 1.15)

    # Prediction
    if p_type == "maintenance":
        future_y = y * 1.85
        staff_limit = (df[df['role'] == 'STAFF'].shape[0] if df is not None else 10) * 12
        bars = ax2.bar(labels, future_y, color='#FFD700', edgecolor='#2F4F4F', linewidth=2)
        ax2.axhline(staff_limit, color='#FF4500', linestyle='--', linewidth=2.5, label="Staff Ceiling")
        for b in bars:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{int(b.get_height())}', 
                    ha='center', va='bottom', weight='bold', color='#2F4F4F')
        ax2.set_title(f"FUTURE: {title} Forecast", fontweight='bold')
        ax2.set_ylim(0, max(max(future_y), staff_limit) * 1.2)

    elif p_type == "academic":
        role_mults = {"STUDENT": 1.40, "FACULTY": 1.15, "STAFF": 1.05}
        future_y = [y[i] * role_mults.get(str(labels[i]).upper(), 1.0) for i in range(len(labels))]
        ax2.bar(labels, future_y, color='#AFEEEE', edgecolor='#5F9EA0', alpha=0.7)
        ax2.set_title(f"FUTURE: {title} Forecast", fontweight='bold')
        ax2.set_ylim(0, max(future_y) * 1.15)

    elif p_type == "traffic":
        future_y = np.roll(y, 2) * 0.8
        future_y[:2] = 12 
        ax2.plot(labels, future_y, "m-x", linewidth=2, markersize=8, label="Weekend Prediction")
        ax2.set_title(f"FUTURE: Weekend Traffic Prediction", fontweight='bold')
        ax2.set_ylim(0, max(y) * 1.15)

    else: 
        m, b = np.polyfit(np.arange(len(y)), y, 1)
        fut_labels = list(labels) + ["Next"]
        fut_y = m * np.arange(len(fut_labels)) + b
        ax2.plot(fut_labels, fut_y, "g--o", linewidth=2)
        ax2.set_title(f"FUTURE: {title} Scaling", fontweight='bold')
        ax2.set_ylim(0, max(max(y), max(fut_y)) * 1.15)

    plt.setp(ax1.get_xticklabels(), rotation=25, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=25, ha='right')
    ax2.legend()
    st.pyplot(fig)
    st.divider()

if df is not None:
    total_tx = len(df)
    ov_count = df['overdue_status'].sum() if 'overdue_status' in df.columns else 627
    
    # FINANCIAL BASELINES
    fee, fin, prt, gnt = total_tx * 10, ov_count * 5, total_tx * 2.5, 30000.0
    c = total_tx * 25.0
    P, L, L_inact = 27500.0, 15675.0, 800.0
    
    tab = st.radio("", ["Overview", "Revenue", "Staffing", "Assets"], horizontal=True, label_visibility="collapsed")
    mos = ["Jan", "Feb", "Mar", "Apr", "May"]

    if tab == "Overview":
        st.subheader("📊 Financial Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Membership Fees", f"${fee:,.2f}")
        c2.metric("Late Fines", f"${fin:,.2f}")
        c3.metric("Printing", f"${prt:,.2f}")
        c4.metric("Grants", f"${gnt:,.2f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Operational Cost", f"${c:,.2f}")
        c6.metric("Future Budget", f"${P:,.2f}")
        c7.metric("Overdue Risk", f"${L:,.2f}")
        c8.metric("Inactive Loss", f"${L_inact:,.2f}")

    elif tab == "Revenue":
        st.header("💳 Revenue & Scaling")
        draw_pair(mos, [fee*0.8, fee*0.9, fee, fee*1.1, fee*1.05], "Membership Revenue", "$", "navy")
        draw_pair(mos, [fin*0.8, fin, fin*1.3, fin*0.9, fin*1.2], "Late Fines", "$", "salmon")
        draw_pair(mos, [40, 55, 75, 60, 95], "New Members", "Qty", "gold")
        draw_pair(mos, [c*0.9, c, c*1.1, c*1.2, c*1.15], "Operational Burn Rate", "$", "grey")

    elif tab == "Staffing":
        st.header("👥 Capacity & Traffic")
        roles = ["STUDENT", "FACULTY", "STAFF"]
        r_vals = [df['role'].value_counts().get(r, 0) for r in roles]
        draw_pair(roles, r_vals, "Academic Load", "Users", "teal", p_type="academic")
        
        times = ["8 AM", "10 AM", "12 PM", "2 PM", "4 PM", "6 PM", "8 PM"]
        draw_pair(times, [15, 45, 130, 85, 120, 30, 10], "Traffic Flow", "Users", "purple", p_type="traffic")
        
        draw_pair(mos, [450, 500, 700, 480, 600], "Digital Assets", "Hits", "darkblue")
        draw_pair(mos, [85, 82, 88, 90, 92], "Satisfaction", "%", "skyblue")

    elif tab == "Assets":
        st.header("📦 Maintenance & Lifecycle")
        draw_pair(mos, [9, 20, 24, 35, 38], "Maintenance Logs", "Units", "orange", p_type="maintenance")
        draw_pair(mos, [10, 25, 40, 35, 60], "Asset Acquisition", "Units", "darkred")
        draw_pair(mos, [L*0.8, L*0.9, L, L*1.1, L*1.05], "Projected Risk", "$", "crimson")

