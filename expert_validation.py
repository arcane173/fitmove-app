import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="FitMove - Expert Validation",
    page_icon="👨‍🏫",
    layout="wide"
)

st.title("👨‍🏫 FitMove - Expert Validation Tool")
st.markdown("Validasi sistem dengan penilaian expert (pelatih olahraga)")

# Inisialisasi session state
if 'validation_data' not in st.session_state:
    st.session_state.validation_data = []

# Sidebar untuk input data
with st.sidebar:
    st.header("📝 Input Data Validasi")
    
    exercise = st.selectbox("Olahraga", ["curl", "pushup", "squat"])
    system_score = st.slider("Skor Sistem (0-100)", 0, 100, 75)
    expert_score = st.slider("Skor Expert (0-100)", 0, 100, 75)
    expert_notes = st.text_area("Catatan Expert", placeholder="Contoh: Form bagus, tapi siku sedikit naik...")
    
    if st.button("➕ Tambah Data", type="primary"):
        st.session_state.validation_data.append({
            'timestamp': datetime.now().isoformat(),
            'exercise': exercise,
            'system_score': system_score,
            'expert_score': expert_score,
            'expert_notes': expert_notes,
            'difference': abs(system_score - expert_score),
            'agreement': 1 if abs(system_score - expert_score) <= 10 else 0
        })
        st.success(f"Data {exercise} ditambahkan!")
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Hapus Semua Data", type="secondary"):
        st.session_state.validation_data = []
        st.success("Semua data dihapus!")
        st.rerun()

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Data Validasi")
    
    if not st.session_state.validation_data:
        st.info("Belum ada data. Tambahkan data validasi dari sidebar ➡️")
    else:
        df = pd.DataFrame(st.session_state.validation_data)
        st.dataframe(df, use_container_width=True)
        
        # Statistik
        st.subheader("📈 Statistik")
        
        if len(df) >= 2:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(df['system_score'], df['expert_score'])
            
            # Mean Absolute Error
            mae = np.mean(np.abs(df['system_score'] - df['expert_score']))
            
            # Agreement percentage
            agreement_pct = df['agreement'].mean() * 100
            
            # Per exercise breakdown
            per_exercise = {}
            for ex in df['exercise'].unique():
                ex_df = df[df['exercise'] == ex]
                if len(ex_df) >= 2:
                    per_exercise[ex] = {
                        'n': len(ex_df),
                        'correlation': stats.pearsonr(ex_df['system_score'], ex_df['expert_score'])[0],
                        'mae': np.mean(np.abs(ex_df['system_score'] - ex_df['expert_score'])),
                        'agreement': ex_df['agreement'].mean() * 100
                    }
            
            # Display metrics
            metric_cols = st.columns(4)
            metric_cols[0].metric("Pearson Correlation", f"{pearson_r:.3f}")
            metric_cols[1].metric("p-value", f"{pearson_p:.4f}")
            metric_cols[2].metric("MAE", f"{mae:.1f} points")
            metric_cols[3].metric("Agreement", f"{agreement_pct:.1f}%")
            
            # Interpretation
            if pearson_r >= 0.8:
                st.success("✅ **Interpretasi:** Korelasi SANGAT KUAT - Sistem sangat selaras dengan expert judgement")
            elif pearson_r >= 0.6:
                st.success("👍 **Interpretasi:** Korelasi KUAT - Sistem memiliki korelasi baik dengan expert")
            elif pearson_r >= 0.4:
                st.warning("⚠️ **Interpretasi:** Korelasi SEDANG - Perlu improvement pada beberapa aspek")
            else:
                st.error("❌ **Interpretasi:** Korelasi LEMAH - Sistem perlu evaluasi ulang")
            
            # Per exercise table
            if per_exercise:
                st.subheader("🏋️ Per Exercise Breakdown")
                per_ex_df = pd.DataFrame(per_exercise).T
                per_ex_df.columns = ['Samples', 'Correlation', 'MAE', 'Agreement %']
                st.dataframe(per_ex_df, use_container_width=True)

with col2:
    st.subheader("📊 Visualisasi")
    
    if len(st.session_state.validation_data) >= 2:
        df = pd.DataFrame(st.session_state.validation_data)
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Color by exercise
        colors = {'curl': '#00d4ff', 'pushup': '#00ff88', 'squat': '#ff6b35'}
        for ex in df['exercise'].unique():
            ex_df = df[df['exercise'] == ex]
            ax.scatter(ex_df['system_score'], ex_df['expert_score'], 
                      label=ex.upper(), c=colors.get(ex, '#888'), s=80, alpha=0.7)
        
        # Perfect agreement line
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Agreement')
        
        # Regression line
        z = np.polyfit(df['system_score'], df['expert_score'], 1)
        p = np.poly1d(z)
        ax.plot(df['system_score'], p(df['system_score']), 'r-', alpha=0.8, 
               label=f'Regression (R={pearson_r:.3f})')
        
        ax.set_xlabel('System Score')
        ax.set_ylabel('Expert Score')
        ax.set_title('Sistem vs Expert Score Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        st.pyplot(fig)
        plt.close()
        
        # Error distribution
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df['error'] = df['system_score'] - df['expert_score']
        
        # Boxplot by exercise
        exercises = df['exercise'].unique()
        box_data = [df[df['exercise'] == ex]['error'].values for ex in exercises]
        
        bp = ax2.boxplot(box_data, labels=exercises, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#00d4ff', '#00ff88', '#ff6b35'][:len(exercises)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Exercise')
        ax2.set_ylabel('Error (System - Expert)')
        ax2.set_title('Error Distribution per Exercise')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        plt.close()

# Export section
st.divider()
st.subheader("💾 Export Data")

if st.session_state.validation_data:
    df_export = pd.DataFrame(st.session_state.validation_data)
    
    csv = df_export.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Data sebagai CSV",
        data=csv,
        file_name=f"expert_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    
    # Generate report
    if st.button("📄 Generate Laporan"):
        if len(df_export) >= 2:
            pearson_r, pearson_p = stats.pearsonr(df_export['system_score'], df_export['expert_score'])
            mae = np.mean(np.abs(df_export['system_score'] - df_export['expert_score']))
            rmse = np.sqrt(np.mean((df_export['system_score'] - df_export['expert_score']) ** 2))
            agreement = df_export['agreement'].mean() * 100
            
            report = f"""
            ============================================================
            👨‍🏫 EXPERT VALIDATION REPORT
            ============================================================
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Total Comparisons: {len(df_export)}
            
            📊 CORRELATION METRICS
            ------------------------------------------------------------
            Pearson Correlation:  {pearson_r:.3f}
            p-value:             {pearson_p:.4f}
            
            📈 ERROR METRICS
            ------------------------------------------------------------
            Mean Absolute Error (MAE):   {mae:.2f} points
            Root Mean Square Error (RMSE): {rmse:.2f} points
            Agreement (within 10 pts):   {agreement:.1f}%
            
            ============================================================
            """
            st.text(report)
        else:
            st.warning("Minimal 2 data diperlukan untuk generate laporan!")

else:
    st.info("Tambahkan data validasi dari sidebar untuk memulai...")

# Footer
st.divider()
st.caption("💡 **Tips:** Bandingkan skor sistem dengan penilaian expert (pelatih) untuk setiap repetisi. Semakin tinggi korelasi, semakin akurat sistem.")