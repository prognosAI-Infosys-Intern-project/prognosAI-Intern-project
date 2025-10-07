import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="PrognosAI - RUL Prediction", layout="wide")

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_test_data(seq_file, meta_file):
    sequences = np.load(seq_file)
    metadata = pd.read_csv(meta_file)
    return sequences, metadata

def plot_rul_trends(df_rul):
    st.subheader("Select up to 5 Engines to Display RUL Trends")
    engines = sorted(df_rul['engine_id'].unique())
    # Allow no selection; if no selection show full dataset
    selected_engines = st.multiselect("Choose engines:", engines, default=engines[:5], max_selections=5)

    if selected_engines:
        filtered_df = df_rul[df_rul['engine_id'].isin(selected_engines)]
    else:
        # If none selected, display the full dataset
        filtered_df = df_rul

    fig = px.line(
        filtered_df,
        x='cycle',
        y='RUL',
        color='engine_id',
        labels={'cycle': 'Cycle', 'RUL': 'Predicted RUL', 'engine_id': 'Engine ID'},
        hover_name='engine_id'
    )
    fig.update_layout(legend_title_text='Engine ID', height=500)
    st.plotly_chart(fig, use_container_width=True)



def plot_alert_zone_counts(df_rul):
    # Prepare engine-level alert counts using last cycle predictions
    latest = df_rul.groupby('engine_id').tail(1).reset_index(drop=True)

    # Define alert zones
    bins = [-1, 10, 30, 1e5]
    labels = ['Critical', 'Warning', 'Safe']
    latest['Alert'] = pd.cut(latest['RUL'], bins=bins, labels=labels)

    counts = latest['Alert'].value_counts().reindex(labels).fillna(0).reset_index()
    counts.columns = ['Alert Zone', 'Number of Engines']

    fig = px.bar(
        counts,
        x='Alert Zone',
        y='Number of Engines',
        color='Alert Zone',
        title='Count of Engines per Alert Zone',
        text='Number of Engines',
        color_discrete_map={'Critical': 'red', 'Warning': 'orange', 'Safe': 'green'}
    )

    fig.update_traces(
        textposition='outside',
        textfont=dict(size=16, color='white'),
        marker_line_width=1.5,
        marker_line_color='black'
    )

    # Configure y-axis to have evenly spaced ticks
    max_count = counts['Number of Engines'].max()
    # Determine tick interval (e.g., even steps of 1, 2,...) based on max_count
    tick_step = max(1, int(np.ceil(max_count / 10)))  # or any preferred granularity
    y_ticks = list(range(0, int(max_count) + tick_step, tick_step))
    
    fig.update_layout(
        yaxis=dict(
            dtick=tick_step,
            tickmode='array',
            tickvals=y_ticks,
            title='Number of Engines'
        ),
        height=500,
        margin=dict(t=70, b=40, l=40, r=40),
        uniformtext_minsize=12,
        uniformtext_mode='show',
        xaxis_tickangle=0
    )

    st.plotly_chart(fig, use_container_width=True)



def render_dashboard(df_rul):
    st.subheader("Latest RUL Predictions per Engine")
    latest = df_rul.groupby('engine_id').tail(1).reset_index(drop=True)
    
    # Define alert levels based on RUL thresholds
    conditions = [
        (latest['RUL'] <= 10),               # Critical alert threshold
        (latest['RUL'] > 10) & (latest['RUL'] <= 30),  # Warning threshold
        (latest['RUL'] > 30)                 # Safe zone
    ]
    alerts = ['Critical', 'Warning', 'Safe']
    latest['Alert'] = pd.Categorical(pd.cut(latest['RUL'], bins=[-1, 10, 30, 1e5], labels=alerts))
    
    # Show alert count summary
    st.markdown("##### Maintenance Alert Summary")
    alert_counts = latest['Alert'].value_counts().reindex(alerts, fill_value=0)
    for level in alerts:
        count = alert_counts[level]
        if level == "Critical":
            st.error(f"❗ {count} engines in CRITICAL condition (RUL ≤ 10 cycles)")
        elif level == "Warning":
            st.warning(f"⚠️ {count} engines in WARNING zone (10 < RUL ≤ 30 cycles)")
        else:
            st.success(f"✅ {count} engines are SAFE (RUL > 30 cycles)")

    # Display detailed table with alerts
    st.dataframe(latest[['engine_id', 'cycle', 'RUL', 'Alert']], use_container_width=True)

def main():
    st.title("🛠️ PrognosAI Predictive Maintenance: RUL Prediction & Alert System")

    model_path = os.path.join('model', 'best_model.keras')
    scaler_path = os.path.join('processed_data', 'train', 'scaler.pkl')
    feature_cols_path = os.path.join('processed_data', 'train', 'feature_columns.txt')

    # Verify required files exist
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_cols_path)):
        st.error("Model, scaler, or feature columns file missing. Please train the model first.")
        return

    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # Load feature columns to confirm input shape
    with open(feature_cols_path, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    expected_feature_dim = len(feature_columns)

    uploaded_seq_file = st.file_uploader("Upload Test Sequences (.npy)", type=["npy"])
    uploaded_meta_file = st.file_uploader("Upload Test Metadata (.csv)", type=["csv"])

    if uploaded_seq_file and uploaded_meta_file:
        X_test_seq, df_meta = load_test_data(uploaded_seq_file, uploaded_meta_file)
        st.success(f"Loaded {X_test_seq.shape[0]} sequences for prediction")

        # Check feature dimension match
        if X_test_seq.shape[2] != expected_feature_dim:
            st.error(f"Feature dimension mismatch! Model expects {expected_feature_dim} features,"
                     f" but found {X_test_seq.shape[2]} in uploaded data. Please preprocess test data correctly.")
            return

        preds = model.predict(X_test_seq).flatten()
        df_meta = df_meta.copy()
        df_meta['RUL'] = preds

        st.subheader("RUL Trends Over Time")
        plot_rul_trends(df_meta)

        st.subheader("RUL Prediction Dashboard with Alert Zones and Maintenance Alerts")
        render_dashboard(df_meta)

        st.subheader("Alert Zone Distribution")
        plot_alert_zone_counts(df_meta)


if __name__ == "__main__":
    main()
