import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Protocol 4 — Bland–Altman", layout="wide")
st.title("Πρωτόκολλο 4: Concurrent Validity & Bland–Altman")
st.caption("Σύγκριση απλού ζυγού με ζυγό ακριβείας")

SCENARIOS = {
    "Σενάριο Α — Υψηλή Συμφωνία": pd.DataFrame({
        "ID": ["S1", "S2", "S3", "S4", "S5"],
        "Weight_Precision": [75.2, 68.0, 82.5, 60.1, 90.0],
        "Weight_Simple": [75.4, 67.9, 82.3, 60.2, 90.1],
    }),
    "Σενάριο Β — Συστηματικό Σφάλμα": pd.DataFrame({
        "ID": ["S1", "S2", "S3", "S4", "S5"],
        "Weight_Precision": [75.2, 68.0, 82.5, 60.1, 90.0],
        "Weight_Simple": [78.1, 71.0, 85.5, 63.2, 92.9],
    }),
}

REQUIRED = ["ID", "Weight_Precision", "Weight_Simple"]


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate input and calculate the two derived columns."""
    missing = [col for col in REQUIRED if col not in data.columns]
    if missing:
        raise ValueError("Λείπουν στήλες: " + ", ".join(missing))
    df = data[REQUIRED].copy()
    df["Weight_Precision"] = pd.to_numeric(df["Weight_Precision"], errors="coerce")
    df["Weight_Simple"] = pd.to_numeric(df["Weight_Simple"], errors="coerce")
    df = df.dropna(subset=["Weight_Precision", "Weight_Simple"])
    if len(df) < 2:
        raise ValueError("Χρειάζονται τουλάχιστον δύο πλήρη ζεύγη μετρήσεων.")
    df["Diff_d"] = df["Weight_Simple"] - df["Weight_Precision"]
    df["Mean_m"] = (df["Weight_Simple"] + df["Weight_Precision"]) / 2
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    precision = df["Weight_Precision"].astype(float)
    simple = df["Weight_Simple"].astype(float)
    differences = df["Diff_d"].astype(float)
    if precision.nunique() < 2 or simple.nunique() < 2:
        r = np.nan
        r2 = np.nan
        see = np.nan
    else:
        r, _ = stats.pearsonr(precision, simple)
        r2 = r ** 2
        see = simple.std(ddof=1) * np.sqrt(max(0.0, 1.0 - r2))
    bias = float(differences.mean())
    sd_d = float(differences.std(ddof=1))
    return {
        "Pearson (r)": float(r),
        "R²": float(r2),
        "SEE": float(see),
        "Mean Bias": bias,
        "SD_d": sd_d,
        "Upper LoA": float(bias + 1.96 * sd_d),
        "Lower LoA": float(bias - 1.96 * sd_d),
    }


with st.sidebar:
    st.header("Επιλογή σεναρίου")
    scenario = st.radio("Σενάριο", list(SCENARIOS.keys()))
    st.markdown("---")
    st.write("Προαιρετικά, ανέβασε CSV με στήλες:")
    st.code("ID, Weight_Precision, Weight_Simple")
    csv_file = st.file_uploader("CSV δεδομένων", type=["csv"])

raw = SCENARIOS[scenario].copy()
if csv_file is not None:
    try:
        raw = pd.read_csv(csv_file)
    except Exception as exc:
        st.sidebar.error(f"Αδυναμία ανάγνωσης CSV: {exc}")

try:
    df = prepare_data(raw)
    metrics = calculate_metrics(df)
except Exception as exc:
    st.error(f"Σφάλμα στα δεδομένα: {exc}")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "1. Data & Validity Metrics",
    "2. Bland-Altman Metrology",
    "3. Diagnostic Alerts",
])

with tab1:
    st.subheader("Δεδομένα")
    st.dataframe(
        df[["ID", "Weight_Precision", "Weight_Simple", "Diff_d", "Mean_m"]],
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Validity & Agreement Metrics")
    metric_names = list(metrics.keys())
    metric_cols = st.columns(4)
    for i, name in enumerate(metric_names):
        val = metrics[name]
        suffix = " kg" if name in ["SEE", "Mean Bias", "SD_d", "Upper LoA", "Lower LoA"] else ""
        text = f"{val:.3f}{suffix}" if np.isfinite(val) else "N/A"
        metric_cols[i % 4].metric(name, text)
    st.download_button(
        "Λήψη αποτελεσμάτων CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bland_altman_results.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Bland–Altman Metrology")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Mean_m"],
        y=df["Diff_d"],
        mode="markers+text",
        text=df["ID"],
        textposition="top center",
        name="Μετρήσεις",
        hovertemplate="ID=%{text}<br>Mean_m=%{x:.2f} kg<br>Diff_d=%{y:.2f} kg<extra></extra>",
    ))
    for value, label in [
        (metrics["Mean Bias"], "Mean Bias"),
        (metrics["Upper LoA"], "Upper LoA"),
        (metrics["Lower LoA"], "Lower LoA"),
    ]:
        fig.add_hline(
            y=value,
            line_dash="dash",
            annotation_text=f"{label}: {value:.3f} kg",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"Διάγραμμα Bland–Altman — {scenario}",
        xaxis_title="Mean_m (kg)",
        yaxis_title="Diff_d (kg)",
        template="plotly_white",
        height=600,
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Diagnostic Alerts")
    abs_bias = abs(metrics["Mean Bias"])
    if abs_bias > 1.0:
        st.error(
            f"ΠΡΟΕΙΔΟΠΟΙΗΣΗ: |Mean Bias| = {abs_bias:.3f} kg, "
            "μεγαλύτερο από 1.0 kg."
        )
    else:
        st.success(
            f"Δεν ενεργοποιείται προειδοποίηση: |Mean Bias| = {abs_bias:.3f} kg."
        )
    st.write(f"Σενάριο: **{scenario}**")
    st.write(f"Πλήρη ζεύγη μετρήσεων: **{len(df)}**")

st.divider()
st.caption("Εκπαιδευτική εφαρμογή για το Πρωτόκολλο 4.")
