"""Streamlit dashboard for the SHM decision-support system."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import BearingDataLoader
from src.data.preprocessor import Preprocessor
from src.utils.config import CFG
from src.utils.seed import set_all_seeds

st.set_page_config(
    page_title="SHM Decision-Support System",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = f"http://localhost:{CFG['api']['port']}"
STATUS_COLOURS = {
    "SAFE": "#22c55e",
    "WARNING": "#f59e0b",
    "CRITICAL": "#ef4444",
}

set_all_seeds(int(CFG.get("project.seed") or 42))

st.markdown(
    """
<style>
    .status-box {
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 14px;
        border: 2px solid transparent;
    }
    .status-text {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    loader = BearingDataLoader(CFG)
    X, y = loader.load()
    splitter = Preprocessor(
        train_frac=float(CFG["data"]["train_frac"]),
        val_frac=float(CFG["data"]["val_frac"]),
        test_frac=float(CFG["data"]["test_frac"]),
        seed=int(CFG.get("project.seed") or 42),
    )
    X_te, y_te = splitter.split(X, y)["test"]
    return X_te, y_te


def _call_json(method: str, path: str, payload: dict | None = None, timeout: int = 10) -> dict:
    try:
        response = requests.request(method, f"{API_URL}{path}", json=payload, timeout=timeout)
        data = response.json()
        if response.ok:
            return data
        return {
            "error": data.get("detail", response.text),
            "error_type": data.get("error", "request_error"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def call_predict(signal: np.ndarray, model: str) -> dict:
    return _call_json("POST", "/predict", {"signal": signal.tolist(), "model": model})


def call_explain(signal: np.ndarray) -> dict:
    return _call_json("POST", "/explain", {"signal": signal.tolist()})


def call_health() -> dict:
    return _call_json("GET", "/health", timeout=4)


def call_metrics() -> dict:
    return _call_json("GET", "/metrics", timeout=4)


def plot_signal(signal: np.ndarray, sampling_rate: int, title: str) -> None:
    import plotly.graph_objects as go
    from scipy.fft import rfft, rfftfreq

    time_axis = np.linspace(0.0, len(signal) / sampling_rate, len(signal))
    frequencies = rfftfreq(len(signal), d=1.0 / sampling_rate)
    magnitudes = np.abs(rfft(signal))

    time_fig = go.Figure()
    time_fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal,
            mode="lines",
            line=dict(color="#38bdf8", width=1.2),
            name="signal",
        )
    )
    time_fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(time_fig, use_container_width=True)

    fft_fig = go.Figure()
    fft_fig.add_trace(
        go.Scatter(
            x=frequencies[: len(frequencies) // 4],
            y=magnitudes[: len(magnitudes) // 4],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#0f766e", width=1.2),
            name="fft",
        )
    )
    fft_fig.update_layout(
        title="FFT Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=240,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(fft_fig, use_container_width=True)


def plot_probabilities(class_probs: dict[str, float]) -> None:
    import plotly.graph_objects as go

    labels = list(class_probs.keys())
    values = list(class_probs.values())
    highlight = max(values) if values else 0.0
    colours = ["#22c55e" if value == highlight else "#64748b" for value in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{value * 100:.1f}%" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], title="Probability"),
        yaxis=dict(title=None),
        height=220,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def health_gauge(index: float) -> None:
    import plotly.graph_objects as go

    colour = "#22c55e" if index >= 0.75 else "#f59e0b" if index >= 0.45 else "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(index * 100.0, 2),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, 45], "color": "#fecaca"},
                    {"range": [45, 75], "color": "#fde68a"},
                    {"range": [75, 100], "color": "#bbf7d0"},
                ],
            },
            title={"text": "Health Index"},
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_status(status: str) -> None:
    colour = STATUS_COLOURS.get(status, "#475569")
    st.markdown(
        f"""
<div class="status-box" style="background:{colour}22; border-color:{colour};">
    <div class="status-text" style="color:{colour};">{status}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Structural Health Monitoring")
    st.caption("AI-based bearing fault detection and decision support")

    class_names = [CFG["data"]["classes"][i] for i in range(len(CFG["data"]["classes"]))]
    sampling_rate = int(CFG["data"]["sampling_rate"])
    window_size = int(CFG["data"]["window_size"])

    with st.sidebar:
        st.subheader("Connection")
        health = call_health()
        if "error" in health:
            st.error(f"API unavailable: {health['error']}")
            st.code("python -m shm serve")
            api_ready = False
        else:
            artifacts = health["artifacts"]
            st.success("API connected")
            for key in ("preprocessor", "rf", "cnn", "anomaly", "manifest"):
                st.write(f"{'OK' if artifacts.get(key) else 'MISSING'}  {key}")
            if artifacts.get("compatible", False):
                st.caption("Artifacts match the active config.")
                api_ready = True
            else:
                st.warning("Artifacts are incompatible with the active config.")
                for issue in artifacts.get("issues", []):
                    st.caption(f"- {issue}")
                api_ready = False

            metrics = call_metrics()
            if "error" not in metrics:
                st.divider()
                st.subheader("Runtime")
                st.write(f"Requests: {metrics['requests_total']}")
                st.write(f"Failure rate: {metrics['failure_rate'] * 100:.1f}%")
                st.write(f"p95 latency: {metrics['latency_ms']['p95']:.2f} ms")
                if metrics["cache"]["enabled"]:
                    st.write(f"Cache hit rate: {metrics['cache']['hit_rate'] * 100:.1f}%")

        st.divider()
        st.subheader("Inference")
        model_choice = st.selectbox("Classifier", ["cnn", "rf"], index=0)
        input_mode = st.radio("Input source", ["Sample from test set", "Upload .npy file", "Synthetic sample"])

        signal_label = "Unknown"
        signal = np.zeros(window_size, dtype=np.float32)

        if input_mode == "Sample from test set":
            X_te, y_te = load_dataset()
            class_filter = st.selectbox("Filter by class", ["All"] + class_names)
            if class_filter == "All":
                candidate_indices = np.arange(len(X_te))
            else:
                class_id = class_names.index(class_filter)
                candidate_indices = np.where(y_te == class_id)[0]

            sample_idx = st.slider("Sample index", 0, max(0, len(candidate_indices) - 1), 0)
            chosen = int(candidate_indices[sample_idx])
            signal = X_te[chosen]
            signal_label = class_names[int(y_te[chosen])]
            st.info(f"True label: {signal_label}")
        elif input_mode == "Upload .npy file":
            uploaded = st.file_uploader("Upload a 1-D .npy signal", type=["npy"])
            if uploaded is not None:
                loaded_signal = np.load(uploaded).astype(np.float32).flatten()
                if loaded_signal.shape[0] != window_size:
                    st.error(f"Expected {window_size} samples, received {loaded_signal.shape[0]}.")
                else:
                    signal = loaded_signal
                    st.success("Signal loaded")
        else:
            fault_type = st.selectbox("Synthetic class", class_names)
            signal_label = fault_type
            synthetic_loader = BearingDataLoader(CFG)
            signal = synthetic_loader._synthesise_window(
                class_names.index(fault_type),
                np.random.default_rng(int(CFG.get("project.seed") or 42)),
            )

        run_btn = st.button("Run analysis", type="primary", use_container_width=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("Signal")
        plot_signal(signal, sampling_rate, f"Window - true label: {signal_label}")

    with col_right:
        st.subheader("Decision")
        if not api_ready:
            st.warning("Start the API and train compatible artifacts to enable inference.")
            st.stop()

        if "last_result" not in st.session_state:
            st.session_state["last_result"] = None
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if run_btn:
            with st.spinner("Running inference..."):
                result = call_predict(signal, model_choice)
            st.session_state["last_result"] = result
            if "error" not in result:
                st.session_state["history"] = [
                    {
                        "status": result["agent_output"]["status"],
                        "prediction": result["model_output"]["predicted_label"],
                        "confidence": round(result["model_output"]["confidence"], 4),
                    },
                    *st.session_state["history"],
                ][:5]

        result = st.session_state["last_result"]
        if result is None:
            st.info("Run analysis to request a live prediction from the backend.")
        elif "error" in result:
            st.error(result["error"])
        else:
            model_output = result["model_output"]
            agent_output = result["agent_output"]
            render_status(agent_output["status"])
            health_gauge(model_output["health_index"])

            metric_cols = st.columns(3)
            metric_cols[0].metric("Prediction", model_output["predicted_label"])
            metric_cols[1].metric("Confidence", f"{model_output['confidence'] * 100:.1f}%")
            anomaly_value = model_output["anomaly_score"]
            metric_cols[2].metric("Anomaly", "N/A" if anomaly_value is None else f"{anomaly_value:.3f}")

            st.markdown("**Class probabilities**")
            plot_probabilities(model_output["class_probs"])

            st.markdown("**Rationale**")
            st.write(agent_output["rationale"])
            st.caption(f"Urgency: {agent_output['urgency']}")

            st.markdown("**Recommended action**")
            st.write(agent_output["action"])

            with st.expander("Signal explanation"):
                explain = call_explain(signal)
                if "error" in explain:
                    st.warning(explain["error"])
                else:
                    st.write(f"Dominant frequency: {explain['fft_peak_hz']:.2f} Hz")
                    st.json(explain["signal_stats"])
                    st.dataframe(explain["top_features"], use_container_width=True, hide_index=True)
                    st.caption(explain["note"])

            if st.session_state["history"]:
                st.markdown("**Recent predictions**")
                st.dataframe(st.session_state["history"], use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Use `python -m shm train`, `python -m shm serve`, and `python -m shm evaluate` for the full workflow.")


if __name__ == "__main__":
    main()
