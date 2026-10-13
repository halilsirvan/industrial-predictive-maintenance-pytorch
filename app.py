"""Interactive Streamlit demo for predictive maintenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import FailurePredictor


ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_predictor() -> FailurePredictor:
    return FailurePredictor(ROOT / "artifacts")


st.set_page_config(page_title="Industrial Failure Risk", page_icon="⚙️", layout="wide")
st.title("⚙️ Industrial Equipment Failure Risk")
st.caption("PyTorch neural network trained on UCI AI4I 2020 data — portfolio demonstration")

st.info(
    "Enter current operating conditions. The output is a calibrated model estimate, "
    "not a safety-certified maintenance instruction."
)

left, right = st.columns(2)
with left:
    product_type = st.selectbox("Product quality type", ["L", "M", "H"], index=1)
    air_temperature = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    process_temperature = st.slider("Process temperature (K)", 305.0, 315.0, 310.0, 0.1)
with right:
    rotational_speed = st.slider("Rotational speed (rpm)", 1100, 2900, 1500, 10)
    torque = st.slider("Torque (Nm)", 3.0, 80.0, 40.0, 0.5)
    tool_wear = st.slider("Tool wear (min)", 0, 260, 100, 1)

inputs = pd.DataFrame(
    [
        {
            "Type": product_type,
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }
    ]
)

if st.button("Estimate failure risk", type="primary", use_container_width=True):
    result = load_predictor().predict(inputs).iloc[0]
    probability = float(result["failure_probability"])
    threshold = float(result["decision_threshold"])
    st.metric("Estimated failure probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if bool(result["maintenance_alert"]):
        st.error(f"Maintenance alert: risk exceeds the model threshold ({threshold:.1%}).")
    else:
        st.success(f"No alert: risk is below the model threshold ({threshold:.1%}).")

with st.expander("What the model uses"):
    st.write(
        "Air and process temperature, rotational speed, torque, tool wear, product type, "
        "and three label-independent engineering features: temperature gap, mechanical "
        "power, and a torque–wear strain proxy. Failure-mode flags are deliberately excluded "
        "to prevent target leakage."
    )

