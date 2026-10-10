import calendar
import json 
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Yield Forecast", page_icon="🌾", layout="wide")


@st.cache_resource
def load():
    here = Path(__file__).parent
    for folder in (here / "artifacts", here.parent / "artifacts", Path("artifacts")):
        f = folder / "forecast_model.json"
        if f.exists():
            return json.load(open(f))
    st.error("forecast_model.json not found. Put it in an 'artifacts' folder next to app.py.")
    st.stop()


A = load()
V = pd.DataFrame(A["validation"]).set_index("checkpoint")
VN = {"PRECTOTCORR": "Rain (mm/day)", "T2M_MAX": "Max temp (°C)", "T2M_MIN": "Min temp (°C)",
      "GWETROOT": "Root-zone soil wetness (0-1)", "ALLSKY_SFC_SW_DWN": "Solar radiation (MJ/m²/day)",
      "RAINYFRAC": "Share of rainy days", "DRYSPELL": "Longest dry spell (days)"}
WN = {"PRE": "Mar-May to date", "MON": "Jun-Sep to date", "POST": "Oct-Nov to date"}
LAST_YEAR = 2018      # trend is held flat beyond this year


def forecast(model, district, crop, year, x):
    cols = A["columns"] + (model["features"] if model["weather_scale"] > 0 else [])
    idx = {c: i for i, c in enumerate(cols)}
    v = np.zeros(len(cols))
    v[idx[f"cd|{crop}|{district}"]] = 1.0
    v[idx[f"ct|{crop}"]] = (min(year, LAST_YEAR) - 2007 - 5) / 3.2
    if model["weather_scale"] > 0:
        for f in model["features"]:
            v[idx[f]] = (x[f] - model["mean"][f]) / model["std"][f]
    z = model["intercept"] + v @ np.array(model["coef"])
    sd, mu = model["sd"][crop], model["mu"][crop]
    g = lambda zz: float(np.exp(zz * sd + mu))
    return g(z), (g(z - model["q80"]), g(z + model["q80"])), (g(z - model["q94"]), g(z + model["q94"]))


st.sidebar.title("🌾 Crop Yield Forecast")
st.sidebar.caption("Kolhapur · Sangli · Satara")
page = st.sidebar.radio("Page", ["Forecast", "Validation", "Biases and limits"])

if page == "Forecast":
    st.title("In-season yield forecast (any date, 1 May - 30 Nov)")
    c1, c2, c3 = st.columns(3)
    district = c1.selectbox("District", A["districts"])
    crop = c2.selectbox("Crop", A["crops"], format_func=str.title)
    year = c3.slider("Year", 2008, 2030, 2018)
    d1, d2 = st.columns(2)
    mon = d1.selectbox("Forecast as of: month", list(range(5, 12)), index=3,
                       format_func=lambda m: calendar.month_name[m])
    day = d2.slider("Day", 1, 30 if mon in (6, 9, 11) else 31, 15)
    label = f"{mon:02d}-{day:02d}"
    model = A["models"][label]
    if year > LAST_YEAR:
        st.info("Beyond 2018 the trend is held flat, so only weather and the district-crop level change.")

    x = {}
    if model["weather_scale"] > 0:
        st.subheader("Weather observed so far this season")
        st.caption("Defaults are this district's 2007-2017 averages for the same date. Replace them with observed "
                   "to-date means (update daily or weekly as the season runs).")
        cols = st.columns(3)
        for i, f in enumerate(model["features"]):
            var, win = f.rsplit("_", 1)
            lo, hi = model["lo"][f], model["hi"][f]
            pad = 0.1 * (hi - lo) or 0.1
            x[f] = cols[i % 3].slider(f"{VN[var]}, {WN[win]}", float(lo - pad), float(hi + pad),
                                      float(model["clim"][district][f]), step=float((hi - lo + 2 * pad) / 200),
                                      key=f"{label}_{district}_{f}")
    else:
        st.info("For this date the fitted model gives weather zero weight, so the forecast uses the "
                "district-crop level and trend only.")
    med, r80, r94 = forecast(model, district, crop, year, x)
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast (median)", f"{med:,.0f} kg/ha")
    m2.metric("80% range", f"{r80[0]:,.0f} - {r80[1]:,.0f}")
    m3.metric("94% range", f"{r94[0]:,.0f} - {r94[1]:,.0f}")
    st.write(f"2007-2017 average for {crop.title()} in {district}: **{A['hist_mean'][crop][district]:,.0f} kg/ha**.")
    v = V.loc[label]
    msg = (f"Back-test at {label}: weather changed the error vs a no-weather baseline by "
           f"{v['skill_vs_baseline_%']:+.1f}% (95% CI on the log-error gain {v['gain_CI_low']:+.3f} to "
           f"{v['gain_CI_high']:+.3f}). The 80% range held the actual yield {v['coverage80_%']:.0f}% of the time.")
    (st.success if v["gain_CI_low"] > 0 else st.warning)(
        msg + ("" if v["gain_CI_low"] > 0 else " Weather is not shown to beat the baseline at this date."))

elif page == "Validation":
    st.title("Past-only validation (train on earlier years, test 2012-2017)")
    st.write("Skill > 0 means the weather model beats the baseline (district-crop level + trend). "
             "Neighbouring dates share almost the same data, so they are not independent tests.")
    st.line_chart(V["skill_vs_baseline_%"])
    show = V.iloc[::7]
    st.write("Weekly checkpoints:")
    st.dataframe(show.round(2))
    st.subheader("By season group (MAPE %)")
    st.dataframe(pd.DataFrame(A["validation_groups"]))

else:
    st.title("Biases addressed and what remains")
    st.markdown("""
**Reduced**
- **Look-ahead features:** each forecast uses only days up to its as-of date.
- **Validation leakage:** only earlier years are used to predict a later year.
- **Preprocessing leakage:** scaling, crop statistics and the weather penalty are fitted inside each training window.
- **Feature-selection bias:** the feature set was fixed in advance (to-date means over three season windows). A second set with rain-pattern features was tested and rejected by a rule set before the run.
- **Trend extrapolation:** the trend is held flat after 2018.

**Soil fertility is not in the data.** Fixed fertility, soil type and irrigation are absorbed by district-crop levels;
changes over time in fertiliser use or soil cannot be seen and sit in the error.

**Still present:** 33 district-years; district-level yield; one weather source; no pest or price data;
80% ranges are slightly narrow; 214 checkpoints are correlated, so a few "significant" dates are expected by chance;
yield is recorded once a year, so day- and week-level means a forecast update, not a daily yield.
""")
