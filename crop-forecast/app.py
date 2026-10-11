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
import calendar
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Yield Forecast", page_icon="ðŸŒ¾", layout="wide")


@st.cache_resource
def load():
    here = Path(__file__).parent
    for folder in (here / "artifacts", here.parent / "artifacts", Path("artifacts")):
        f = folder / "forecast_model.json"
        if f.exists():
            with open(f) as fh:
                return json.load(fh)
    st.error("forecast_model.json not found. Put it in an 'artifacts' folder next to app.py.")
    st.stop()


A = load()
V = pd.DataFrame(A["validation"]).set_index("checkpoint")
VN = {"PRECTOTCORR": "Rain (mm/day)", "T2M_MAX": "Max temp (Â°C)", "T2M_MIN": "Min temp (Â°C)",
      "GWETROOT": "Root-zone soil wetness (0-1)", "ALLSKY_SFC_SW_DWN": "Solar radiation (MJ/mÂ²/day)",
      "RAINYFRAC": "Share of rainy days", "DRYSPELL": "Longest dry spell (days)"}
WN = {"PRE": "Mar-May to date", "MON": "Jun-Sep to date", "POST": "Oct-Nov to date"}

LAST_OBS = 2017      # last year with yield data (training window 2007-2017)
N_TRAIN = 11         # years of data per district-crop
T_MEAN, T_SCALE = 2012, 3.2   # centring / scaling of the trend term used when the model was fitted
SXX = N_TRAIN * 10.0          # sum of squares of year about its mean (years 2007-2017)
Z = NormalDist().inv_cdf


def trend_year(year, mode, phi=0.8):
    """Effective year fed to the trend term."""
    if year <= LAST_OBS:
        return year
    if mode == "Flat after 2017":
        return LAST_OBS
    h = year - LAST_OBS                       # damped: increments shrink by phi each year
    return LAST_OBS + phi * (1 - phi ** h) / (1 - phi)


def horizon_factor(year):
    """Widening of the prediction interval beyond the data (OLS prediction-interval ratio, heuristic)."""
    if year <= LAST_OBS:
        return 1.0
    f = lambda y: np.sqrt(1 + 1 / N_TRAIN + (y - T_MEAN) ** 2 / SXX)
    return float(f(year) / f(LAST_OBS))


def calibration(label):
    """Inflate bands so back-test coverage matches the nominal 80% / 94% (never narrower than fitted)."""
    v = V.loc[label]
    c80 = Z(0.90) / Z(0.5 + v["coverage80_%"] / 200)
    c94 = Z(0.97) / Z(0.5 + v["coverage94_%"] / 200)
    return float(np.clip(c80, 1, 1.6)), float(np.clip(c94, 1, 1.6))


def forecast(model, label, district, crop, year, x, mode, phi=0.8):
    cols = A["columns"] + (model["features"] if model["weather_scale"] > 0 else [])
    idx = {c: i for i, c in enumerate(cols)}
    v = np.zeros(len(cols))
    v[idx[f"cd|{crop}|{district}"]] = 1.0
    v[idx[f"ct|{crop}"]] = (trend_year(year, mode, phi) - T_MEAN) / T_SCALE
    if model["weather_scale"] > 0:
        for f in model["features"]:
            v[idx[f]] = (x[f] - model["mean"][f]) / model["std"][f]
    z = model["intercept"] + v @ np.array(model["coef"])
    sd, mu = model["sd"][crop], model["mu"][crop]
    g = lambda zz: float(np.exp(zz * sd + mu))
    k80, k94 = calibration(label)
    h = horizon_factor(year)
    q80, q94 = model["q80"] * k80 * h, model["q94"] * k94 * h
    return g(z), (g(z - q80), g(z + q80)), (g(z - q94), g(z + q94))


def trend_pct(model, crop):
    i = A["columns"].index(f"ct|{crop}")
    return 100 * model["coef"][i] * model["sd"][crop] / T_SCALE


st.sidebar.title("ðŸŒ¾ Crop Yield Forecast")
st.sidebar.caption("Kolhapur Â· Sangli Â· Satara")
page = st.sidebar.radio("Page", ["Forecast", "Validation", "Biases and limits"])

if page == "Forecast":
    st.title("In-season yield forecast (any date, 1 May - 30 Nov)")
    c1, c2, c3 = st.columns(3)
    district = c1.selectbox("District", A["districts"])
    crop = c2.selectbox("Crop", A["crops"], format_func=str.title)
    year = c3.slider("Year", 2008, 2030, 2017)
    d1, d2 = st.columns(2)
    mon = d1.selectbox("Forecast as of: month", list(range(5, 12)), index=3,
                       format_func=lambda m: calendar.month_name[m])
    day = d2.slider("Day", 1, 30 if mon in (6, 9, 11) else 31, 15)
    label = f"{mon:02d}-{day:02d}"
    model = A["models"][label]

    mode = st.radio("Trend after 2017 (no yield data beyond 2017)",
                    ["Flat after 2017", "Damped trend"], horizontal=True,
                    help="Flat: no further trend. Damped: the 2007-2017 trend continues but fades by 20% each year.")
    tp = trend_pct(model, crop)
    if year > LAST_OBS:
        st.info(f"Projection year: the fitted 2007-2017 trend for {crop.title()} is {tp:+.1f}% per year. "
                f"Bands widen by x{horizon_factor(year):.2f} because the year is outside the data.")
        if abs(tp) > 5:
            st.warning("This crop's fitted trend is very steep for only 11 years. Do not use the damped option "
                       "for it.")

    x = {}
    if model["weather_scale"] > 0:
        st.subheader("Weather observed so far this season")
        st.caption("Defaults are this district's 2007-2017 averages for the same date. Replace them with observed "
                   "to-date means (update daily or weekly as the season runs). For future years, use a scenario.")
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

    med, r80, r94 = forecast(model, label, district, crop, year, x, mode)
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast (median)", f"{med:,.0f} kg/ha")
    m2.metric("80% range", f"{r80[0]:,.0f} - {r80[1]:,.0f}")
    m3.metric("94% range", f"{r94[0]:,.0f} - {r94[1]:,.0f}")
    st.write(f"2007-2017 average for {crop.title()} in {district}: **{A['hist_mean'][crop][district]:,.0f} kg/ha**.")

    # Multi-year view with widening uncertainty (same weather inputs for every year)
    yrs = list(range(LAST_OBS, 2031))
    rows = []
    for y in yrs:
        m_, a_, b_ = forecast(model, label, district, crop, y, x, mode)
        rows.append({"year": y, "median": m_, "80% low": a_[0], "80% high": a_[1],
                     "94% low": b_[0], "94% high": b_[1]})
    st.subheader("Projection 2017-2030 (weather held at the values above)")
    st.line_chart(pd.DataFrame(rows).set_index("year")[["median", "80% low", "80% high", "94% low", "94% high"]])
    st.caption("A flat median is expected when there is no trend and no weather change. Compare the width of the "
               "bands: they widen with distance from the data, and that is the honest forecast signal.")

    v = V.loc[label]
    msg = (f"Back-test at {label}: weather changed the error vs a no-weather baseline by "
           f"{v['skill_vs_baseline_%']:+.1f}% (95% CI on the log-error gain {v['gain_CI_low']:+.3f} to "
           f"{v['gain_CI_high']:+.3f}). The original 80% range held the actual yield {v['coverage80_%']:.0f}% of "
           f"the time; bands shown here are widened to correct this.")
    n_sig = int((V["gain_CI_low"] > 0).sum())
    msg += (f" Note: {n_sig} of {len(V)} dates show a positive gain, but the dates are highly correlated "
            f"and no multiplicity correction was applied, so treat single dates cautiously.")
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
    st.markdown(""" **Reduced** - **Look-ahead features:** each forecast uses only days up to its as-of date. - **Validation leakage:** only earlier years are used to predict a later year. - **Preprocessing leakage:** scaling, crop statistics and the weather penalty are fitted inside each training window. - **Feature-selection bias:** the feature set was fixed in advance (to-date means over three season windows). - **Trend extrapolation:** the trend is flat after 2017 by default. A damped option is available but unvalidated. - **Interval calibration:** back-test coverage of the 80% range was about 72%. Bands are widened to match, and widened further for years beyond the data (OLS prediction-interval ratio, a heuristic). **Soil fertility is not in the data.** Fixed fertility, soil type and irrigation are absorbed by district-crop levels; changes over time in fertiliser use or soil cannot be seen and sit in the error. **Still present:** 33 district-years; yields only to 2017 (2018-25 missing); district-level yield; one weather source; no pest or price data; 214 correlated checkpoints, so a few "significant" dates are expected by chance; yield is recorded once a year, so day- and week-level means a forecast update, not a daily yield. The model forecasts yield (kg/ha), not total production (that also needs area under crop). """)
