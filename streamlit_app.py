import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "dashboard" / "app_data"
REPO_URL = "https://github.com/talhabamusa/air-quality-platform"

st.set_page_config(page_title="Air Quality Intelligence Platform", layout="wide")


@st.cache_data
def load_data():
    return {
        "mart": pd.read_csv(DATA_DIR / "mart_city_daily.csv", parse_dates=["date_local"]),
        "forecast": pd.read_csv(DATA_DIR / "forecast_latest.csv"),
        "tiers": pd.read_csv(DATA_DIR / "quality_tiers.csv"),
        "profile": pd.read_csv(DATA_DIR / "hourly_profile.csv"),
        "runs": pd.read_csv(DATA_DIR / "model_runs.csv"),
        "backtest": pd.read_csv(DATA_DIR / "backtest_predictions.csv", parse_dates=["target_date"]),
        "meta": json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8")),
    }


def nice(name):
    return str(name).title()


data = load_data()
mart, forecast, tiers = data["mart"], data["forecast"], data["tiers"]
profile, runs, backtest, meta = data["profile"], data["runs"], data["backtest"], data["meta"]
cities = sorted(mart["city_key"].unique())
last_day = mart["date_local"].max().date()

st.title("Air Quality Intelligence Platform")
st.caption(f"PM2.5 and weather for five Indian cities. Data window: {meta['data_window']} "
           f"(snapshot generated {meta['generated_at_utc'][:10]}).")

tab_fc, tab_trend, tab_day, tab_quality, tab_models, tab_about = st.tabs(
    ["Latest forecast", "Trends", "Daily cycle", "Data quality", "Model results", "About"])

# ---------------------------------------------------------------- forecast
with tab_fc:
    st.subheader("Next-day PM2.5 forecast, by city")
    st.warning(
        f"This is a snapshot, not a live feed: the data ends on {last_day}, so these forecasts are for days "
        "in the past. In backtests, no model clearly beat simply assuming that tomorrow equals today "
        "(see the Model results tab).")
    ordered = forecast.sort_values("city_key")
    for col, (_, r) in zip(st.columns(len(ordered)), ordered.iterrows()):
        with col:
            st.metric(label=f"{nice(r['city_key'])}: forecast for {r['forecast_date']}",
                      value=f"{r['forecast_median']:.0f} ug/m3",
                      delta=f"{r['forecast_median'] - r['today_pm25']:+.0f} vs {r['as_of_date']}",
                      delta_color="off")
            st.caption(f"80% range: {r['forecast_p10']:.0f} to {r['forecast_p90']:.0f}")
    st.dataframe(
        ordered.rename(columns={
            "city_key": "city", "as_of_date": "last data day", "forecast_date": "forecast for",
            "today_pm25": "last value", "persistence": "persistence forecast",
            "forecast_p10": "10% quantile", "forecast_median": "median", "forecast_p90": "90% quantile",
        }).round(1),
        hide_index=True)
    st.caption("Values are daily city means of PM2.5 in micrograms per cubic metre. The range is an 80% "
               "interval from quantile regression on the change from the last value.")

# ------------------------------------------------------------------ trends
with tab_trend:
    st.subheader("Daily PM2.5 by city")
    chosen = st.multiselect("Cities", cities, default=cities, format_func=nice)
    smooth = st.checkbox("Show 7-day average", value=True)
    view = mart[mart["city_key"].isin(chosen)]
    if view.empty:
        st.info("Pick at least one city.")
    else:
        wide = view.pivot(index="date_local", columns="city_key", values="pm25_mean_ugm3").asfreq("D")
        if smooth:
            wide = wide.rolling(7, min_periods=3).mean()
        st.line_chart(wide.rename(columns=nice))
        st.caption("Gaps are days when too few sensors reported to compute a city value.")

        monthly = (view.assign(month=view["date_local"].dt.strftime("%Y-%m"))
                   .pivot_table(index="month", columns="city_key", values="pm25_mean_ugm3", aggfunc="mean")
                   .round(1).rename(columns=nice))
        st.write("Monthly mean of the daily city values (the first and last months are partial):")
        st.dataframe(monthly)

    st.subheader("Dry days vs rainy days")
    rain = (mart.assign(rainy=mart["precip_total_mm"] >= 1)
            .pivot_table(index="city_key", columns="rainy", values="pm25_mean_ugm3", aggfunc="mean")
            .rename(columns={False: "dry days", True: "rainy days (1 mm or more)"}).rename(index=nice))
    st.bar_chart(rain)
    st.caption("Mean daily PM2.5. Season affects both rain and pollution, so this describes the data "
               "and does not show that rain causes the difference.")

# ------------------------------------------------------------- daily cycle
with tab_day:
    st.subheader("A typical day: PM2.5 by hour")
    prof = profile.pivot(index="hour", columns="city_key", values="pm25")
    mode = st.radio("Show", ["Relative to each city's own average", "Actual ug/m3"], horizontal=True)
    shown = prof / prof.mean() if mode.startswith("Relative") else prof
    st.line_chart(shown.rename(columns=nice))
    st.caption("Hour of day in India Standard Time (start of the hour). Averages use observed and "
               "imputed hours over the whole year.")

# ----------------------------------------------------------- data quality
with tab_quality:
    st.subheader("How much of the data is measured?")
    t = tiers.pivot(index="city_key", columns="quality_tier", values="hours").fillna(0)
    pct = t.div(t.sum(axis=1), axis=0) * 100
    order = [c for c in ["observed", "imputed", "imputed_low_confidence", "missing"] if c in pct.columns]
    st.bar_chart(pct[order].rename(index=nice))
    st.dataframe(pct[order].round(1).rename(index=nice))
    st.markdown(
        "- **observed**: a real reading that passed the quality rules\n"
        "- **imputed**: a short gap filled by interpolation or from neighbouring stations (benchmarked)\n"
        "- **imputed_low_confidence**: a longer gap filled with a method that tended to run high; "
        "kept apart and left out of the daily city values\n"
        "- **missing**: gaps over 24 hours, edges and quarantined readings, never filled")

# ----------------------------------------------------------- model results
with tab_models:
    st.subheader("How good are the forecasts?")
    st.markdown(
        "Each method forecasts tomorrow's city PM2.5 and was tested on four consecutive 30-day windows "
        "(July to early November 2026), always trained only on earlier data. The baseline is "
        "**persistence**: tomorrow equals today. Errors are in ug/m3 (lower is better). "
        "`diff_ci` is a 95% range for how much a method's average error differs from persistence; "
        "when it contains 0, the two cannot be told apart. `change_oracle_weather` peeks at tomorrow's "
        "real weather and is an upper bound, not a usable forecast.")
    f = runs[runs["experiment"] == "pm25-next-day-forecast"]
    cols = [c for c in ["run", "mae", "rmse", "mae_vs_persistence_pct", "win_rate_vs_persistence_pct",
                        "diff_ci_low", "diff_ci_high"] if c in f.columns]
    st.dataframe(f[cols].round(2), hide_index=True)

    st.subheader("80% prediction intervals")
    i = runs[runs["experiment"] == "pm25-next-day-intervals"]
    icols = [c for c in ["run", "coverage_pct", "mean_width", "interval_score", "below_pct", "above_pct",
                         "median_mae", "median_diff_ci_low", "median_diff_ci_high"] if c in i.columns]
    st.dataframe(i[icols].round(1), hide_index=True)
    st.caption("Coverage should be close to 80: the share of days the real value fell inside the range.")

    st.subheader("Forecast vs actual in the test windows")
    city = st.selectbox("City", cities, format_func=nice, index=cities.index("delhi") if "delhi" in cities else 0)
    bt = backtest[backtest["city_key"] == city]
    base = alt.Chart(bt).encode(x=alt.X("target_date:T", title=None))
    band = base.mark_area(opacity=0.25).encode(y=alt.Y("model_lo:Q", title="PM2.5 (ug/m3)"), y2="model_hi:Q")
    actual = base.mark_line(color="black").encode(y="target:Q")
    median = base.mark_line(strokeDash=[4, 2]).encode(y="model_med:Q")
    persist = base.mark_line(color="orange", opacity=0.7).encode(y="persistence:Q")
    st.altair_chart((band + actual + median + persist).properties(height=380, width="container"))
    st.caption("Black: actual. Dashed blue: model median. Shaded: model 80% range. Orange: persistence.")

# ------------------------------------------------------------------- about
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
An end-to-end data platform: OpenAQ air-quality readings and Open-Meteo weather are downloaded, cleaned,
loaded into a PostgreSQL warehouse modelled with dbt, analysed and used for next-day forecasts.
Source code and write-up: [{REPO_URL}]({REPO_URL})

**Problems found and handled along the way**
- Half-hour timestamp offsets in the source (hourly buckets start at :30 UTC), found when a safety check
  failed, and handled explicitly.
- Frozen and faulty sensors, caught by quality rules and kept in a quarantine table with the reason.
- One sensor inflated a city average on days when it was the only one reporting; a minimum-sensor rule and
  a test now prevent that.
- Gap filling was benchmarked by hiding known values, and the methods were chosen by gap length.

**Limitations**
- This is a static snapshot, not a live feed.
- Weather is gridded model data at one point per city, not station measurements.
- The gap-filling rules and quality thresholds were tuned on one month of data.
- The test windows cover July to November; the winter smog season is only in the training data.
- No model clearly beat the persistence baseline on average error.

Data: air quality from [OpenAQ](https://openaq.org); weather from [Open-Meteo](https://open-meteo.com)
(CC BY 4.0).
""")