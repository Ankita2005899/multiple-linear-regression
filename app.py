"""Mumbai House Price Predictor - Streamlit app trained on real listings (run: streamlit run app.py)."""
import base64, io, json, difflib
import pandas as pd
import streamlit as st
from src import config, predict as predictor
from src.explain import explain

st.set_page_config(page_title="Mumbai House Price Predictor", page_icon="🏠", layout="wide")
OUT = config.OUT_DIR

@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_PATH)

def load_metrics():
    p = OUT / "best_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

D = load_data()
REGIONS = sorted(D["region"].unique())
TYPES = sorted(D["type"].unique())
for k, v in dict(f_region="Thane West" if "Thane West" in REGIONS else REGIONS[0], f_type="Apartment",
                 f_status="Ready to move", f_age="New", uses=0).items():
    st.session_state.setdefault(k, v)

if not config.MODEL_PATH.exists():
    from src import train
    train.main()

def run_model(row):
    try:
        return float(predictor.predict(row)[0])
    except Exception:            # e.g. library version mismatch: retrain once on this machine
        from src import train
        train.main()
        return float(predictor.predict(row)[0])

def fmt_inr(lakh):
    return f"₹{lakh / 100:,.2f} crore" if lakh >= 100 else f"₹{lakh:,.1f} lakh"

PROMPT = """Each photo is preceded by a text label giving its room type. Estimate the floor area of each room in square metres.
Return ONLY a JSON object: {"rooms": [{"room_type": "label", "area_sqm": number or null, "note": "short"}], "notes": "one sentence on what you could not tell"}.
List the rooms in the same order as the photos. Use null if the size cannot be judged. Never invent values."""
SQM = 10.7639
ROOMS = {"Bedroom": (2, 12.0), "Bathroom": (2, 4.0), "Living room": (1, 20.0), "Kitchen": (1, 9.0)}
for r, (n, sz) in ROOMS.items():
    st.session_state.setdefault(f"rn_{r}", n)
    st.session_state.setdefault(f"rsz_{r}", sz)

def prep(file):
    from PIL import Image
    im = Image.open(file).convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()

def get_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        k = None
    return k or st.session_state.get("user_key")

def parse_json(text):
    return json.loads(text[text.index("{"): text.rindex("}") + 1])

def analyse_rooms():
    s = st.session_state
    items = [(r, f) for r in ROOMS if s.get(f"rmode_{r}") == "Upload photo" for f in (s.get(f"ph_{r}") or [])][:12]
    key = get_key()
    if not items:
        s.ai = {"error": "Choose 'Upload photo' for a room and upload a photo first."}
    elif not key:
        s.ai = {"error": "No API key. Paste your Gemini key in the sidebar, or type the sizes instead."}
    elif s.uses >= 10:
        s.ai = {"error": "Analysis limit reached for this session."}
    else:
        try:
            from google import genai
            from google.genai import types
            parts = []
            for r, f in items:
                parts += [f"Photo of a {r}:", types.Part.from_bytes(data=prep(f), mime_type="image/jpeg")]
            parts.append(PROMPT)
            try:
                first = st.secrets.get("GEMINI_MODEL")
            except Exception:
                first = None
            client, err, text = genai.Client(api_key=key), None, None
            for model in [m for m in (first, "gemini-2.5-flash", "gemini-flash-latest") if m]:
                try:
                    text = client.models.generate_content(
                        model=model, contents=parts,
                        config=types.GenerateContentConfig(response_mime_type="application/json")).text
                    break
                except Exception as e:
                    err = e
            if text is None:
                raise err
            res = parse_json(text)
            sizes = {r: [] for r in ROOMS}
            for (r, _), room in zip(items, res.get("rooms", [])):
                if room.get("area_sqm"):
                    sizes[r].append(max(1.0, min(200.0, float(room["area_sqm"]))))
            for r in ROOMS:
                files = s.get(f"ph_{r}") or []
                s[f"rpx_{r}"], s[f"rsig_{r}"] = sizes[r], [(f.name, f.size) for f in files]
            s.uses += 1
            s.ai = res
        except Exception as e:
            s.ai = {"error": f"Photo analysis failed: {e}"}

def photo_sizes(r):
    s, files = st.session_state, st.session_state.get(f"ph_{r}") or []
    ok = files and s.get(f"rsig_{r}") == [(f.name, f.size) for f in files]
    return s.get(f"rpx_{r}", []) if ok else []

def room_total(r):
    s, n = st.session_state, st.session_state[f"rn_{r}"]
    if s[f"rmode_{r}"] == "Upload photo" and photo_sizes(r):
        v = photo_sizes(r)[:n]
        return sum(v) + max(0, n - len(v)) * sum(v) / len(v)
    return n * s[f"rsz_{r}"]


def find_region():
    s = st.session_state
    text = s.get("addr", "").lower()
    hits = [r for r in REGIONS if r != "Other" and r.lower() in text]
    if not hits:
        low = {r.lower(): r for r in REGIONS if r != "Other"}
        words = [w.strip(",.") for w in text.replace("-", " ").split()]
        cand = [m for w in [text] + words for m in difflib.get_close_matches(w, low, n=1, cutoff=0.75)]
        hits = [low[c] for c in cand]
    if hits:
        s.f_region = max(hits, key=len)
        s.reg_msg = {"ok": f"Matched area: {s.f_region}. Change it below if this is wrong."}
    else:
        s.reg_msg = {"error": "No matching area in the data. Pick the nearest area from the list below."}

with st.sidebar:
    st.header("Photo analysis key (free)")
    st.caption("Only needed to read room sizes from photos. Kept for this session only.")
    st.text_input("Google Gemini API key", type="password", key="user_key")

st.title("🏠 Mumbai House Price Predictor")
st.caption("Linear regression trained on real property listings from the Mumbai region (asking prices).")
t1, t2, t3, t4, t5 = st.tabs(["🔮 Predict", "📊 Data Explorer", "📈 Model Performance", "💡 What drives the price", "ℹ️ About"])

with t1:
    st.subheader("1. House size")
    mode = st.radio("How do you want to give the size?", ["Type the total area", "Room by room (photo or size)"],
                    horizontal=True, key="size_mode")
    if mode.startswith("Type"):
        u1, u2 = st.columns(2)
        unit = u1.selectbox("Unit", ["square feet (sq ft)", "square metres (sq m)"], key="unit")
        metric = "metres" in unit
        total = u2.number_input("Total area", 10.0, 10000.0, 80.0 if metric else 860.0, step=5.0, key="total")
        area_sqft = total * SQM if metric else total
        bhk = st.number_input("BHK (number of bedrooms)", 1, 6, 2, key="bhk_t")
    else:
        st.caption("For each room type give how many there are, then type the size of one room or upload photos (one per room).")
        for r in ROOMS:
            a, b, c = st.columns([1, 1.3, 2])
            a.number_input(f"{r}s", 1 if r in ("Bedroom", "Bathroom") else 0, 6 if r == "Bedroom" else 10, key=f"rn_{r}")
            b.radio("Size from", ["Type size", "Upload photo"], key=f"rmode_{r}", horizontal=True)
            if st.session_state[f"rmode_{r}"] == "Upload photo":
                files = c.file_uploader(f"{r} photos (one per {r.lower()})", type=["jpg", "jpeg", "png"],
                                        accept_multiple_files=True, key=f"ph_{r}")
                v = photo_sizes(r)
                if files and v:
                    c.caption(f"Sizes read (sq m): {', '.join(f'{x:.1f}' for x in v)}. Rooms without a photo use the average.")
                elif files:
                    c.caption("Click 'Read sizes from photos' below.")
                if len(files) > st.session_state[f"rn_{r}"]:
                    c.warning(f"You uploaded {len(files)} photos but the count is {st.session_state[f'rn_{r}']}. Increase the count.")
            else:
                c.number_input(f"Size of one {r.lower()} (sq m)", 1.0, 200.0, step=0.5, key=f"rsz_{r}")
        st.button("🔍 Read sizes from photos", on_click=analyse_rooms)
        ai = st.session_state.get("ai")
        if ai and "error" in ai:
            st.error(ai["error"])
        elif ai:
            st.success("Photos analysed. The sizes read from the photos are used in the total. They are rough guesses, "
                       "so type the sizes instead if you know them.")
            if ai.get("rooms"):
                st.dataframe(pd.DataFrame(ai["rooms"]), hide_index=True)
            if ai.get("notes"):
                st.info(ai["notes"])
        area_sqft = sum(room_total(r) for r in ROOMS) * SQM
        bhk = st.session_state["rn_Bedroom"]
    st.info(f"Area used by the model: **{area_sqft:,.0f} sq ft** ({area_sqft / SQM:,.0f} sq m), **{bhk} BHK**")

    st.subheader("2. Location")
    lm = st.radio("How do you want to give the location?", ["Choose the area from the list", "Type the address"], horizontal=True)
    if lm.startswith("Type"):
        st.text_input("Address or area name (for example: Powai, Mumbai)", key="addr")
        st.button("📍 Find area", on_click=find_region)
        msg = st.session_state.get("reg_msg")
        if msg:
            (st.success if "ok" in msg else st.error)(msg.get("ok") or msg["error"])
    region = st.selectbox("Area / region (you can type to search)", REGIONS, key="f_region")

    st.subheader("3. Property details")
    p1, p2, p3 = st.columns(3)
    ptype = p1.selectbox("Property type", TYPES, key="f_type")
    status = p2.selectbox("Status", ["Ready to move", "Under Construction"], key="f_status")
    age = p3.selectbox("New or resale", ["New", "Resale", "Unknown"], key="f_age")

    st.subheader("4. Currency")
    g1, g2 = st.columns(2)
    cur = g1.radio("Show price in", ["Indian rupees (₹)", "US dollars ($)"])
    rate = g2.number_input("Today's rate: ₹ for $1", 1.0, 1000.0, 85.0, step=0.1,
                           help="Type the current dollar rate, for example from Google.")

    if st.button("Predict price", type="primary"):
        row = [dict(area=area_sqft, bhk=bhk, type=ptype, region=region, status=status, age=age)]
        lakh, m = run_model(row), load_metrics()
        usd = lakh * 1e5 / rate
        show = (lambda v: f"${v * 1e5 / rate:,.0f}") if "$" in cur else fmt_inr
        st.metric("Estimated price", show(lakh))
        lo, hi = lakh * m.get("ratio_p10", 0.75), lakh * m.get("ratio_p90", 1.3)
        st.caption((f"Likely range: {show(lo)} to {show(hi)}. In tests on houses the model had not seen, about 80% of "
                    f"predictions fell in a range like this. Also: {fmt_inr(lakh)} = ${usd:,.0f} at ₹{rate:g} per $1.").replace("$", "\\$"))
        sub = D[D["region"] == region]
        if len(sub) >= 20:
            ppsf = (sub["price_lakh"] * 1e5 / sub["area"]).median()
            st.caption(f"Check: this model's price is about ₹{lakh * 1e5 / area_sqft:,.0f} per sq ft. "
                       f"Listings in {region} have a median of ₹{ppsf:,.0f} per sq ft ({len(sub):,} listings).")
        if not D["area"].min() <= area_sqft <= D["area"].max():
            st.warning(f"Area {area_sqft:,.0f} sq ft is outside the training data ({D['area'].min():.0f} to {D['area'].max():.0f}), "
                       "so this estimate is less reliable.")
        if region == "Other":
            st.warning("'Other' groups many small areas, so the estimate is rough.")
        try:
            ex = explain(row[0])
            with st.expander("💡 Why this price? (compared with the average listing)", expanded=True):
                st.caption(f"The average listing in the data is about {fmt_inr(ex['baseline_lakh'])}. "
                           "Each bar shows how much that group of details moves your price up (+) or down (-).")
                st.bar_chart(pd.Series(ex["effects_pct"], name="% effect on price"), horizontal=True)
        except Exception as err:
            st.caption(f"(Explanation not available: {err})")

with t2:
    st.write(f"{D.shape[0]:,} real listings, {D.shape[1]} columns. Price is in lakh rupees (1 crore = 100 lakh).")
    st.dataframe(D.head(50))
    st.subheader("Summary statistics")
    st.dataframe(D.describe().T)
    st.subheader("Median price by area (the 15 areas with most listings)")
    top = D["region"].value_counts().head(15).index
    st.bar_chart(D[D["region"].isin(top)].groupby("region")["price_lakh"].median().sort_values())
    st.subheader("Area against price")
    st.scatter_chart(D.sample(min(3000, len(D)), random_state=1), x="area", y="price_lakh", color="bhk")

with t3:
    m = load_metrics()
    if m:
        a, b, c, e = st.columns(4)
        a.metric("Best model", m["best_model"].replace("poly2_ridge", "Polynomial + Ridge").title())
        b.metric("R²", f"{m['R2']:.3f}")
        c.metric("Typical error (RMSE)", f"{m['RMSE']:,.0f} lakh")
        e.metric("Average % error", f"{m['MAPE']:.0%}")
        st.caption(f"Trained on {m.get('n_train', 0):,} listings and tested on {m.get('n_test', 0):,} the model had not seen. "
                   f"Average absolute error (MAE): {m['MAE']:,.1f} lakh.")
    p = OUT / "model_comparison.csv"
    if p.exists():
        cmp = pd.read_csv(p)
        st.subheader("Model comparison")
        st.dataframe(cmp, hide_index=True)
        if "boosting_benchmark" in set(cmp["model"]):
            b = cmp[cmp["model"] == "boosting_benchmark"].iloc[0]
            st.caption(f"The 'boosting_benchmark' row is a non-linear model used only for comparison. It reaches R² {b['R2']:.3f} against "
                       f"{m.get('R2', 0):.3f} for the linear model, so the linear model gives up only a little accuracy and stays fully explainable.")
        st.bar_chart(cmp.set_index("model")["RMSE"])
    p = OUT / "error_by_group.csv"
    if p.exists():
        eg = pd.read_csv(p)
        eg["MAPE"] = (eg["MAPE"] * 100).round(1).astype(str) + "%"
        st.subheader("Where is the model weaker?")
        st.caption("Average % error for different kinds of houses on the test listings. Groups with few listings (small n) are unreliable.")
        st.dataframe(eg, hide_index=True)
    for f, cap in [(f"{m.get('best_model', 'ridge')}_diagnostics.png", "Actual vs predicted, residuals and Q-Q plot (log price)"),
                   ("learning_curve.png", "Learning curve: would more data help?"),
                   ("ridge_coefficients.png", "Which features push the price up or down (Ridge)"),
                   ("correlation_heatmap.png", "Correlation between numeric columns"),
                   ("distributions.png", "Distributions"),
                   ("gradient_descent_loss.png", "From-scratch gradient descent loss")]:
        if (OUT / f).exists():
            st.subheader(cap)
            st.image(str(OUT / f))

with t4:
    st.write("How each detail changes the price, measured on all real listings (log-linear regression with robust standard errors).")
    p = OUT / "price_drivers.csv"
    if p.exists():
        dr = pd.read_csv(p)
        a, b = dr.iloc[0], dr.iloc[1]
        st.info(f"Each extra 100 sq ft adds about {a.effect_pct:.1f}% to the price. One more BHK at the same area adds about {b.effect_pct:.0f}% "
                "(the rooms are smaller, but the flat is in a higher price class).")
        show = dr.rename(columns={"effect_pct": "price change %", "ci_low_pct": "95% low", "ci_high_pct": "95% high", "p_value": "p-value"})
        st.dataframe(show.round(2), hide_index=True)
    p = OUT / "region_effects.csv"
    if p.exists():
        rg = pd.read_csv(p)
        rg = rg[(rg["n"] >= 50) & (rg["region"] != "Other")]
        c1, c2 = st.columns(2)
        c1.subheader("Most expensive areas")
        c1.caption("Price change % compared with the average area (same size and type).")
        c1.bar_chart(rg.head(10).set_index("region")["effect_pct"], horizontal=True)
        c2.subheader("Least expensive areas")
        c2.bar_chart(rg.tail(10).set_index("region")["effect_pct"], horizontal=True)
    st.caption("These show associations in the listings, not proof that changing one detail would cause a price change.")

with t5:
    st.markdown("""
**Data:** about 76,000 property listings from the Mumbai region (a public dataset of asking prices from a property website).
Cleaning: repeated listings removed, prices converted to lakh rupees, areas outside 200 to 4,000 sq ft and BHK above 6 removed,
extreme prices per sq ft removed, and small areas grouped as "Other". About 55,000 listings remain.

**Model:** five linear models compared with 5-fold cross-validation (Linear, Ridge, Lasso, ElasticNet, Polynomial + Ridge).
Prices are modelled on a log scale, then converted back to rupees.

**Limits:** these are asking prices, not final sale prices, and the data has no floor, view, builder or exact address.
It covers the Mumbai region only. Photo sizes are rough guesses: a photo cannot measure a room.
""")
