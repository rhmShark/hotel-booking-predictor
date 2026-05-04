import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="🏨", layout="centered")

st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }
    .title    { font-size:1.7rem; font-weight:700; color:#1a3c5e; margin-bottom:0.1rem; }
    .subtitle { font-size:0.9rem; color:#777; margin-bottom:1.2rem; }
    .box-cancelled {
        background:#fff0f0; border:2px solid #e74c3c; color:#c0392b;
        border-radius:10px; padding:1rem; font-size:1.2rem;
        font-weight:700; text-align:center; margin-top:1rem;
    }
    .box-kept {
        background:#f0fff4; border:2px solid #27ae60; color:#1e8449;
        border-radius:10px; padding:1rem; font-size:1.2rem;
        font-weight:700; text-align:center; margin-top:1rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ML BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vt_threshold=0.01, corr_threshold=0.9, mi_threshold=0.01):
        self.vt_threshold   = vt_threshold
        self.corr_threshold = corr_threshold
        self.mi_threshold   = mi_threshold

    def fit(self, X, y):
        X = np.array(X)
        self.vt_ = VarianceThreshold(self.vt_threshold)
        X_vt = self.vt_.fit_transform(X)
        corr = np.corrcoef(X_vt, rowvar=False)
        self.to_drop_ = set()
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                if abs(corr[i, j]) > self.corr_threshold:
                    self.to_drop_.add(j)
        self.keep_mask_ = np.array([i not in self.to_drop_ for i in range(X_vt.shape[1])])
        X_corr = X_vt[:, self.keep_mask_]
        mi = mutual_info_classif(X_corr, y, random_state=42)
        self.mi_mask_ = mi > self.mi_threshold
        return self

    def transform(self, X):
        X    = np.array(X)
        X_vt = self.vt_.transform(X)
        X_c  = X_vt[:, self.keep_mask_]
        return X_c[:, self.mi_mask_]


def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # String cleaning — cast to str first to avoid AttributeError on mixed types
    for col in ['meal', 'market_segment', 'distribution_channel', 'hotel',
                'deposit_type', 'customer_type', 'reserved_room_type',
                'assigned_room_type', 'arrival_date_month']:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip()

    if 'country' in X.columns:
        X['country'] = X['country'].astype(str).str.upper().str.strip()

    # Drop leaky / high-cardinality columns
    for col in ['agent', 'company', 'reservation_status']:
        if col in X.columns:
            X = X.drop(col, axis=1)

    # Fix children / babies / adults: NaN → 0 before any arithmetic
    for col in ['children', 'babies', 'adults']:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).clip(lower=0)

    # Drop rows where total guests == 0
    adults_s   = X['adults']   if 'adults'   in X.columns else pd.Series(1, index=X.index)
    children_s = X['children'] if 'children' in X.columns else pd.Series(0, index=X.index)
    babies_s   = X['babies']   if 'babies'   in X.columns else pd.Series(0, index=X.index)
    X = X[~((adults_s == 0) & (children_s == 0) & (babies_s == 0))].copy()

    # ADR: coerce → replace inf → fillna median → clip
    if 'adr' in X.columns:
        X['adr'] = pd.to_numeric(X['adr'], errors='coerce')
        X['adr'] = X['adr'].replace([np.inf, -np.inf], np.nan)
        med = X['adr'].median()
        X['adr'] = X['adr'].fillna(med if not np.isnan(med) else 0).clip(0, 5000)

    # Safe int conversion for children (NaN must already be 0)
    if 'children' in X.columns:
        X['children'] = X['children'].fillna(0).astype(int)

    # reservation_status_date → numeric features
    if 'reservation_status_date' in X.columns:
        rsd = pd.to_datetime(X['reservation_status_date'], errors='coerce')
        X['year']      = rsd.dt.year.fillna(0).astype(int)
        X['month_num'] = rsd.dt.month.fillna(0).astype(int)
        X['day']       = rsd.dt.day.fillna(0).astype(int)
        X = X.drop('reservation_status_date', axis=1)

    # Engineered features
    X['total_people'] = (
        X.get('adults',   pd.Series(0, index=X.index)) +
        X.get('children', pd.Series(0, index=X.index)) +
        X.get('babies',   pd.Series(0, index=X.index))
    )
    X['total_nights'] = (
        X.get('stays_in_weekend_nights', pd.Series(0, index=X.index)) +
        X.get('stays_in_week_nights',    pd.Series(0, index=X.index))
    )

    # Final safety: replace any leftover inf (sklearn will error on inf)
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    return X


MODEL_PATH = "hotel_svm_pipeline.pkl"


@st.cache_resource(show_spinner=False)
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)

    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = clean_data(X_train)
    y_train = y_train.loc[X_train.index]

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selector",     FeatureSelector()),
        ("pca",          PCA(n_components=0.95, svd_solver="full")),
        ("svm",          SVC(C=1, kernel="linear", class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


def load_or_train(csv_path=None):
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    if csv_path:
        return train_model(csv_path)
    return None


def make_prediction(pipeline, raw: dict):
    row  = pd.DataFrame([raw])
    row  = clean_data(row)
    pred = pipeline.predict(row)[0]
    try:
        score = pipeline.decision_function(row)[0]
        conf  = round(min(abs(score) / (abs(score) + 1) * 100, 99.9), 1)
    except Exception:
        conf = None
    return pred, conf


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="title">🏨 Hotel Booking Cancellation Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fill in the booking details and click Predict.</p>', unsafe_allow_html=True)

# ── Sidebar: model training ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Setup")
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model ready")
        if st.button("Retrain model"):
            os.remove(MODEL_PATH)
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("Upload **hotel_bookings.csv** to train the model.")
        uploaded = st.file_uploader("hotel_bookings.csv", type="csv")
        if uploaded:
            tmp = os.path.join(tempfile.gettempdir(), "hotel_bookings.csv")
            with open(tmp, "wb") as f:
                f.write(uploaded.read())
            with st.spinner("Training… this may take a few minutes"):
                try:
                    load_or_train(tmp)
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.exception(e)

pipeline = load_or_train()
if pipeline is None:
    st.warning("⬅️ Please upload the dataset in the sidebar to train the model.")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### Booking Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-hdr">🏩 Hotel & Stay</p>', unsafe_allow_html=True)
    hotel           = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    lead_time       = st.number_input("Lead Time (days before arrival)", 0, 1000, 30)
    stays_weekend   = st.number_input("Weekend Nights", 0, 20, 1)
    stays_week      = st.number_input("Week Nights", 0, 50, 2)
    meal            = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC", "Undefined"])
    adr             = st.number_input("Average Daily Rate (ADR) $", 0.0, 5000.0, 90.0, step=5.0)

with col2:
    st.markdown('<p class="section-hdr">👤 Guest Info</p>', unsafe_allow_html=True)
    adults          = st.number_input("Adults", 1, 20, 2)
    children        = st.number_input("Children", 0, 10, 0)
    babies          = st.number_input("Babies", 0, 10, 0)
    country         = st.text_input("Country Code (e.g. PRT, USA, EGY)", "PRT").upper()
    customer_type   = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
    is_repeated     = st.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col3:
    st.markdown('<p class="section-hdr">📅 Booking Info</p>', unsafe_allow_html=True)
    market_segment  = st.selectbox("Market Segment",
                                   ["Direct", "Corporate", "Online TA", "Offline TA/TO",
                                    "Complementary", "Groups", "Aviation", "Undefined"])
    dist_channel    = st.selectbox("Distribution Channel",
                                   ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
    deposit_type    = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    reserved_room   = st.selectbox("Reserved Room Type", list("ABCDEFGHI"))
    assigned_room   = st.selectbox("Assigned Room Type", list("ABCDEFGHIK"))
    booking_changes = st.number_input("Booking Changes", 0, 20, 0)

col4, col5 = st.columns(2)
with col4:
    st.markdown('<p class="section-hdr">📆 Arrival Date</p>', unsafe_allow_html=True)
    arrival_year    = st.selectbox("Arrival Year", [2024, 2025, 2026], index=1)
    arrival_month   = st.selectbox("Arrival Month",
                                   ["January","February","March","April","May","June",
                                    "July","August","September","October","November","December"],
                                   index=4)
    arrival_week    = st.number_input("Arrival Week Number", 1, 53, 20)
    arrival_day     = st.number_input("Day of Month", 1, 31, 15)

with col5:
    st.markdown('<p class="section-hdr">📝 History & Requests</p>', unsafe_allow_html=True)
    prev_cancels    = st.number_input("Previous Cancellations", 0, 30, 0)
    prev_bookings   = st.number_input("Previous Bookings Not Cancelled", 0, 100, 0)
    special_requests= st.number_input("Special Requests", 0, 10, 0)
    days_waiting    = st.number_input("Days in Waiting List", 0, 500, 0)
    required_parking= st.number_input("Required Car Parking Spaces", 0, 8, 0)
    total_of_special= st.number_input("Total Special Requests", 0, 10, 0)

# ── Build raw dict ────────────────────────────────────────────────────────────
raw_input = {
    "hotel":                          hotel,
    "lead_time":                      lead_time,
    "arrival_date_year":              arrival_year,
    "arrival_date_month":             arrival_month,
    "arrival_date_week_number":       arrival_week,
    "arrival_date_day_of_month":      arrival_day,
    "stays_in_weekend_nights":        stays_weekend,
    "stays_in_week_nights":           stays_week,
    "adults":                         adults,
    "children":                       float(children),
    "babies":                         babies,
    "meal":                           meal,
    "country":                        country,
    "market_segment":                 market_segment,
    "distribution_channel":           dist_channel,
    "is_repeated_guest":              is_repeated,
    "previous_cancellations":         prev_cancels,
    "previous_bookings_not_canceled": prev_bookings,
    "reserved_room_type":             reserved_room,
    "assigned_room_type":             assigned_room,
    "booking_changes":                booking_changes,
    "deposit_type":                   deposit_type,
    "days_in_waiting_list":           days_waiting,
    "customer_type":                  customer_type,
    "adr":                            adr,
    "required_car_parking_spaces":    required_parking,
    "total_of_special_requests":      total_of_special,
}

try:
    with st.spinner("Predicting…"):
        result, conf = make_prediction(pipeline, raw)
        if result == 1:
            st.markdown('<div class="box-cancelled">❌ Cancelled</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-kept">✅ Not Cancelled</div>', unsafe_allow_html=True)

        if conf is not None:
            st.progress(int(conf), text=f"Confidence: {conf}%")

except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)
'''
# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Predict Cancellation", use_container_width=True, type="primary")

if predict_btn:
    try:
        with st.spinner("Running SVM prediction…"):
            result, confidence = predict(pipeline, raw_input)

        if result == 1:
            label = "⚠️ Booking Likely to be CANCELLED"
            css   = "cancelled"
        else:
            label = "✅ Booking Likely to be KEPT"
            css   = "not-cancelled"

        st.markdown(f'<div class="result-box {css}">{label}</div>', unsafe_allow_html=True)

        if confidence is not None:
            st.progress(int(confidence), text=f"Model confidence: {confidence}%")

        with st.expander("📊 Input Summary"):
            summary = pd.DataFrame([raw_input]).T
            summary.columns = ["Value"]
            st.dataframe(summary, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SVM pipeline · Preprocessing → Variance filter → Correlation filter → MI filter → PCA → SVC (linear, C=1)")

'''
