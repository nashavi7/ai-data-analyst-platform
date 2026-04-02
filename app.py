import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Glassmorphism UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Cards / Glass panels */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 40px rgba(99,179,237,0.12);
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(99,179,237,0.06) 0%, transparent 60%);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #63b3ed;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(159,122,234,0.06));
    border: 1px solid rgba(99,179,237,0.2);
    border-left: 4px solid #63b3ed;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.7;
}
.insight-title {
    font-weight: 600;
    color: #63b3ed;
    margin-bottom: 6px;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Warning / alert boxes */
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #fcd34d;
    font-size: 0.88rem;
}
.good-box {
    background: rgba(52,211,153,0.06);
    border: 1px solid rgba(52,211,153,0.2);
    border-left: 4px solid #34d399;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #6ee7b7;
    font-size: 0.88rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-blue { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-purple { background: rgba(159,122,234,0.15); color: #a78bfa; border: 1px solid rgba(159,122,234,0.3); }
.badge-green { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.badge-red { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.badge-yellow { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

/* Tab override */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(99,179,237,0.15) !important;
    color: #63b3ed !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }

/* Plotly chart background */
.js-plotly-plot .plotly .svg-container { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ENGINE 1: DATA PROFILER
# ─────────────────────────────────────────────
def profile_data(df):
    """Return a comprehensive profile of the dataset."""
    profile = {}
    profile["rows"] = len(df)
    profile["cols"] = len(df.columns)
    profile["missing_cells"] = int(df.isnull().sum().sum())
    profile["missing_pct"] = round(df.isnull().mean().mean() * 100, 2)
    profile["duplicates"] = int(df.duplicated().sum())
    profile["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    profile["num_cols"] = num_cols
    profile["cat_cols"] = cat_cols
    profile["num_count"] = len(num_cols)
    profile["cat_count"] = len(cat_cols)

    # Per-column stats
    col_stats = []
    for col in df.columns:
        s = {"name": col, "dtype": str(df[col].dtype), "missing": int(df[col].isnull().sum()),
             "unique": int(df[col].nunique())}
        if col in num_cols:
            s["mean"] = round(df[col].mean(), 4)
            s["std"] = round(df[col].std(), 4)
            s["min"] = round(df[col].min(), 4)
            s["max"] = round(df[col].max(), 4)
            s["skew"] = round(df[col].skew(), 3)
            s["kurt"] = round(df[col].kurtosis(), 3)
            cv = (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0
            s["cv"] = round(abs(cv), 2)
        col_stats.append(s)
    profile["col_stats"] = col_stats
    return profile


# ─────────────────────────────────────────────
# ENGINE 2: KPI DETECTOR
# ─────────────────────────────────────────────
KPI_KEYWORDS = [
    "revenue", "profit", "sales", "churn", "conversion", "retention",
    "quality", "score", "rate", "yield", "cost", "output", "efficiency",
    "satisfaction", "nps", "kpi", "metric", "target", "performance",
    "amount", "price", "income", "expense", "margin", "growth", "count",
    "total", "volume", "loss", "gain", "return", "risk", "error", "defect",
]

def detect_kpis(df, profile):
    """Identify KPI columns using naming conventions + variance analysis."""
    num_cols = profile["num_cols"]
    if not num_cols:
        return []

    scores = {}
    for col in num_cols:
        score = 0
        cl = col.lower().replace("_", " ").replace("-", " ")
        for kw in KPI_KEYWORDS:
            if kw in cl:
                score += 3

        # High variance / spread → important metric
        cv = next((s["cv"] for s in profile["col_stats"] if s["name"] == col), 0)
        if cv > 50:
            score += 2
        elif cv > 25:
            score += 1

        # Unique ratio — not an ID column
        uniq_ratio = df[col].nunique() / len(df)
        if 0.05 < uniq_ratio < 0.95:
            score += 1

        scores[col] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Return top-5 or those scoring > 0
    kpis = [col for col, s in ranked if s > 0][:5]
    if not kpis:
        kpis = [ranked[0][0]] if ranked else []
    return kpis


# ─────────────────────────────────────────────
# ENGINE 3: TARGET SELECTOR
# ─────────────────────────────────────────────
def suggest_target(df, profile):
    """Heuristically suggest the best target column."""
    kpi_cols = detect_kpis(df, profile)
    if kpi_cols:
        return kpi_cols[0]
    num_cols = profile["num_cols"]
    if num_cols:
        return num_cols[-1]
    return None


# ─────────────────────────────────────────────
# ENGINE 4: CORRELATION ENGINE
# ─────────────────────────────────────────────
def compute_correlations(df, target, profile):
    """Compute correlation between numeric features and the target."""
    num_cols = [c for c in profile["num_cols"] if c != target]
    if not num_cols:
        return {}
    corr_map = {}
    for col in num_cols:
        valid = df[[col, target]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[col], valid[target])
        corr_map[col] = {"r": round(r, 4), "p": round(p, 4), "abs_r": abs(r)}
    return dict(sorted(corr_map.items(), key=lambda x: x[1]["abs_r"], reverse=True))


# ─────────────────────────────────────────────
# ENGINE 5: OUTLIER DETECTOR
# ─────────────────────────────────────────────
def detect_outliers(df, profile):
    """IQR-based outlier detection for all numeric columns."""
    results = {}
    for col in profile["num_cols"]:
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((series < lower) | (series > upper)).sum()
        pct = round(outliers / len(series) * 100, 2)
        results[col] = {
            "count": int(outliers),
            "pct": pct,
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "severity": "High" if pct > 10 else ("Moderate" if pct > 3 else "Low"),
        }
    return results


# ─────────────────────────────────────────────
# ENGINE 6: AUTOML ENGINE
# ─────────────────────────────────────────────
def run_automl(df, target, profile):
    """Train a Random Forest and return performance + feature importances."""
    num_cols = [c for c in profile["num_cols"] if c != target]
    if not num_cols or target not in df.columns:
        return None

    X = df[num_cols].dropna()
    y = df.loc[X.index, target]
    if y.isnull().sum() > 0:
        mask = y.notna()
        X, y = X[mask], y[mask]
    if len(X) < 20:
        return None

    # Determine task type
    n_unique = y.nunique()
    task = "classification" if (n_unique <= 15 and n_unique / len(y) < 0.05) else "regression"

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task == "classification":
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train.astype(str))
            y_test_enc = le.transform(y_test.astype(str))
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train_enc)
            score = round(accuracy_score(y_test_enc, model.predict(X_test)) * 100, 2)
            metric_name = "Accuracy"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            score = round(r2_score(y_test, model.predict(X_test)) * 100, 2)
            metric_name = "R² Score"

        importances = dict(zip(num_cols, model.feature_importances_.tolist()))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        return {
            "task": task,
            "score": score,
            "metric": metric_name,
            "importances": importances,
            "features": num_cols,
            "target": target,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# ENGINE 7: NARRATIVE SYNTHESIZER
# ─────────────────────────────────────────────
def generate_narrative(df, profile, kpis, corr_map, outliers, automl_result, target):
    """Generate human-readable AI-style narrative insights."""
    insights = []

    # --- Dataset overview ---
    q = profile["missing_pct"]
    dup = profile["duplicates"]
    overview = (
        f"Your dataset contains **{profile['rows']:,} records** across **{profile['cols']} features** "
        f"({profile['num_count']} numeric, {profile['cat_count']} categorical). "
    )
    if q < 1:
        overview += "Data completeness is excellent — virtually no missing values detected. ✅"
    elif q < 5:
        overview += f"There is a minor {q}% missing data rate which is generally acceptable. ⚠️"
    else:
        overview += f"**{q}% of values are missing** — data imputation is strongly recommended before modelling. 🚨"
    if dup > 0:
        overview += f" Additionally, **{dup} duplicate rows** were found and may skew results."
    insights.append(("📊 Dataset Overview", overview))

    # --- KPI insight ---
    if kpis:
        kpi_str = ", ".join([f"`{k}`" for k in kpis])
        insights.append(("🎯 Key Performance Indicators",
            f"The system identified **{len(kpis)} KPI column(s)**: {kpi_str}. "
            f"These columns exhibit high business relevance based on naming conventions and statistical variance. "
            f"Focus your analysis and reporting on these metrics for maximum impact."))

    # --- Target & correlation insight ---
    if target and corr_map:
        top_corr = list(corr_map.items())[:3]
        strong = [(c, v) for c, v in top_corr if v["abs_r"] >= 0.5]
        moderate = [(c, v) for c, v in top_corr if 0.3 <= v["abs_r"] < 0.5]

        txt = f"Analyzing relationships with target **`{target}`**: "
        if strong:
            parts = [f"`{c}` (r={v['r']:+.2f})" for c, v in strong]
            txt += f"**Strong correlations** found with {', '.join(parts)}. These are primary drivers. "
        if moderate:
            parts = [f"`{c}` (r={v['r']:+.2f})" for c, v in moderate]
            txt += f"**Moderate correlations** found with {', '.join(parts)}. "
        if not strong and not moderate:
            txt += "No strong linear correlations detected — consider non-linear models or feature engineering."
        insights.append(("🔗 Relationship Discovery", txt))

    # --- Outlier insight ---
    if outliers:
        high_outliers = {c: v for c, v in outliers.items() if v["severity"] == "High"}
        mod_outliers = {c: v for c, v in outliers.items() if v["severity"] == "Moderate"}
        txt = ""
        if high_outliers:
            cols = ", ".join([f"`{c}` ({v['pct']}%)" for c, v in high_outliers.items()])
            txt += f"🚨 **High outlier rate** detected in {cols}. These may represent data errors or extreme business events. "
        if mod_outliers:
            cols = ", ".join([f"`{c}`" for c in mod_outliers])
            txt += f"⚠️ Moderate outliers in {cols} — review before modelling."
        if not txt:
            txt = "✅ Outlier levels are within acceptable ranges across all numeric features."
        insights.append(("🚨 Data Quality Assessment", txt))

    # --- AutoML insight ---
    if automl_result and "error" not in automl_result:
        s = automl_result["score"]
        metric = automl_result["metric"]
        task = automl_result["task"]
        top_feat = list(automl_result["importances"].items())[:3]
        feat_str = ", ".join([f"`{f}` ({round(imp*100,1)}%)" for f, imp in top_feat])

        if task == "classification":
            perf = "excellent" if s >= 85 else ("good" if s >= 70 else "moderate")
        else:
            perf = "excellent" if s >= 80 else ("good" if s >= 60 else "moderate")

        txt = (
            f"A **Random Forest model** was automatically trained to predict `{target}` "
            f"({task}) and achieved a **{metric} of {s}%** — {perf} performance. "
            f"The most influential features are: {feat_str}. "
            f"These features should be prioritised in business decisions and future data collection."
        )
        insights.append(("🤖 AutoML Prediction Engine", txt))

    # --- Final recommendation ---
    recs = []
    if profile["missing_pct"] > 5:
        recs.append("Address missing data via imputation or removal")
    if profile["duplicates"] > 0:
        recs.append("Remove duplicate records before analysis")
    h_out = [c for c, v in outliers.items() if v["severity"] == "High"] if outliers else []
    if h_out:
        recs.append(f"Investigate outliers in: {', '.join(h_out)}")
    if automl_result and "score" in automl_result and automl_result["score"] < 60:
        recs.append("Consider feature engineering or additional data sources to improve model accuracy")

    if recs:
        txt = "**Next steps recommended:**\n" + "\n".join([f"- {r}" for r in recs])
    else:
        txt = "✅ The dataset is in good shape. You can proceed confidently with reporting, modelling, and business decisions."
    insights.append(("💡 Recommendations", txt))

    return insights


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="#94a3b8"),
    margin=dict(l=20, r=20, t=40, b=20),
)

def _apply_theme(fig):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(**PLOTLY_THEME)
    return fig

def chart_distribution(df, col):
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#63b3ed"],
                       title=f"Distribution of {col}")
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.1)")
    return _apply_theme(fig)

def chart_box(df, col):
    fig = px.box(df, y=col, color_discrete_sequence=["#9f7aea"],
                 title=f"Box Plot — {col}")
    return _apply_theme(fig)

def chart_correlation_heatmap(df, num_cols):
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Feature Correlation Matrix", **PLOTLY_THEME)
    return fig

def chart_feature_importance(importances):
    df_imp = pd.DataFrame({"Feature": list(importances.keys()),
                           "Importance": [v*100 for v in importances.values()]})
    fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#9f7aea", "#63b3ed", "#34d399"],
                 title="Feature Importance (%)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return _apply_theme(fig)

def chart_scatter(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                     color_discrete_sequence=["#63b3ed"],
                     trendline_color_override="#f687b3",
                     title=f"{x_col} vs {y_col}")
    return _apply_theme(fig)

def chart_outlier_summary(outliers):
    df_out = pd.DataFrame([
        {"Feature": c, "Outlier %": v["pct"], "Severity": v["severity"]}
        for c, v in outliers.items() if v["count"] > 0
    ])
    if df_out.empty:
        return None
    color_map = {"High": "#f87171", "Moderate": "#fbbf24", "Low": "#34d399"}
    fig = px.bar(df_out, x="Feature", y="Outlier %", color="Severity",
                 color_discrete_map=color_map, title="Outlier Rate by Feature (%)")
    return _apply_theme(fig)

def chart_missing(profile):
    rows = [{"Feature": s["name"], "Missing %": round(s["missing"] / profile["rows"] * 100, 2)}
            for s in profile["col_stats"] if s["missing"] > 0]
    if not rows:
        return None
    df_m = pd.DataFrame(rows).sort_values("Missing %", ascending=False)
    fig = px.bar(df_m, x="Feature", y="Missing %", color="Missing %",
                 color_continuous_scale=["#34d399", "#f59e0b", "#f87171"],
                 title="Missing Value Rate by Feature")
    fig.update_layout(coloraxis_showscale=False)
    return _apply_theme(fig)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px'>
        <div style='font-size:2.4rem'>🧠</div>
        <div style='font-size:1.1rem; font-weight:700; color:#63b3ed; margin:4px 0'>Offline AI Analyst</div>
        <div style='font-size:0.78rem; color:#64748b'>100% Local · No APIs · Private</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📂 Upload Dataset**")
    uploaded_file = st.file_uploader("Drop a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        sep = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)

    st.markdown("---")
    st.markdown("**⚙️ Analysis Settings**")
    max_cat_unique = st.slider("Max unique values for categorical", 5, 50, 20)
    outlier_iqr_mult = st.slider("Outlier IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
    automl_enabled = st.checkbox("Enable AutoML", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; padding:4px 0; text-align:center'>
        Built with Scikit-Learn · Plotly · Streamlit<br>
        <span style='color:#34d399'>■</span> No internet required
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <div class='hero-title'>🧠 Offline AI Analyst</div>
    <p class='hero-subtitle'>
        Upload any CSV → Get instant KPI detection, correlation analysis, AutoML predictions,
        and human-readable business insights — completely offline.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
if uploaded_file is None:
    # Landing state
    col1, col2, col3 = st.columns(3)
    features = [
        ("📊", "Instant Understanding", "Dataset overview, statistics, and distributions in seconds."),
        ("🎯", "KPI Detection", "Automatically identifies your most important business metrics."),
        ("🔗", "Correlation Engine", "Discovers what features drive your outcomes."),
        ("🚨", "Data Quality", "Detects outliers, missing data, and noise automatically."),
        ("🤖", "AutoML Predictions", "Trains a model and generates feature importance."),
        ("🧠", "Smart Insights", "Human-readable, ChatGPT-style business narratives."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='font-size:2rem;margin-bottom:10px'>{icon}</div>
                <div style='font-weight:700;color:#e2e8f0;margin-bottom:8px'>{title}</div>
                <div style='font-size:0.85rem;color:#64748b'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.info("👈 Upload a CSV from the sidebar to begin your analysis.")
    st.stop()

# ─── Load Data ───
try:
    df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# ─── Run all engines ───
with st.spinner("🔍 Analysing your data..."):
    profile = profile_data(df)
    kpis = detect_kpis(df, profile)
    target = suggest_target(df, profile)
    corr_map = compute_correlations(df, target, profile) if target and target in profile["num_cols"] else {}
    outliers = detect_outliers(df, profile)
    automl_result = run_automl(df, target, profile) if automl_enabled and target else None

# Allow user to override target
if profile["num_cols"]:
    target = st.selectbox(
        "🎯 Analysis Target (editable)",
        profile["num_cols"],
        index=profile["num_cols"].index(target) if target in profile["num_cols"] else 0,
        key="target_sel",
    )
    corr_map = compute_correlations(df, target, profile)
    if automl_enabled:
        automl_result = run_automl(df, target, profile)

narrative = generate_narrative(df, profile, kpis, corr_map, outliers, automl_result, target)

# ─────────────────────────────────────────────
# TOP METRICS BAR
# ─────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
metrics = [
    (m1, str(f"{profile['rows']:,}"), "Rows"),
    (m2, str(profile['cols']), "Features"),
    (m3, str(profile['num_count']), "Numeric"),
    (m4, str(profile['cat_count']), "Categorical"),
    (m5, f"{profile['missing_pct']}%", "Missing"),
    (m6, str(profile['duplicates']), "Duplicates"),
]
for col, val, label in metrics:
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{val}</div>
        <div class='metric-label'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Understanding",
    "🎯 KPIs",
    "🔗 Relationships",
    "🚨 Data Health",
    "🤖 AutoML",
    "🧠 Insights",
])

# ──────────────────────────────────────────────
# TAB 1: INSTANT UNDERSTANDING
# ──────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>📊 Instant Data Understanding</div>", unsafe_allow_html=True)

    # Summary table
    rows = []
    for s in profile["col_stats"]:
        badge_type = "badge-blue" if s["name"] in profile["num_cols"] else "badge-purple"
        badge_label = "Numeric" if s["name"] in profile["num_cols"] else "Categorical"
        rows.append({
            "Column": s["name"],
            "Type": badge_label,
            "Missing": f"{s['missing']} ({round(s['missing']/profile['rows']*100,1)}%)",
            "Unique": s["unique"],
            "Mean / Mode": str(round(s.get("mean", ""), 4)) if "mean" in s else "—",
            "Std Dev": str(round(s.get("std", ""), 4)) if "std" in s else "—",
            "Skewness": str(s.get("skew", "—")),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Distribution charts
    st.markdown("<div class='section-header'>📈 Feature Distributions</div>", unsafe_allow_html=True)
    num_cols = profile["num_cols"]
    if num_cols:
        sel_col = st.selectbox("Select feature to visualise", num_cols, key="dist_col")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(chart_distribution(df, sel_col), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_box(df, sel_col), use_container_width=True)

    # Missing value chart
    miss_fig = chart_missing(profile)
    if miss_fig:
        st.plotly_chart(miss_fig, use_container_width=True)
    else:
        st.markdown("<div class='good-box'>✅ No missing values detected in any column.</div>", unsafe_allow_html=True)

    # Raw data preview
    with st.expander("🔎 Preview Raw Data"):
        st.dataframe(df.head(50), use_container_width=True)


# ──────────────────────────────────────────────
# TAB 2: KPIs
# ──────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>🎯 KPI Identification</div>", unsafe_allow_html=True)

    if kpis:
        badges = " ".join([f"<span class='badge badge-blue'>📌 {k}</span>" for k in kpis])
        st.markdown(f"<div class='insight-box'><div class='insight-title'>Detected KPIs</div>{badges}</div>",
                    unsafe_allow_html=True)

        # KPI stats cards
        cols_kpi = st.columns(min(len(kpis), 3))
        for i, kpi_col in enumerate(kpis[:3]):
            s = df[kpi_col].dropna()
            with cols_kpi[i]:
                st.markdown(f"""
                <div class='glass-card'>
                    <div class='insight-title'>{kpi_col}</div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;'>
                        <div><div style='color:#63b3ed;font-weight:600'>{round(s.mean(),2)}</div><div style='color:#64748b;font-size:0.75rem'>Mean</div></div>
                        <div><div style='color:#9f7aea;font-weight:600'>{round(s.std(),2)}</div><div style='color:#64748b;font-size:0.75rem'>Std Dev</div></div>
                        <div><div style='color:#34d399;font-weight:600'>{round(s.min(),2)}</div><div style='color:#64748b;font-size:0.75rem'>Min</div></div>
                        <div><div style='color:#f87171;font-weight:600'>{round(s.max(),2)}</div><div style='color:#64748b;font-size:0.75rem'>Max</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(chart_distribution(df, kpi_col), use_container_width=True, key=f"kpi_dist_{i}")
    else:
        st.info("No KPI columns were detected based on naming and variance analysis. Try renaming your key columns with business terms.")

    # Coefficient of variation ranking
    st.markdown("<div class='section-header'>📐 Feature Variance Ranking</div>", unsafe_allow_html=True)
    cv_data = sorted(
        [(s["name"], s.get("cv", 0)) for s in profile["col_stats"] if "cv" in s],
        key=lambda x: x[1], reverse=True
    )
    if cv_data:
        df_cv = pd.DataFrame(cv_data, columns=["Feature", "CV (%)"])
        fig_cv = px.bar(df_cv, x="CV (%)", y="Feature", orientation="h",
                        color="CV (%)", color_continuous_scale=["#4f46e5", "#63b3ed", "#34d399"],
                        title="Coefficient of Variation (Higher = More Variable)")
        fig_cv.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        _apply_theme(fig_cv)
        st.plotly_chart(fig_cv, use_container_width=True)


# ──────────────────────────────────────────────
# TAB 3: RELATIONSHIPS
# ──────────────────────────────────────────────
with tabs[2]:
    st.markdown(f"<div class='section-header'>🔗 Relationship Discovery — Target: <code style='color:#63b3ed'>{target}</code></div>",
                unsafe_allow_html=True)

    if corr_map:
        # Correlation bar chart
        df_corr = pd.DataFrame([
            {"Feature": c, "Correlation": v["r"], "Strength": abs(v["r"])}
            for c, v in corr_map.items()
        ]).sort_values("Strength", ascending=False)

        fig_corr = px.bar(df_corr, x="Correlation", y="Feature", orientation="h",
                          color="Correlation", color_continuous_scale="RdBu",
                          color_continuous_midpoint=0,
                          title=f"Pearson Correlation with '{target}'")
        fig_corr.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=True)
        _apply_theme(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Correlation breakdown
        strong = [(c, v) for c, v in corr_map.items() if v["abs_r"] >= 0.5]
        moderate = [(c, v) for c, v in corr_map.items() if 0.3 <= v["abs_r"] < 0.5]
        weak = [(c, v) for c, v in corr_map.items() if v["abs_r"] < 0.3]

        if strong:
            st.markdown("**💪 Strong Correlations**")
            for c, v in strong:
                direction = "positively" if v["r"] > 0 else "negatively"
                st.markdown(f"""
                <div class='insight-box'>
                    <span class='badge badge-{'green' if v['r']>0 else 'red'}'>{'+' if v['r']>0 else ''}{v['r']}</span>
                    <strong>{c}</strong> is <em>{direction}</em> correlated with <strong>{target}</strong>.
                    As {c} increases, {target} tends to {'increase' if v['r']>0 else 'decrease'}.
                    (p-value: {v['p']})
                </div>
                """, unsafe_allow_html=True)

        # Scatter plot explorer
        st.markdown("<div class='section-header'>🔭 Scatter Explorer</div>", unsafe_allow_html=True)
        other_num = [c for c in profile["num_cols"] if c != target]
        if other_num:
            scatter_x = st.selectbox("X-axis feature", other_num, key="scatter_x")
            st.plotly_chart(chart_scatter(df, scatter_x, target), use_container_width=True)

    # Full correlation heatmap
    if len(profile["num_cols"]) >= 2:
        st.markdown("<div class='section-header'>🗺️ Full Correlation Heatmap</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_correlation_heatmap(df, profile["num_cols"]), use_container_width=True)


# ──────────────────────────────────────────────
# TAB 4: DATA HEALTH
# ──────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-header'>🚨 Data Quality & Health</div>", unsafe_allow_html=True)

    # Overall health score
    issues = 0
    if profile["missing_pct"] > 5: issues += 2
    elif profile["missing_pct"] > 1: issues += 1
    if profile["duplicates"] > 0: issues += 1
    high_out = sum(1 for v in outliers.values() if v["severity"] == "High")
    issues += high_out
    health_score = max(0, 100 - issues * 15)
    health_color = "#34d399" if health_score >= 80 else ("#f59e0b" if health_score >= 60 else "#f87171")

    st.markdown(f"""
    <div class='glass-card' style='text-align:center;'>
        <div style='font-size:0.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>Overall Data Health Score</div>
        <div style='font-size:3.5rem;font-weight:800;color:{health_color}'>{health_score}</div>
        <div style='color:#64748b;font-size:0.85rem'>out of 100</div>
    </div>
    """, unsafe_allow_html=True)

    # Outlier summary chart
    out_fig = chart_outlier_summary(outliers)
    if out_fig:
        st.plotly_chart(out_fig, use_container_width=True)

    # Outlier table
    st.markdown("<div class='section-header'>📋 Outlier Details</div>", unsafe_allow_html=True)
    out_rows = []
    for col, v in outliers.items():
        sev_badge = {"High": "badge-red", "Moderate": "badge-yellow", "Low": "badge-green"}[v["severity"]]
        out_rows.append({
            "Feature": col,
            "Outliers": v["count"],
            "Outlier %": f"{v['pct']}%",
            "Lower Fence": v["lower"],
            "Upper Fence": v["upper"],
            "Severity": v["severity"],
        })
    st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    # Skewness analysis
    st.markdown("<div class='section-header'>📐 Skewness Analysis</div>", unsafe_allow_html=True)
    skew_data = [(s["name"], s.get("skew", 0)) for s in profile["col_stats"] if "skew" in s]
    if skew_data:
        df_skew = pd.DataFrame(skew_data, columns=["Feature", "Skewness"]).sort_values("Skewness")
        fig_skew = px.bar(df_skew, x="Feature", y="Skewness", color="Skewness",
                          color_continuous_scale="RdBu", color_continuous_midpoint=0,
                          title="Skewness by Feature (|>1| = Highly Skewed)")
        fig_skew.update_layout(coloraxis_showscale=False)
        _apply_theme(fig_skew)
        st.plotly_chart(fig_skew, use_container_width=True)


# ──────────────────────────────────────────────
# TAB 5: AUTOML
# ──────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>🤖 AutoML Prediction Engine</div>", unsafe_allow_html=True)

    if not automl_enabled:
        st.info("AutoML is disabled. Enable it in the sidebar.")
    elif automl_result is None:
        st.warning("AutoML could not run — not enough data or no suitable numeric target.")
    elif "error" in automl_result:
        st.error(f"AutoML error: {automl_result['error']}")
    else:
        task_badge_color = "badge-blue" if automl_result["task"] == "classification" else "badge-purple"
        score_color = "#34d399" if automl_result["score"] >= 75 else ("#f59e0b" if automl_result["score"] >= 55 else "#f87171")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='font-size:0.8rem;color:#94a3b8;text-transform:uppercase;margin-bottom:6px'>Model Performance</div>
                <div style='font-size:3rem;font-weight:800;color:{score_color}'>{automl_result['score']}%</div>
                <div style='color:#64748b;font-size:0.85rem'>{automl_result['metric']}</div>
                <div style='margin-top:10px'><span class='badge {task_badge_color}'>{automl_result['task'].title()}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            top_3 = list(automl_result["importances"].items())[:3]
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='insight-title'>Top Predictors</div>", unsafe_allow_html=True)
            for feat, imp in top_3:
                pct = round(imp * 100, 1)
                bar_w = max(10, pct)
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                        <span style='color:#e2e8f0;font-size:0.88rem'>{feat}</span>
                        <span style='color:#63b3ed;font-weight:600;font-size:0.88rem'>{pct}%</span>
                    </div>
                    <div style='background:rgba(255,255,255,0.06);border-radius:4px;height:6px;'>
                        <div style='background:linear-gradient(90deg,#63b3ed,#9f7aea);border-radius:4px;height:6px;width:{bar_w}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance chart
        st.plotly_chart(chart_feature_importance(automl_result["importances"]), use_container_width=True)

        # Actual vs Predicted (regression only)
        if automl_result["task"] == "regression":
            try:
                num_cols_feat = automl_result["features"]
                X_all = df[num_cols_feat].dropna()
                y_all = df.loc[X_all.index, target].dropna()
                X_all = X_all.loc[y_all.index]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_all, y_all)
                preds = model.predict(X_all)
                df_pv = pd.DataFrame({"Actual": y_all.values, "Predicted": preds})
                fig_pv = px.scatter(df_pv, x="Actual", y="Predicted",
                                    color_discrete_sequence=["#63b3ed"],
                                    title="Actual vs Predicted")
                fig_pv.add_shape(type="line", x0=y_all.min(), x1=y_all.max(),
                                 y0=y_all.min(), y1=y_all.max(),
                                 line=dict(color="#f687b3", dash="dash"))
                _apply_theme(fig_pv)
                st.plotly_chart(fig_pv, use_container_width=True)
            except Exception:
                pass


# ──────────────────────────────────────────────
# TAB 6: SMART INSIGHTS (NARRATIVE)
# ──────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-header'>🧠 Smart Insights — AI-Generated Business Narrative</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.15);border-radius:10px;padding:14px 18px;margin-bottom:20px;font-size:0.88rem;color:#94a3b8'>
        ⚡ These insights are synthesised automatically from statistical analysis — no external AI APIs required.
        The engine interprets your data statistically and translates findings into business language.
    </div>
    """, unsafe_allow_html=True)

    for title, text in narrative:
        st.markdown(f"""
        <div class='glass-card'>
            <div class='section-header' style='font-size:1rem;margin-bottom:10px'>{title}</div>
            <div style='color:#cbd5e1;line-height:1.8;font-size:0.93rem'>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download report
    st.markdown("---")
    report_text = f"# Offline AI Analyst Report\n\nDataset: {uploaded_file.name}\nRows: {profile['rows']} | Columns: {profile['cols']}\n\n"
    for title, text in narrative:
        report_text += f"## {title}\n{text}\n\n"
    st.download_button(
        "📥 Download Insight Report (.txt)",
        data=report_text,
        file_name="ai_analyst_report.txt",
        mime="text/plain",
    )
